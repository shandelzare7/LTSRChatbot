// Chatbot Web Application JavaScript

let currentBotName = 'Chatbot';

// 获取当前会话状态（用于开场白等 UI）
async function fetchSessionStatus() {
    try {
        const response = await fetch('/api/session/status', {
            credentials: 'include',
        });
        if (!response.ok) return null;
        return await response.json();
    } catch (error) {
        console.error('获取会话状态失败:', error);
        return null;
    }
}

function buildFirstBotMessage(status) {
    const botName = (status && status.bot_name) ? status.bot_name : 'Chatbot';
    currentBotName = botName || 'Chatbot';
    const basicInfo = (status && status.bot_basic_info) ? status.bot_basic_info : {};
    const age = basicInfo.age;
    const occupation = basicInfo.occupation;

    // 若有历史（但当前页面不渲染历史），给一个“继续聊”的开场，仍由 bot 发起
    if (status && status.has_history) {
        return `欢迎回来，我是${botName}。我们继续聊吧：你现在最想聊什么？也可以先说说你此刻的心情。`;
    }

    const parts = [];
    if (occupation) parts.push(`职业是${occupation}`);
    if (age) parts.push(`今年${age}岁`);
    const introTail = parts.length ? `我${parts.join('，')}。` : '';

    return `你好，我是${botName}。${introTail}我可以陪你聊天、倾听，或者一起梳理想法。你也可以先简单介绍一下你自己吗？（昵称/想聊的话题/此刻的心情都可以）`;
}

async function ensureFirstBotMessage() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    if (chatMessages.children && chatMessages.children.length > 0) return;

    const status = await fetchSessionStatus();
    addMessage('bot', buildFirstBotMessage(status));
}

function _notifyEnabled() {
    try {
        return localStorage.getItem('ltsr_notify_enabled') === '1';
    } catch (e) {
        return false;
    }
}

function _pushEnabled() {
    try {
        return localStorage.getItem('ltsr_push_enabled') === '1';
    } catch (e) {
        return false;
    }
}

function _setPushEnabled(v) {
    try {
        localStorage.setItem('ltsr_push_enabled', v ? '1' : '0');
    } catch (e) {}
}

function _setNotifyEnabled(v) {
    try {
        localStorage.setItem('ltsr_notify_enabled', v ? '1' : '0');
    } catch (e) {}
}

function _shouldNotifyNow() {
    try {
        if (document.visibilityState && document.visibilityState !== 'visible') return true;
        if (typeof document.hasFocus === 'function' && !document.hasFocus()) return true;
    } catch (e) {}
    return false;
}

async function _registerServiceWorkerIfPossible() {
    try {
        if (!('serviceWorker' in navigator)) return null;
        // SW must be at /sw.js to cover '/' scope.
        await navigator.serviceWorker.register('/sw.js');
        return await navigator.serviceWorker.ready;
    } catch (e) {
        return null;
    }
}

async function maybeNotifyBotMessage(text) {
    try {
        // If Web Push is enabled, do NOT do local Notification here (avoid duplicates).
        if (_pushEnabled()) return;
        if (!_notifyEnabled()) return;
        if (!('Notification' in window)) return;
        if (Notification.permission !== 'granted') return;
        if (!_shouldNotifyNow()) return;

        const body = String(text || '').trim();
        if (!body) return;

        // Prefer SW showNotification (more reliable in background tabs).
        const reg = await _registerServiceWorkerIfPossible();
        if (reg && reg.showNotification) {
            await reg.showNotification(currentBotName || 'Chatbot', {
                body,
                tag: 'ltsr-bot-message',
                renotify: false,
            });
            return;
        }
        // Fallback: direct Notification.
        // eslint-disable-next-line no-new
        new Notification(currentBotName || 'Chatbot', { body });
    } catch (e) {
        // best-effort: never break chat flow
    }
}

// 自动请求推送权限（隐藏按钮，直接弹出权限请求）
async function autoRequestNotificationPermission() {
    try {
        const supportsLocal = ('Notification' in window);
        const supportsPush = ('serviceWorker' in navigator) && ('PushManager' in window);
        
        // 如果都不支持，直接返回
        if (!supportsLocal && !supportsPush) {
            return;
        }
        
        // 如果已经授权，直接返回
        if (Notification.permission === 'granted') {
            // 如果支持推送但还没订阅，尝试订阅
            if (supportsPush && !_pushEnabled()) {
                await _trySubscribePush();
            }
            return;
        }
        
        // 如果被拒绝，不自动请求（避免骚扰用户）
        if (Notification.permission === 'denied') {
            return;
        }
        
        // 先注册 Service Worker（不触发权限请求）
        const reg = await _registerServiceWorkerIfPossible();
        
        // 延迟一下再请求权限（避免页面加载时立即弹出，给用户一点时间）
        setTimeout(async () => {
            try {
                // 优先尝试 Web Push
                if (supportsPush && reg && reg.pushManager) {
                    const ok = await _trySubscribePush();
                    if (ok) return;
                }
                
                // 回退：本地通知
                if (supportsLocal) {
                    const perm = await Notification.requestPermission();
                    if (perm === 'granted') {
                        _setNotifyEnabled(true);
                    }
                }
            } catch (e) {
                // 忽略错误
            }
        }, 1000); // 延迟1秒，让页面先加载完成
    } catch (e) {
        // 忽略错误
    }
}

// 尝试订阅推送（内部辅助函数）
async function _trySubscribePush() {
    try {
        const supportsLocal = ('Notification' in window);
        const supportsPush = ('serviceWorker' in navigator) && ('PushManager' in window);
        if (!supportsPush) return false;
        
        // 请求通知权限
        const perm = supportsLocal ? await Notification.requestPermission() : 'granted';
        if (perm !== 'granted') return false;
        
        const reg = await _registerServiceWorkerIfPossible();
        if (!reg || !reg.pushManager) return false;
        
        // 检查是否已经订阅
        const existingSub = await reg.pushManager.getSubscription();
        if (existingSub) {
            // 已经订阅，保存状态
            _setPushEnabled(true);
            return true;
        }
        
        // 获取 VAPID 公钥并订阅
        const pub = await getVapidPublicKey();
        const sub = await reg.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: urlBase64ToUint8Array(pub),
        });
        
        const payload = { subscription: sub.toJSON ? sub.toJSON() : sub };
        const r = await fetch('/api/push/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload),
        });
        if (!r.ok) return false;
        
        _setPushEnabled(true);
        _setNotifyEnabled(false); // 推送启用时禁用本地通知
        return true;
    } catch (e) {
        return false;
    }
}

// VAPID 公钥获取和转换函数（从 setupNotificationButton 中提取）
function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) outputArray[i] = rawData.charCodeAt(i);
    return outputArray;
}

async function getVapidPublicKey() {
    const resp = await fetch('/api/push/public-key', { credentials: 'include' });
    if (!resp.ok) throw new Error('get public key failed');
    const data = await resp.json();
    if (!data || !data.public_key) throw new Error('missing public key');
    return String(data.public_key);
}

async function setupNotificationButton() {
    // 按钮已隐藏，但保留函数以防将来需要
    const btn = document.getElementById('notify-btn');
    if (btn) {
        btn.style.display = 'none';
    }
    // 这个函数现在不再使用，推送权限由 autoRequestNotificationPermission 自动请求

    async function subscribePush() {
        if (!supportsPush) return false;
        const perm = supportsLocal ? await Notification.requestPermission() : 'granted';
        if (perm !== 'granted') return false;
        const reg = await _registerServiceWorkerIfPossible();
        if (!reg || !reg.pushManager) return false;
        const pub = await getVapidPublicKey();
        const sub = await reg.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: urlBase64ToUint8Array(pub),
        });
        const payload = { subscription: sub.toJSON ? sub.toJSON() : sub };
        const r = await fetch('/api/push/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload),
        });
        if (!r.ok) throw new Error('subscribe api failed');
        return true;
    }

    async function unsubscribePush() {
        if (!supportsPush) return;
        const reg = await _registerServiceWorkerIfPossible();
        if (!reg || !reg.pushManager) return;
        const sub = await reg.pushManager.getSubscription();
        if (sub) {
            try { await sub.unsubscribe(); } catch (e) {}
        }
        try {
            await fetch('/api/push/unsubscribe', { method: 'POST', credentials: 'include' });
        } catch (e) {}
    }

    const refresh = () => {
        if (_pushEnabled()) {
            btn.textContent = '推送已开启';
            return;
        }
        if (!_notifyEnabled()) {
            btn.textContent = supportsPush ? '开启推送' : '开启通知';
            return;
        }
        if (!supportsLocal) {
            btn.textContent = '开启通知';
            return;
        }
        const perm = Notification.permission;
        if (perm === 'denied') btn.textContent = '通知被禁用';
        else if (perm === 'granted') btn.textContent = '通知已开启';
        else btn.textContent = '点击授权通知';
    };

    btn.onclick = async () => {
        try {
            // Toggle off push if already enabled
            if (_pushEnabled()) {
                _setPushEnabled(false);
                await unsubscribePush();
                refresh();
                return;
            }

            // Prefer Web Push when supported
            if (supportsPush) {
                const ok = await subscribePush();
                if (ok) {
                    _setPushEnabled(true);
                    // When push is enabled, disable local notifications to avoid duplicates.
                    _setNotifyEnabled(false);
                    refresh();
                    return;
                }
            }

            // Fallback: local notification
            if (!supportsLocal) return;
            const perm = await Notification.requestPermission();
            if (perm === 'granted') {
                _setNotifyEnabled(true);
                refresh();
            } else {
                _setNotifyEnabled(false);
                refresh();
            }
        } catch (e) {
            // ignore
        }
    };

    refresh();
}

async function fetchChatHistory(limit = 2000) {
    try {
        const response = await fetch(`/api/chat/history?limit=${encodeURIComponent(limit)}`, {
            credentials: 'include',
        });
        if (!response.ok) return [];
        const data = await response.json();
        if (!data || data.status !== 'success' || !Array.isArray(data.messages)) return [];
        return data.messages;
    } catch (error) {
        console.error('获取聊天历史失败:', error);
        return [];
    }
}

async function loadAndRenderChatHistory() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return 0;

    const history = await fetchChatHistory(5000);
    if (!history || history.length === 0) return 0;

    // clear any existing UI messages (should be empty on first load)
    chatMessages.innerHTML = '';

    history.forEach(m => {
        const role = (m && m.role) ? String(m.role) : '';
        const content = (m && m.content) ? String(m.content) : '';
        const createdAt = (m && m.created_at) ? String(m.created_at) : null;
        if (!content) return;
        if (role === 'user') addMessage('user', content, { timestamp: createdAt });
        else if (role === 'ai') addMessage('bot', content, { timestamp: createdAt });
        // ignore system messages in UI
    });

    return history.length;
}

// 加载bot列表
async function loadBots() {
    const botList = document.getElementById('bot-list');
    if (!botList) return;
    
    try {
        const response = await fetch('/api/bots');
        const data = await response.json();
        
        if (data.bots && data.bots.length > 0) {
            botList.innerHTML = '';
            data.bots.forEach(bot => {
                const botCard = document.createElement('div');
                botCard.className = 'bot-card';
                
                const name = bot.name || 'Unnamed Bot';
                const basicInfo = bot.basic_info || {};
                const age = basicInfo.age || '未知';
                const occupation = basicInfo.occupation || '未知';
                
                botCard.innerHTML = `
                    <div class="bot-card-content">
                        <div class="bot-name">${name}</div>
                        <div class="bot-info">年龄: ${age} | 职业: ${occupation}</div>
                    </div>
                    <div class="bot-card-actions">
                        <button class="btn-select" onclick="selectBot('${bot.id}')">开始对话</button>
                        <button class="btn-share" onclick="showShareDialog('${bot.id}', '${name}')" title="分享链接">🔗</button>
                    </div>
                `;
                botList.appendChild(botCard);
            });
        } else {
            botList.innerHTML = '<div class="error">暂无可用的 Chatbot</div>';
        }
    } catch (error) {
        console.error('加载bot列表失败:', error);
        botList.innerHTML = '<div class="error">加载失败，请刷新页面重试</div>';
    }
}

// 选择bot
async function selectBot(botId) {
    try {
        const response = await fetch('/api/session/init', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include', // 重要：包含Cookie
            body: JSON.stringify({ bot_id: botId }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            alert('初始化会话失败: ' + (error.detail || '未知错误'));
            return;
        }
        
        const data = await response.json();
        if (data.status === 'ready') {
            // 刷新页面进入聊天界面
            window.location.href = '/';
        }
    } catch (error) {
        console.error('选择bot失败:', error);
        alert('选择bot失败，请重试');
    }
}

// 生成分享链接
async function generateShareLink(botId) {
    try {
        const response = await fetch(`/api/share-link/${botId}`);
        if (!response.ok) {
            throw new Error('生成分享链接失败');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('生成分享链接失败:', error);
        return null;
    }
}

// 显示分享链接对话框
function showShareDialog(botId, botName) {
    generateShareLink(botId).then(data => {
        if (!data) {
            alert('生成分享链接失败');
            return;
        }
        
        const shareLink = data.share_link;
        const qrCodeUrl = data.qr_code_url;
        
        // 创建对话框
        const dialog = document.createElement('div');
        dialog.className = 'share-dialog-overlay';
        dialog.innerHTML = `
            <div class="share-dialog">
                <div class="share-dialog-header">
                    <h3>分享 ${botName}</h3>
                    <button class="close-btn" onclick="this.closest('.share-dialog-overlay').remove()">×</button>
                </div>
                <div class="share-dialog-content">
                    <div class="share-link-container">
                        <input type="text" id="share-link-input" value="${shareLink}" readonly class="share-link-input">
                        <button class="btn-copy" onclick="copyShareLink()">复制</button>
                    </div>
                    <div class="qr-code-container">
                        <img src="${qrCodeUrl}" alt="QR Code" class="qr-code">
                        <p class="qr-hint">扫描二维码访问</p>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(dialog);
        
        // 点击外部关闭
        dialog.onclick = (e) => {
            if (e.target === dialog) {
                dialog.remove();
            }
        };
    });
}

// 复制分享链接
function copyShareLink() {
    const input = document.getElementById('share-link-input');
    if (input) {
        input.select();
        document.execCommand('copy');
        const btn = document.querySelector('.btn-copy');
        if (btn) {
            const originalText = btn.textContent;
            btn.textContent = '已复制!';
            setTimeout(() => {
                btn.textContent = originalText;
            }, 2000);
        }
    }
}

// 初始化聊天界面
function initChat() {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    
    if (!messageInput || !sendBtn) return;

    // 自动请求推送权限（隐藏按钮，直接弹出权限请求）
    autoRequestNotificationPermission().catch(() => {});

    // 先加载历史（如果有），再决定是否插入开场白
    loadAndRenderChatHistory()
        .then(() => ensureFirstBotMessage())
        .catch(() => ensureFirstBotMessage().catch(() => {}));
    
    // 发送消息
    const sendMessage = async () => {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // 显示用户消息
        const localIso = new Date().toISOString();
        const userMsgId = addMessage('user', message, { timestamp: localIso });
        messageInput.value = '';
        // 不要在等待期间禁用按钮：后端支持“中断上一轮并合并/重启”，
        // 因此用户应当可以在等待时继续通过按钮或 Enter 发送下一条消息。
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({ message: message }),
            });
            
            if (!response.ok) {
                let errorDetail = '未知错误';
                try {
                    const error = await response.json();
                    // Superseded is a normal flow when user sends again quickly.
                    if (error && error.status === 'superseded') return;
                    if (error && typeof error.detail === 'string' && error.detail.includes('superseded')) return;
                    errorDetail = error.detail || `HTTP ${response.status}: ${response.statusText}`;
                } catch (parseError) {
                    // 如果无法解析 JSON，使用状态码和状态文本
                    errorDetail = `HTTP ${response.status}: ${response.statusText}`;
                    console.error('无法解析错误响应:', parseError);
                }
                console.error('服务器错误:', response.status, errorDetail);
                addMessage('bot', `服务器错误 (${response.status}): ${errorDetail}`);
                return;
            }
            
            const data = await response.json();
            // Superseded is NOT a failure: ignore this response.
            if (data && data.status === 'superseded') return;
            if (data.status === 'success') {
                if (userMsgId && data.user_created_at) {
                    setMessageTimestamp(userMsgId, data.user_created_at);
                }
                const segments = Array.isArray(data.segments) ? data.segments : [];
                if (segments.length >= 1) {
                    // 第一条消息：立即显示（不应用任何 delay）
                    const firstSeg = segments[0];
                    const firstContent = typeof firstSeg === 'string' ? firstSeg : (firstSeg.content || firstSeg);
                    addMessage('bot', firstContent, { timestamp: data.ai_created_at || new Date().toISOString() });
                    // Notify only once per bot turn (use the first segment).
                    maybeNotifyBotMessage(firstContent).catch(() => {});
                    
                    // 后续消息：只对 action === "typing" 的 segment 应用打字 delay
                    let cumulativeDelayMs = 0;
                    const DEFAULT_TYPING_DELAY_MS = 800; // 如果后端没有提供 delay，使用默认值
                    
                    for (let i = 1; i < segments.length; i++) {
                        const seg = segments[i];
                        const content = typeof seg === 'string' ? seg : (seg.content || seg);
                        const action = typeof seg === 'object' && seg !== null ? (seg.action || 'typing') : 'typing';
                        
                        // 只对 action === "typing" 的 segment 应用打字 delay
                        if (action === 'typing') {
                            // 获取 delay（秒），转换为毫秒
                            let delayMs = DEFAULT_TYPING_DELAY_MS;
                            if (typeof seg === 'object' && seg !== null && typeof seg.delay === 'number') {
                                delayMs = Math.max(0, seg.delay * 1000); // 秒转毫秒，确保非负
                            }
                            cumulativeDelayMs += delayMs;
                            
                            setTimeout(() => {
                                addMessage('bot', content);
                                const el = document.getElementById('chat-messages');
                                if (el) el.scrollTop = el.scrollHeight;
                            }, cumulativeDelayMs);
                        } else {
                            // action !== "typing"（如 "idle"）：立即显示，不累积 delay
                            setTimeout(() => {
                                addMessage('bot', content);
                                const el = document.getElementById('chat-messages');
                                if (el) el.scrollTop = el.scrollHeight;
                            }, cumulativeDelayMs);
                        }
                    }
                } else {
                    addMessage('bot', data.reply, { timestamp: data.ai_created_at || new Date().toISOString() });
                    maybeNotifyBotMessage(data.reply).catch(() => {});
                }
            } else {
                addMessage('bot', '回复失败');
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            // 显示详细的错误信息
            let errorMsg = '网络错误';
            if (error instanceof TypeError && error.message.includes('fetch')) {
                errorMsg = `网络连接失败: ${error.message}`;
            } else if (error instanceof Error) {
                errorMsg = `错误: ${error.name} - ${error.message}`;
                if (error.stack) {
                    console.error('错误堆栈:', error.stack);
                }
            } else if (typeof error === 'string') {
                errorMsg = `错误: ${error}`;
            } else {
                errorMsg = `网络错误: ${JSON.stringify(error)}`;
            }
            addMessage('bot', errorMsg + '，请重试');
        } finally {
            messageInput.focus();
        }
    };
    
    // 绑定事件
    sendBtn.onclick = sendMessage;
    messageInput.onkeypress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };
    
    // 聚焦输入框
    messageInput.focus();
}

// 添加消息到聊天界面
let messageIdCounter = 0;
function formatTimestamp(isoString) {
    try {
        if (!isoString) return '';
        const d = new Date(isoString);
        if (Number.isNaN(d.getTime())) return '';
        // 日期 + 时:分:秒（本地时间）
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const h = String(d.getHours()).padStart(2, '0');
        const min = String(d.getMinutes()).padStart(2, '0');
        const sec = String(d.getSeconds()).padStart(2, '0');
        return `${y}-${m}-${day} ${h}:${min}:${sec}`;
    } catch (e) {
        return '';
    }
}

function setMessageTimestamp(messageId, isoString) {
    const message = document.getElementById(messageId);
    if (!message) return;
    const t = message.querySelector('.message-time');
    if (!t) return;
    const s = formatTimestamp(isoString);
    if (!s) return;
    t.textContent = s;
    t.dataset.iso = String(isoString);
}

function addMessage(role, content, options = {}) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return null;
    
    const isTemporary = !!options.isTemporary;
    const messageId = isTemporary ? `temp-${Date.now()}` : `msg-${messageIdCounter++}`;
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `message message-${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    const timeText = formatTimestamp(options.timestamp);
    timeDiv.textContent = timeText;
    if (options.timestamp) timeDiv.dataset.iso = String(options.timestamp);
    messageDiv.appendChild(timeDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// 移除消息
function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}
