// B 版聊天前端：紫色玻璃风界面，复用同一套 API

let currentBotName = 'Chatbot';

async function fetchSessionStatus() {
    try {
        const response = await fetch('/api/session/status', { credentials: 'include' });
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.error('获取会话状态失败:', e);
        return null;
    }
}

function buildFirstBotMessage(status) {
    const botName = (status && status.bot_name) ? status.bot_name : 'Chatbot';
    currentBotName = botName || 'Chatbot';
    const basicInfo = (status && status.bot_basic_info) ? status.bot_basic_info : {};
    const age = basicInfo.age;
    const occupation = basicInfo.occupation;
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
    if (!chatMessages || chatMessages.children.length > 0) return;
    const status = await fetchSessionStatus();
    addMessage('bot', buildFirstBotMessage(status));
}

function _notifyEnabled() {
    try { return localStorage.getItem('ltsr_notify_enabled') === '1'; } catch (e) { return false; }
}
function _pushEnabled() {
    try { return localStorage.getItem('ltsr_push_enabled') === '1'; } catch (e) { return false; }
}
function _setPushEnabled(v) {
    try { localStorage.setItem('ltsr_push_enabled', v ? '1' : '0'); } catch (e) {}
}
function _setNotifyEnabled(v) {
    try { localStorage.setItem('ltsr_notify_enabled', v ? '1' : '0'); } catch (e) {}
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
        await navigator.serviceWorker.register('/sw.js');
        return await navigator.serviceWorker.ready;
    } catch (e) { return null; }
}
async function maybeNotifyBotMessage(text) {
    try {
        if (_pushEnabled() || !_notifyEnabled() || !('Notification' in window) || Notification.permission !== 'granted' || !_shouldNotifyNow()) return;
        const body = String(text || '').trim();
        if (!body) return;
        const reg = await _registerServiceWorkerIfPossible();
        if (reg && reg.showNotification) {
            await reg.showNotification(currentBotName || 'Chatbot', { body, tag: 'ltsr-bot-message', renotify: false });
            return;
        }
        new Notification(currentBotName || 'Chatbot', { body });
    } catch (e) {}
}

async function fetchChatHistory(limit) {
    try {
        const response = await fetch(`/api/chat/history?limit=${encodeURIComponent(limit || 2000)}`, { credentials: 'include' });
        if (!response.ok) return [];
        const data = await response.json();
        if (!data || data.status !== 'success' || !Array.isArray(data.messages)) return [];
        return data.messages;
    } catch (e) {
        console.error('获取聊天历史失败:', e);
        return [];
    }
}

function formatTimestamp(isoString) {
    try {
        if (!isoString) return '';
        const d = new Date(isoString);
        if (Number.isNaN(d.getTime())) return '';
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const h = String(d.getHours()).padStart(2, '0');
        const min = String(d.getMinutes()).padStart(2, '0');
        const sec = String(d.getSeconds()).padStart(2, '0');
        return `${y}-${m}-${day} ${h}:${min}:${sec}`;
    } catch (e) { return ''; }
}

let messageIdCounter = 0;

function setMessageTimestamp(messageId, isoString) {
    const row = document.getElementById(messageId);
    if (!row) return;
    const meta = row.querySelector('.msg-meta');
    if (!meta) return;
    const s = formatTimestamp(isoString);
    if (!s) return;
    meta.textContent = s;
    meta.dataset.iso = String(isoString);
}

function segmentToDisplayString(seg) {
    if (typeof seg === 'string') return seg;
    if (seg && typeof seg === 'object' && seg !== null) {
        var c = seg.content;
        if (typeof c === 'string') return c;
        if (c != null) return String(c);
    }
    return String(seg);
}

function addMessage(role, content, options) {
    options = options || {};
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return null;
    const isTemporary = !!options.isTemporary;
    const messageId = isTemporary ? `temp-${Date.now()}` : `msg-${messageIdCounter++}`;
    const row = document.createElement('div');
    row.id = messageId;
    row.className = `b-msg-row ${role === 'user' ? 'msg-me' : 'msg-bot'}`;
    const bubble = document.createElement('div');
    bubble.className = `msg msg--${role === 'user' ? 'me' : 'bot'}`;
    bubble.textContent = typeof content === 'string' ? content : segmentToDisplayString(content);
    row.appendChild(bubble);
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = formatTimestamp(options.timestamp);
    if (options.timestamp) meta.dataset.iso = String(options.timestamp);
    row.appendChild(meta);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageId;
}

async function loadAndRenderChatHistory() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return 0;
    const history = await fetchChatHistory(5000);
    if (!history || history.length === 0) return 0;
    chatMessages.innerHTML = '';
    history.forEach(function (m) {
        const role = (m && m.role) ? String(m.role) : '';
        const content = (m && m.content) ? String(m.content) : '';
        const createdAt = (m && m.created_at) ? String(m.created_at) : null;
        if (!content) return;
        if (role === 'user') addMessage('user', content, { timestamp: createdAt });
        else if (role === 'ai') addMessage('bot', content, { timestamp: createdAt });
    });
    return history.length;
}

// 从 basic_info 或占位生成 2～3 个 chip 文案
function chipsForBot(bot) {
    const basicInfo = bot.basic_info || {};
    const list = [];
    if (basicInfo.occupation) list.push(basicInfo.occupation);
    if (basicInfo.age) list.push(basicInfo.age + '岁');
    if (list.length >= 2) return list.slice(0, 3);
    const placeholders = ['温柔', '倾听', '建议型'];
    placeholders.forEach(function (p) {
        if (list.length < 3 && list.indexOf(p) === -1) list.push(p);
    });
    return list.slice(0, 3);
}

// 头像首字（或 emoji）
function avatarLetter(name) {
    if (!name || typeof name !== 'string') return '?';
    const trimmed = name.trim();
    if (!trimmed) return '?';
    return trimmed[0];
}

async function loadBots() {
    const botList = document.getElementById('bot-list');
    if (!botList) return;
    try {
        const response = await fetch('/api/bots', { credentials: 'include' });
        const data = await response.json();
        if (data.bots && data.bots.length > 0) {
            botList.innerHTML = '';
            data.bots.forEach(function (bot) {
                const name = bot.name || 'Unnamed Bot';
                const basicInfo = bot.basic_info || {};
                const age = basicInfo.age || '未知';
                const occupation = basicInfo.occupation || '未知';
                const chips = chipsForBot(bot);
                const card = document.createElement('div');
                card.className = 'bot-card';
                card.innerHTML =
                    '<div class="bot-card-top">' +
                    '<div class="avatar">' + escapeHtml(avatarLetter(name)) + '</div>' +
                    '<div class="bot-card-body">' +
                    '<div class="bot-card-name">' + escapeHtml(name) + '</div>' +
                    '<div class="bot-card-info">年龄: ' + escapeHtml(String(age)) + ' | 职业: ' + escapeHtml(String(occupation)) + '</div>' +
                    '<div class="bot-card-chips">' +
                    chips.map(function (c) { return '<span class="chip">' + escapeHtml(c) + '</span>'; }).join('') +
                    '</div>' +
                    '</div></div>' +
                    '<div class="bot-card-actions">' +
                    '<button class="btn-primary" onclick="selectBot(\'' + escapeHtml(bot.id) + '\')">开始对话</button>' +
                    '<button class="icon-btn" type="button" onclick="showShareDialog(\'' + escapeHtml(bot.id) + '\', \'' + escapeHtml(name).replace(/'/g, "\\'") + '\')" title="分享链接（生成可分享的对话链接）" aria-label="分享链接">🔗</button>' +
                    '</div>';
                botList.appendChild(card);
            });
        } else {
            botList.innerHTML = '<div class="error">暂无可用的 Chatbot</div>';
        }
    } catch (e) {
        console.error('加载bot列表失败:', e);
        botList.innerHTML = '<div class="error">加载失败，请刷新页面重试</div>';
    }
}

function escapeHtml(s) {
    if (s == null) return '';
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

async function selectBot(botId) {
    try {
        const response = await fetch('/api/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ bot_id: botId }),
        });
        if (!response.ok) {
            const err = await response.json();
            alert('初始化会话失败: ' + (err.detail || '未知错误'));
            return;
        }
        const data = await response.json();
        if (data.status === 'ready') {
            window.location.href = '/chat/' + encodeURIComponent(botId);
        }
    } catch (e) {
        console.error('选择bot失败:', e);
        alert('选择bot失败，请重试');
    }
}

async function resumeByUserId() {
    const input = document.getElementById('resume-user-id');
    if (!input) return;
    const userId = input.value.trim();
    if (!userId) {
        alert('请输入会话ID');
        return;
    }
    try {
        const response = await fetch('/api/session/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ user_db_id: userId }),
        });
        if (!response.ok) {
            const err = await response.json();
            alert('恢复会话失败: ' + (err.detail || '未知错误'));
            return;
        }
        const data = await response.json();
        if (data.status === 'ready' && data.bot_id) {
            window.location.href = '/chat/' + encodeURIComponent(data.bot_id);
        }
    } catch (e) {
        console.error('恢复会话失败:', e);
        alert('恢复会话失败，请重试');
    }
}

async function generateShareLink(botId) {
    try {
        const response = await fetch('/api/share-link/' + encodeURIComponent(botId), { credentials: 'include' });
        if (!response.ok) throw new Error('生成分享链接失败');
        return await response.json();
    } catch (e) {
        console.error('生成分享链接失败:', e);
        return null;
    }
}

function showShareDialog(botId, botName) {
    generateShareLink(botId).then(function (data) {
        if (!data) {
            alert('生成分享链接失败');
            return;
        }
        const shareLink = data.share_link;
        const qrCodeUrl = data.qr_code_url || '';
        const dialog = document.createElement('div');
        dialog.className = 'share-dialog-overlay';
        dialog.innerHTML =
            '<div class="share-dialog">' +
            '<div class="share-dialog-header"><h3>分享 ' + escapeHtml(botName) + '</h3><button class="close-btn" onclick="this.closest(\'.share-dialog-overlay\').remove()">×</button></div>' +
            '<div class="share-dialog-content">' +
            '<div class="share-link-container"><input type="text" id="share-link-input" value="' + escapeHtml(shareLink) + '" readonly class="share-link-input"><button class="btn-copy" onclick="copyShareLink()">复制</button></div>' +
            '<div class="qr-code-container"><img src="' + escapeHtml(qrCodeUrl) + '" alt="QR Code" class="qr-code"><p class="qr-hint">扫描二维码访问</p></div>' +
            '</div></div>';
        document.body.appendChild(dialog);
        dialog.onclick = function (e) {
            if (e.target === dialog) dialog.remove();
        };
    });
}

function copyShareLink() {
    const input = document.getElementById('share-link-input');
    if (!input) return;
    input.select();
    try {
        document.execCommand('copy');
    } catch (e) {}
    const btn = document.querySelector('.btn-copy');
    if (btn) {
        var orig = btn.textContent;
        btn.textContent = '已复制!';
        setTimeout(function () { btn.textContent = orig; }, 2000);
    }
}

async function autoRequestNotificationPermission() {
    try {
        if (!('Notification' in window)) return;
        if (Notification.permission === 'granted' || Notification.permission === 'denied') return;
        setTimeout(function () {
            Notification.requestPermission().catch(function () {});
        }, 1000);
    } catch (e) {}
}

function initChat() {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    if (!messageInput || !sendBtn) return;

    autoRequestNotificationPermission();

    // 聊天页头部：更新 bot 头像与名称
    fetchSessionStatus().then(function (status) {
        const botName = (status && status.bot_name) ? status.bot_name : 'Chatbot';
        currentBotName = botName;
        const headerAvatar = document.getElementById('chat-header-avatar');
        const headerTitle = document.getElementById('chat-header-title');
        if (headerAvatar) {
            headerAvatar.textContent = avatarLetter(botName);
        }
        if (headerTitle) {
            headerTitle.textContent = '对话中 · ' + botName;
        }
        // 会话 ID 展示
        if (status && status.user_db_id) {
            const el = document.getElementById('user-db-id');
            if (el) {
                el.innerHTML = '';
                var span = document.createElement('span');
                span.className = 'user-id-text';
                span.textContent = '会话ID: ' + status.user_db_id;
                el.appendChild(span);
                var copyBtn = document.createElement('button');
                copyBtn.type = 'button';
                copyBtn.className = 'user-id-copy';
                copyBtn.title = '复制会话ID';
                copyBtn.textContent = '📋';
                copyBtn.addEventListener('click', function () {
                    navigator.clipboard.writeText(status.user_db_id).then(function () {
                        copyBtn.title = '已复制';
                        copyBtn.textContent = '✓';
                        setTimeout(function () {
                            copyBtn.title = '复制会话ID';
                            copyBtn.textContent = '📋';
                        }, 1500);
                    }).catch(function () {
                        copyBtn.title = '复制失败';
                    });
                });
                el.appendChild(copyBtn);
            }
        }
    }).catch(function () {});

    loadAndRenderChatHistory()
        .then(function () { return ensureFirstBotMessage(); })
        .catch(function () { ensureFirstBotMessage().catch(function () {}); });

    function sendMessage() {
        var message = (messageInput.value || '').trim();
        if (!message) return;
        var localIso = new Date().toISOString();
        var userMsgId = addMessage('user', message, { timestamp: localIso });
        messageInput.value = '';

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ message: message }),
        })
            .then(function (response) {
                if (!response.ok) {
                    return response.json().then(function (err) {
                        if (err && (err.status === 'superseded' || (typeof err.detail === 'string' && err.detail.indexOf('superseded') !== -1))) return;
                        throw new Error(err.detail || 'HTTP ' + response.status);
                    }).catch(function (e) {
                        if (e.message && e.message.indexOf('superseded') !== -1) return;
                        addMessage('bot', '服务器错误: ' + (e.message || response.status));
                    });
                }
                return response.json();
            })
            .then(function (data) {
                if (!data) return;
                if (data.status === 'superseded') return;
                if (data.status === 'success') {
                    if (userMsgId && data.user_created_at) setMessageTimestamp(userMsgId, data.user_created_at);
                    var segments = Array.isArray(data.segments) ? data.segments : [];
                    if (segments.length >= 1) {
                        var firstSeg = segments[0];
                        var firstContent = segmentToDisplayString(firstSeg);
                        addMessage('bot', firstContent, { timestamp: data.ai_created_at || new Date().toISOString() });
                        maybeNotifyBotMessage(firstContent).catch(function () {});
                        var DEFAULT_TYPING_DELAY_MS = 3800;
                        var cumulativeDelayMs = 0;
                        for (var i = 1; i < segments.length; i++) {
                            var seg = segments[i];
                            var content = segmentToDisplayString(seg);
                            var action = (seg && typeof seg === 'object' && seg.action) ? seg.action : 'typing';
                            var delayMs = action === 'typing' ? (typeof seg.delay === 'number' ? Math.max(0, seg.delay * 1000) : DEFAULT_TYPING_DELAY_MS) : 0;
                            cumulativeDelayMs += delayMs;
                            (function (c, t) {
                                setTimeout(function () {
                                    addMessage('bot', c);
                                    var el = document.getElementById('chat-messages');
                                    if (el) el.scrollTop = el.scrollHeight;
                                }, t);
                            })(content, cumulativeDelayMs);
                        }
                    } else {
                        addMessage('bot', data.reply || '', { timestamp: data.ai_created_at || new Date().toISOString() });
                        maybeNotifyBotMessage(data.reply).catch(function () {});
                    }
                } else {
                    addMessage('bot', '回复失败');
                }
            })
            .catch(function (error) {
                console.error('发送消息失败:', error);
                addMessage('bot', '网络错误，请重试');
            })
            .finally(function () {
                messageInput.focus();
            });
    }

    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    messageInput.focus();
}
