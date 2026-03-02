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
    return '你好，我是' + botName;
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

function _urlBase64ToUint8Array(base64String) {
    var padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    var base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    var rawData = window.atob(base64);
    var outputArray = new Uint8Array(rawData.length);
    for (var i = 0; i < rawData.length; ++i) outputArray[i] = rawData.charCodeAt(i);
    return outputArray;
}
async function _getVapidPublicKey() {
    var resp = await fetch('/api/push/public-key', { credentials: 'include' });
    if (!resp.ok) throw new Error('get public key failed');
    var data = await resp.json();
    if (!data || !data.public_key) throw new Error('missing public key');
    return String(data.public_key);
}
/** 将当前会话的推送订阅同步到服务端（B 版必须调用，否则服务端没有订阅、非活动时收不到推送）。每次进入聊天页且已授权时调用。 */
async function syncPushSubscriptionToServer() {
    try {
        if (!('Notification' in window) || Notification.permission !== 'granted') return;
        if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;
        var reg = await _registerServiceWorkerIfPossible();
        if (!reg || !reg.pushManager) return;
        var sub = await reg.pushManager.getSubscription();
        if (!sub) {
            var pub = await _getVapidPublicKey();
            sub = await reg.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: _urlBase64ToUint8Array(pub),
            });
        }
        if (!sub) return;
        var payload = { subscription: sub.toJSON ? sub.toJSON() : sub };
        var r = await fetch('/api/push/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload),
        });
        if (r.ok) _setPushEnabled(true);
    } catch (e) {
        console.warn('syncPushSubscriptionToServer:', e);
    }
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

var _historyLoadInProgress = false;
async function loadAndRenderChatHistory() {
    if (_historyLoadInProgress) return 0;
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return 0;
    _historyLoadInProgress = true;
    try {
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
    } finally {
        _historyLoadInProgress = false;
    }
}

// 最多 2 个 chip，优先 persona（quirks/hobbies），文案简短不拥挤
// 预设人格标签（按 bot 名称），用于替换原有 chips 展示
var BOT_PERSONA_LABELS = {
    '林静怡': '【温和笃定的规范者】',
    '沈默言': '【严谨纠结的话痨】',
    '苏絮': '【冷漠怯场的脑洞派】',
    '谢凌锋': '【散漫嘴毒的吐槽狂】',
    '阿澈': '【盲目自信的直白大喇叭】',
    '陆燃': '【绝对自信的带刺修辞家】',
    '顾沉': '【冰冷笃定的直白机器】'
};

function personaLabelForBot(bot) {
    var name = (bot && bot.name) ? String(bot.name).trim() : '';
    return BOT_PERSONA_LABELS[name] || '';
}

function chipsForBot(bot) {
    const maxChips = 2;
    function short(s) {
        if (s == null || typeof s !== 'string') return '';
        var t = s.trim();
        return t.length > 6 ? t.slice(0, 6) : t;
    }
    const list = [];
    const persona = bot.persona || {};
    const collections = persona.collections || {};
    const quirks = [].concat(collections.quirks || []).slice(0, 1);
    const hobbies = [].concat(collections.hobbies || []).slice(0, 1);
    quirks.forEach(function (q) { if (short(q) && list.length < maxChips) list.push(short(q)); });
    hobbies.forEach(function (h) { if (short(h) && list.length < maxChips) list.push(short(h)); });
    if (list.length >= maxChips) return list;
    const basicInfo = bot.basic_info || {};
    if (basicInfo.occupation && list.indexOf(short(basicInfo.occupation)) === -1 && list.length < maxChips) list.push(short(basicInfo.occupation));
    if (list.length < maxChips && basicInfo.age && list.indexOf(basicInfo.age + '岁') === -1) list.push(basicInfo.age + '岁');
    if (list.length >= maxChips) return list;
    ['温柔', '倾听'].forEach(function (p) {
        if (list.length < maxChips && list.indexOf(p) === -1) list.push(p);
    });
    return list.slice(0, maxChips);
}

// 头像首字（或 emoji）
function avatarLetter(name) {
    if (!name || typeof name !== 'string') return '?';
    const trimmed = name.trim();
    if (!trimmed) return '?';
    return trimmed[0];
}

function renderBigFiveBars(bigFive) {
    if (!bigFive || typeof bigFive !== 'object') return '';
    var dims = [
        ['O 开放', bigFive.O != null ? bigFive.O : bigFive.openness],
        ['C 尽责', bigFive.C != null ? bigFive.C : bigFive.conscientiousness],
        ['E 外向', bigFive.E != null ? bigFive.E : bigFive.extraversion],
        ['A 宜人', bigFive.A != null ? bigFive.A : bigFive.agreeableness],
        ['N 神经', bigFive.N != null ? bigFive.N : bigFive.neuroticism],
    ];
    var rows = dims.map(function (d) {
        if (d[1] == null) return '';
        var pct = Math.round(Math.min(1, Math.max(0, d[1])) * 100);
        return '<div class="bf-row"><span class="bf-label">' + escapeHtml(d[0]) + '</span>' +
               '<div class="bf-track"><div class="bf-fill" style="width:' + pct + '%"></div></div>' +
               '<span class="bf-val">' + pct + '</span></div>';
    }).join('');
    return rows ? '<div class="big-five-bars">' + rows + '</div>' : '';
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
                var personaLabel = personaLabelForBot(bot);
                var chips = chipsForBot(bot);
                var tagOrChipsHtml = personaLabel
                    ? ('<div class="bot-card-tag">' + escapeHtml(personaLabel) + '</div>')
                    : ('<div class="bot-card-chips">' + chips.map(function (c) { return '<span class="chip">' + escapeHtml(c) + '</span>'; }).join('') + '</div>');
                const card = document.createElement('div');
                card.className = 'bot-card';
                card.innerHTML =
                    '<div class="bot-card-top">' +
                    '<div class="avatar">' + escapeHtml(avatarLetter(name)) + '</div>' +
                    '<div class="bot-card-body">' +
                    '<div class="bot-card-name">' + escapeHtml(name) + '</div>' +
                    '<div class="bot-card-info">年龄: ' + escapeHtml(String(age)) + ' | 职业: ' + escapeHtml(String(occupation)) + '</div>' +
                    tagOrChipsHtml +
                    renderBigFiveBars(bot.big_five) +
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
        const source = new URLSearchParams(window.location.search).get('source') || undefined;
        const response = await fetch('/api/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ bot_id: botId, source: source }),
        });
        if (!response.ok) {
            var detail = '未知错误';
            try {
                var err = await response.json();
                detail = err.detail || err.message || 'HTTP ' + response.status;
            } catch (_) {
                detail = 'HTTP ' + response.status + ': ' + response.statusText;
            }
            alert('初始化会话失败: ' + detail);
            return;
        }
        const data = await response.json();
        if (data.status === 'ready') {
            window.location.href = '/b/chat/' + encodeURIComponent(botId);
        }
    } catch (e) {
        console.error('选择bot失败:', e);
        var msg = e instanceof Error ? (e.name + ': ' + e.message) : String(e);
        alert('选择bot失败: ' + msg);
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
            var detail = '未知错误';
            try {
                var err = await response.json();
                detail = err.detail || err.message || 'HTTP ' + response.status;
            } catch (_) {
                detail = 'HTTP ' + response.status + ': ' + response.statusText;
            }
            alert('恢复会话失败: ' + detail);
            return;
        }
        const data = await response.json();
        if (data.status === 'ready' && data.bot_id) {
            window.location.href = '/b/chat/' + encodeURIComponent(data.bot_id);
        }
    } catch (e) {
        console.error('恢复会话失败:', e);
        var msg = e instanceof Error ? (e.name + ': ' + e.message) : String(e);
        alert('恢复会话失败: ' + msg);
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
        if (Notification.permission === 'granted') {
            syncPushSubscriptionToServer().catch(function () {});
            return;
        }
        if (Notification.permission === 'denied') return;
        setTimeout(function () {
            Notification.requestPermission()
                .then(function (result) {
                    if (result === 'granted') syncPushSubscriptionToServer().catch(function () {});
                })
                .catch(function () {});
        }, 1000);
    } catch (e) {}
}

function initChat() {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    if (!messageInput || !sendBtn) return;

    autoRequestNotificationPermission();
    if ('Notification' in window && Notification.permission === 'granted') syncPushSubscriptionToServer().catch(function () {});
    setTimeout(function () {
        if ('Notification' in window && Notification.permission === 'granted') syncPushSubscriptionToServer().catch(function () {});
    }, 2500);

    // 聊天页头部：更新 bot 头像与名称，并初始化状态面板
    fetchSessionStatus().then(function (status) {
        const botName = (status && status.bot_name) ? status.bot_name : 'Chatbot';
        currentBotName = botName;
        renderStatePanel(status);
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
                        var detail = err.detail || err.message || 'HTTP ' + response.status;
                        addMessage('bot', '服务器错误 (' + response.status + '): ' + detail);
                        return;
                    }).catch(function (e) {
                        if (e.message && e.message.indexOf('superseded') !== -1) return;
                        addMessage('bot', '服务器错误 (' + response.status + '): ' + (e.message || response.statusText || response.status));
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
                        // 后端未提供 delay 时默认 2–4 秒随机
                        var getDefaultTypingDelayMs = function () { return Math.round(2000 + Math.random() * 2000); };
                        var cumulativeDelayMs = 0;
                        for (var i = 1; i < segments.length; i++) {
                            var seg = segments[i];
                            var content = segmentToDisplayString(seg);
                            var action = (seg && typeof seg === 'object' && seg.action) ? seg.action : 'typing';
                            var delayMs = action === 'typing' ? (typeof seg.delay === 'number' ? Math.max(0, seg.delay * 1000) : getDefaultTypingDelayMs()) : 0;
                            cumulativeDelayMs += delayMs;
                            (function (c, t) {
                                setTimeout(function () {
                                    addMessage('bot', c);
                                    var el = document.getElementById('chat-messages');
                                    if (el) el.scrollTop = el.scrollHeight;
                                }, t);
                            })(content, cumulativeDelayMs);
                        }
                        // 所有 segment 播放完后刷新状态面板
                        setTimeout(function () {
                            fetchSessionStatus().then(renderStatePanel).catch(function () {});
                        }, cumulativeDelayMs + 200);
                    } else {
                        addMessage('bot', data.reply || '', { timestamp: data.ai_created_at || new Date().toISOString() });
                        maybeNotifyBotMessage(data.reply).catch(function () {});
                        fetchSessionStatus().then(renderStatePanel).catch(function () {});
                    }
                } else {
                    var failDetail = (data && (data.detail || data.message)) ? String(data.detail || data.message) : '';
                    addMessage('bot', failDetail ? '回复失败: ' + failDetail : '回复失败');
                }
            })
            .catch(function (error) {
                console.error('发送消息失败:', error);
                var errorMsg = '';
                if (error instanceof TypeError && error.message && error.message.indexOf('fetch') !== -1) {
                    errorMsg = '网络连接失败: ' + error.message;
                } else if (error instanceof Error) {
                    errorMsg = error.name + ': ' + error.message;
                } else if (typeof error === 'string') {
                    errorMsg = error;
                } else {
                    errorMsg = (error && error.message) ? String(error.message) : ('错误: ' + JSON.stringify(error));
                }
                addMessage('bot', (errorMsg || '未知错误') + '。请重试');
            })
            .finally(function () {
                messageInput.focus();
            });
    }

    // 手机端用 touchend 更可靠，避免 click 不触发；桌面端用 click；防重复触发
    var lastSendTime = 0;
    function sendMessageWithDebounce() {
        var now = Date.now();
        if (now - lastSendTime < 400) return;
        lastSendTime = now;
        sendMessage();
    }
    sendBtn.addEventListener('touchend', function (e) {
        e.preventDefault();
        sendMessageWithDebounce();
    }, { passive: false });
    sendBtn.addEventListener('click', sendMessageWithDebounce);
    messageInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    // 手机端：输入框获焦时把发送按钮滚入视区，避免被键盘挡住
    messageInput.addEventListener('focus', function () {
        if ('ontouchstart' in window) {
            setTimeout(function () {
                sendBtn.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }, 300);
        }
    });
    // 仅非触屏设备在加载时自动聚焦；手机端不自动 focus，否则键盘会立刻弹起并挡住发送按钮
    if (!('ontouchstart' in window)) messageInput.focus();

    // ── 状态抽屉 ──
    var panelBtn = document.getElementById('state-panel-btn');
    if (panelBtn) panelBtn.addEventListener('click', openStatePanel);
    var spCloseBtn = document.getElementById('state-panel-close');
    if (spCloseBtn) spCloseBtn.addEventListener('click', closeStatePanel);
    var spOverlay = document.getElementById('state-panel-overlay');
    if (spOverlay) spOverlay.addEventListener('click', closeStatePanel);
}

// ── 状态面板：开关 ──
function openStatePanel() {
    var panel = document.getElementById('state-panel');
    var overlay = document.getElementById('state-panel-overlay');
    if (panel) panel.classList.add('open');
    if (overlay) overlay.classList.add('open');
}
function closeStatePanel() {
    var panel = document.getElementById('state-panel');
    var overlay = document.getElementById('state-panel-overlay');
    if (panel) panel.classList.remove('open');
    if (overlay) overlay.classList.remove('open');
}

// ── 状态面板：渲染数据 ──
var _KNAPP_LABELS = {
    initiating: '起始 Initiating', experimenting: '探索 Experimenting', intensifying: '加深 Intensifying',
    integrating: '融合 Integrating', bonding: '承诺 Bonding', differentiating: '分化 Differentiating',
    circumscribing: '限缩 Circumscribing', stagnating: '停滞 Stagnating', avoiding: '回避 Avoiding', terminating: '结束 Terminating'
};
var _KNAPP_PHASE = {
    initiating: '接近期 Coming Together', experimenting: '接近期 Coming Together', intensifying: '接近期 Coming Together',
    integrating: '接近期 Coming Together', bonding: '接近期 Coming Together', differentiating: '疏远期 Coming Apart',
    circumscribing: '疏远期 Coming Apart', stagnating: '疏远期 Coming Apart', avoiding: '疏远期 Coming Apart', terminating: '疏远期 Coming Apart'
};
var _REL_DIMS = [
    ['closeness', '亲密 Closeness'], ['trust', '信任 Trust'], ['liking', '喜爱 Liking'],
    ['respect', '尊重 Respect'], ['attractiveness', '吸引 Attractiveness'], ['power', '主导 Power']
];
var _PADB_DIMS = [
    ['pleasure', 'P 愉悦'], ['arousal', 'Ar 激动'],
    ['dominance', 'D 掌控'], ['busyness', '忙碌']
];

function _makeBarsHtml(dims, data) {
    if (!data || typeof data !== 'object') return '';
    var rows = dims.map(function (d) {
        var v = data[d[0]];
        if (v == null) return '';
        // 兼容 m1_1 scale (-1~1)，统一转换到 0~1 显示
        var norm = (v < 0) ? (v + 1) / 2 : v;
        var pct = Math.round(Math.min(1, Math.max(0, norm)) * 100);
        return '<div class="bf-row">' +
               '<span class="bf-label" style="width:54px">' + escapeHtml(d[1]) + '</span>' +
               '<div class="bf-track"><div class="bf-fill" style="width:' + pct + '%"></div></div>' +
               '<span class="bf-val">' + pct + '</span></div>';
    }).join('');
    return rows || '';
}

function renderStatePanel(status) {
    if (!status) return;
    console.log('[StatePanel] status:', JSON.stringify({
        bot_big_five: status.bot_big_five,
        bot_mood_state: status.bot_mood_state,
        user_dimensions: status.user_dimensions,
        user_current_stage: status.user_current_stage,
    }));

    // 大五人格
    var bfEl = document.getElementById('sp-big-five');
    if (bfEl) {
        var bf = status.bot_big_five || {};
        var bfNorm = {
            openness: bf.O != null ? bf.O : bf.openness,
            conscientiousness: bf.C != null ? bf.C : bf.conscientiousness,
            extraversion: bf.E != null ? bf.E : bf.extraversion,
            agreeableness: bf.A != null ? bf.A : bf.agreeableness,
            neuroticism: bf.N != null ? bf.N : bf.neuroticism,
        };
        var bfDims = [
            ['openness', 'O 开放'], ['conscientiousness', 'C 尽责'],
            ['extraversion', 'E 外向'], ['agreeableness', 'A 宜人'],
            ['neuroticism', 'N 神经']
        ];
        bfEl.innerHTML = _makeBarsHtml(bfDims, bfNorm) || '暂无数据';
    }

    // PADB 情绪
    var padbEl = document.getElementById('sp-padb');
    if (padbEl) {
        padbEl.innerHTML = _makeBarsHtml(_PADB_DIMS, status.bot_mood_state || {}) || '暂无数据';
    }

    // 关系维度
    var relEl = document.getElementById('sp-relationship');
    if (relEl) {
        relEl.innerHTML = _makeBarsHtml(_REL_DIMS, status.user_dimensions || {}) || '暂无数据';
    }

    // Knapp 阶段
    var stageEl = document.getElementById('sp-stage');
    if (stageEl) {
        var s = status.user_current_stage;
        if (s) {
            stageEl.innerHTML =
                '<div class="knapp-badge">' + escapeHtml(_KNAPP_LABELS[s] || s) + '</div>' +
                (_KNAPP_PHASE[s] ? '<div class="knapp-phase">' + escapeHtml(_KNAPP_PHASE[s]) + '</div>' : '');
        } else {
            stageEl.textContent = '暂无数据';
        }
    }
}
