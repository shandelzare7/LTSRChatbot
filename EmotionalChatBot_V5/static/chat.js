// Chatbot Web Application JavaScript

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
    const resetBtn = document.getElementById('reset-btn');
    
    if (!messageInput || !sendBtn) return;

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
                const error = await response.json();
                addMessage('bot', '错误: ' + (error.detail || '未知错误'));
                return;
            }
            
            const data = await response.json();
            if (data.status === 'success') {
                if (userMsgId && data.user_created_at) {
                    setMessageTimestamp(userMsgId, data.user_created_at);
                }
                const segments = Array.isArray(data.segments) ? data.segments : [];
                if (segments.length >= 1) {
                    addMessage('bot', segments[0], { timestamp: data.ai_created_at || new Date().toISOString() });
                    const TYPING_DELAY_MS = 800;
                    for (let i = 1; i < segments.length; i++) {
                        const seg = segments[i];
                        setTimeout(() => {
                            addMessage('bot', seg);
                            const el = document.getElementById('chat-messages');
                            if (el) el.scrollTop = el.scrollHeight;
                        }, TYPING_DELAY_MS * i);
                    }
                } else {
                    addMessage('bot', data.reply, { timestamp: data.ai_created_at || new Date().toISOString() });
                }
            } else {
                addMessage('bot', '回复失败');
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            addMessage('bot', '网络错误，请重试');
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
    
    // 重置会话
    if (resetBtn) {
        resetBtn.onclick = async () => {
            if (!confirm('确定要清空历史吗？这会从数据库删除你在该 bot 下的所有聊天记录（不可恢复）。')) {
                return;
            }
            
            try {
                const response = await fetch('/api/session/reset', {
                    method: 'POST',
                    credentials: 'include',
                });
                
                if (response.ok) {
                    const chatMessages = document.getElementById('chat-messages');
                    if (chatMessages) {
                        chatMessages.innerHTML = '';
                    }
                    // 重置后也显示开场白
                    ensureFirstBotMessage().catch(() => {});
                    alert('历史已清空');
                } else {
                    alert('重置失败');
                }
            } catch (error) {
                console.error('重置会话失败:', error);
                alert('重置失败，请重试');
            }
        };
    }
    
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
