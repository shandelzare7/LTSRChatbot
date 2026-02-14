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

    // 开场白：让第一句由 chatbot 发起
    ensureFirstBotMessage().catch(() => {});
    
    // 发送消息
    const sendMessage = async () => {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // 显示用户消息
        addMessage('user', message);
        messageInput.value = '';
        sendBtn.disabled = true;
        
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
                addMessage('bot', data.reply);
            } else {
                addMessage('bot', '回复失败');
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            addMessage('bot', '网络错误，请重试');
        } finally {
            sendBtn.disabled = false;
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
            if (!confirm('确定要重置会话吗？这将清空所有对话历史。')) {
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
                    alert('会话已重置');
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
function addMessage(role, content, isTemporary = false) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return null;
    
    const messageId = isTemporary ? `temp-${Date.now()}` : `msg-${messageIdCounter++}`;
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `message message-${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);
    
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
