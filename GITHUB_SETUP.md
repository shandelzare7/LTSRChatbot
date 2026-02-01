# GitHub 同步指南

本地 Git 仓库已初始化并创建了初始提交。按照以下步骤将代码推送到 GitHub：

## 步骤 1: 在 GitHub 上创建新仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `LTSRChatbot` (或你喜欢的名字)
   - Description: "LangGraph Chatbot 流程实现"
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有）
4. 点击 "Create repository"

## 步骤 2: 连接本地仓库到 GitHub

在终端中执行以下命令（将 `YOUR_USERNAME` 替换为你的 GitHub 用户名）：

```bash
cd /Users/huangshenze/Downloads/LTSRChatbot

# 添加远程仓库（使用 HTTPS）
git remote add origin https://github.com/YOUR_USERNAME/LTSRChatbot.git

# 或者使用 SSH（如果你配置了 SSH key）
# git remote add origin git@github.com:YOUR_USERNAME/LTSRChatbot.git

# 推送代码到 GitHub
git branch -M main
git push -u origin main
```

## 步骤 3: 在另一台电脑上克隆仓库

在另一台电脑上，执行：

```bash
# 使用 HTTPS
git clone https://github.com/YOUR_USERNAME/LTSRChatbot.git

# 或使用 SSH
# git clone git@github.com:YOUR_USERNAME/LTSRChatbot.git

cd LTSRChatbot
pip install -r requirements.txt
```

## 日常使用

### 推送更改到 GitHub
```bash
git add .
git commit -m "描述你的更改"
git push
```

### 从 GitHub 拉取最新更改
```bash
git pull
```

## 注意事项

- 如果使用 HTTPS 推送，GitHub 可能需要 Personal Access Token 而不是密码
- 如果使用 SSH，确保已配置 SSH key
- 建议定期提交和推送，保持代码同步
