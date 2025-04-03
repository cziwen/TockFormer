#!/bin/bash

####################################################################################
# 🤖 setup.sh - 运行于 RunPod 容器中的初始化脚本
#
# ✅ 用途：
#   - 设置 SSH 权限，添加私钥到 ssh-agent
#   - 测试 SSH 是否能连接 GitHub（用于私有仓库拉取）
#   - 自动安装 requirements.txt 中的 Python 依赖
#
# ⚠️ 注意：
#   - 在运行本脚本之前，你必须先从本地使用 scp 上传 SSH 私钥到容器：
#     示例命令（在本地终端执行）：
#     scp -P <PORT> ~/.ssh/id_ed25519 root@<IP_ADDRESS>:/root/.ssh/
#
#   - 然后进入容器后，运行：
#     bash setup.sh
####################################################################################

echo "🔐 Step 1: 设置 .ssh 权限"
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519

echo "⚙️ Step 2: 启动 ssh-agent 并添加私钥"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

echo "🔗 Step 3: 测试 SSH 是否能连上 GitHub"
ssh -T git@github.com || {
    echo "❌ SSH 连接 GitHub 失败，请确认你已将本地公钥添加到 GitHub"
    exit 1
}

echo "📦 Step 4: 安装 Python 依赖"
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt 不存在，请确认路径正确"
fi

echo "✅ 环境准备完成！现在可以正常训练、git 操作啦 🚀"