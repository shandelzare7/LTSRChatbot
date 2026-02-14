#!/usr/bin/env python3
"""
快速获取 Cloudflare Tunnel 临时链接
"""
import subprocess
import re
import sys
import time
import signal

def get_cloudflare_link():
    """启动 cloudflared 并提取临时链接"""
    print("🌐 正在启动 Cloudflare Tunnel...")
    print("   请确保 FastAPI 已在 http://127.0.0.1:8000 运行")
    print()
    
    # 启动 cloudflared
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    link_pattern = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')
    link_found = None
    
    try:
        # 读取输出，最多等待 30 秒
        for _ in range(30):
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.5)
                continue
            
            print(line.rstrip())
            
            # 查找链接
            match = link_pattern.search(line)
            if match:
                link_found = match.group(0)
                print()
                print("=" * 80)
                print(f"✅ Cloudflare 临时链接已生成:")
                print(f"   {link_found}")
                print("=" * 80)
                print()
                print("💡 提示:")
                print("   - 此链接在 cloudflared 进程运行期间有效")
                print("   - 停止 cloudflared (Ctrl+C) 后链接会失效")
                print("   - 要生成分享链接，运行: WEB_DOMAIN='你的链接' python generate_share_links.py")
                print()
                break
        
        if not link_found:
            print("⚠️  未能在 30 秒内检测到链接，请检查 cloudflared 输出")
            print("   你可以手动查看上面的输出来找到链接")
        
        # 保持运行，直到用户中断
        print("📌 Cloudflare Tunnel 正在运行中...")
        print("   按 Ctrl+C 停止")
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 正在停止 Cloudflare Tunnel...")
        process.terminate()
        process.wait()
        print("✅ 已停止")
    except Exception as e:
        print(f"❌ 错误: {e}")
        process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    # 检查 cloudflared 是否安装
    try:
        subprocess.run(["cloudflared", "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ cloudflared 未安装")
        print("   请先安装: brew install cloudflared (macOS)")
        print("   或访问: https://github.com/cloudflare/cloudflared/releases")
        sys.exit(1)
    
    get_cloudflare_link()
