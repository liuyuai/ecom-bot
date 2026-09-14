"""
项目命令封装 - 用简单命令替代复杂操作

用法：
    python run.py dev          # 启动开发服务器
    python run.py test         # 运行评估测试
    python run.py build        # 重建向量库
    python run.py docker-up    # Docker 构建并启动
    python run.py docker-down  # Docker 停止
    python run.py docker-logs  # Docker 实时日志
    python run.py clean        # 清理缓存和运行时数据
    python run.py              # 显示帮助
"""
import sys
import os
import subprocess

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def run(cmd, cwd=None, shell=False):
    """执行命令并实时输出"""
    print(f"\n{'='*50}")
    print(f"执行: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*50}\n")
    try:
        subprocess.run(cmd, cwd=cwd or ROOT, shell=shell)
    except KeyboardInterrupt:
        print("\n[已停止]")
    except FileNotFoundError as e:
        print(f"[错误] 命令不存在: {e}")


def cmd_dev():
    """启动开发服务器（自动打开浏览器）"""
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    run([sys.executable, "main.py"], cwd=SRC)


def cmd_test():
    """运行评估测试"""
    run([sys.executable, "tests/evaluate.py"], cwd=SRC)


def cmd_build():
    """重建向量库（知识库文档变更后执行）"""
    run([sys.executable, "rag/ingest.py"], cwd=SRC)


def cmd_docker_up():
    """Docker 构建并后台启动"""
    run(["docker-compose", "up", "-d", "--build"], cwd=ROOT)


def cmd_docker_down():
    """Docker 停止并删除容器"""
    run(["docker-compose", "down"], cwd=ROOT)


def cmd_docker_logs():
    """Docker 实时日志"""
    run(["docker-compose", "logs", "-f"], cwd=ROOT)


def cmd_clean():
    """清理缓存和运行时数据（不删除 docs 和源码）"""
    import shutil
    targets = [
        os.path.join(ROOT, "chroma_db"),
        os.path.join(ROOT, "logs"),
        os.path.join(ROOT, "checkpoints.db"),
        os.path.join(ROOT, "embed_cache.json"),
        os.path.join(ROOT, "__pycache__"),
        os.path.join(SRC, "__pycache__"),
    ]
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t, ignore_errors=True)
            print(f"已删除目录: {t}")
        elif os.path.isfile(t):
            os.remove(t)
            print(f"已删除文件: {t}")
    print("\n清理完成")


# 命令注册表
COMMANDS = {
    "dev": (cmd_dev, "启动开发服务器"),
    "test": (cmd_test, "运行评估测试"),
    "build": (cmd_build, "重建向量库"),
    "docker-up": (cmd_docker_up, "Docker 构建并启动"),
    "docker-down": (cmd_docker_down, "Docker 停止"),
    "docker-logs": (cmd_docker_logs, "Docker 实时日志"),
    "clean": (cmd_clean, "清理缓存和运行时数据"),
}


def print_help():
    print("\n电商客服机器人 - 命令工具\n")
    print("用法: python run.py <命令>\n")
    print("可用命令:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<14} {desc}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd_name = sys.argv[1]
    if cmd_name in COMMANDS:
        COMMANDS[cmd_name][0]()
    else:
        print(f"[错误] 未知命令: {cmd_name}")
        print_help()
        sys.exit(1)
