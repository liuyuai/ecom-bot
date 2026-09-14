# 电商客服机器人 - Dockerfile
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=prod

# 安装系统依赖（chromadb 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码和知识库文档
COPY src/ ./src/
COPY docs/ ./docs/

# 创建运行时目录
RUN mkdir -p /app/chroma_db /app/logs

# 暴露端口
EXPOSE 8000

# 启动脚本：先建库（如果不存在），再启动服务
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
