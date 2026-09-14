#!/bin/bash
set -e

cd /app/src

# 如果向量库不存在，先建库
if [ ! -d "/app/chroma_db" ] || [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
    echo "[启动] 向量库不存在，开始建库..."
    python rag/ingest.py
    echo "[启动] 建库完成"
else
    echo "[启动] 向量库已存在，跳过建库"
fi

# 启动服务
echo "[启动] 启动服务 0.0.0.0:${PORT:-8000}"
exec python main.py
