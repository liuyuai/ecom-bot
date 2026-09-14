# 电商客服机器人

基于 LangChain + LangGraph 的企业级电商客服 AI 机器人，支持 RAG 知识库检索、Agent 工具调用、流式输出、多轮对话、HITL 人工确认等完整能力。

## 功能特性

- **RAG 知识库**：向量检索（Chroma）+ BM25 关键词检索 + Reranker 重排序，混合检索提升准确率
- **Agent 工具调用**：7 个工具（知识库查询、订单查询、商品搜索、删除订单、退款等），LangGraph 编排
- **HITL 人工确认**：删除/退款等危险操作两阶段执行，必须人工确认
- **流式输出**：SSE 逐字返回，前端实时渲染
- **多轮对话**：Redis（生产）/ SQLite（开发）会话存储 + 前端 localStorage 历史持久化
- **Query Rewrite**：多轮对话自动改写省略/代词问题
- **用户反馈 + 自主学习**：好评问答自动加入知识库，系统越用越准
- **安全防护**：Prompt 注入检测、内容审核、系统提示加固
- **工程化**：限流、健康检查、请求超时、输入校验、优雅停机、结构化日志、监控指标
- **评估体系**：35 条测试用例，覆盖 RAG 召回率、Agent 工具准确率、HITL 安全、安全拦截
- **Docker 部署**：一键部署，含 Redis 会话存储

## 技术栈

| 层级 | 技术 |
|---|---|
| Agent 编排 | LangGraph |
| 大模型 | DeepSeek（ChatOpenAI 兼容） |
| Embedding | 硅基流动 BAAI/bge-large-zh-v1.5 |
| Reranker | 硅基流动 BAAI/bge-reranker-v2-m3 |
| 向量库 | Chroma |
| 关键词检索 | rank_bm25 + jieba |
| Web 框架 | FastAPI + Uvicorn |
| 会话存储 | Redis（生产）/ SQLite（开发） |
| 前端 | 原生 HTML + marked.js |
| 重试/缓存 | tenacity + 本地文件缓存 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
cp .env.example .env
```

```env
# 大模型（DeepSeek）
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# Embedding / Reranker（硅基流动）
EMBED_API_KEY=sk-xxx
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_MODEL=BAAI/bge-large-zh-v1.5
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

### 3. 构建知识库

```bash
python run.py build
```

### 4. 启动服务

```bash
python run.py dev
```

浏览器自动打开 `http://127.0.0.1:8000`

## 命令一览

```bash
python run.py dev           # 启动开发服务器（自动打开浏览器）
python run.py build         # 重建向量库（知识库文档变更后执行）
python run.py test          # 运行评估测试
python run.py docker-up     # Docker 构建并启动
python run.py docker-down   # Docker 停止
python run.py docker-logs   # Docker 实时日志
python run.py clean         # 清理缓存和运行时数据
```

## 项目结构

```
ecom-bot/
├── src/
│   ├── main.py              # 入口层：FastAPI 路由、SSE 流式输出
│   ├── config.py            # 配置管理（多环境 .env 加载）
│   ├── agent/               # Agent 层
│   │   ├── prompt.py        #   System Prompt
│   │   ├── tools.py         #   工具定义（7个工具）
│   │   └── workflow.py      #   LangGraph 工作流 + 流式调用
│   ├── rag/                 # RAG 层
│   │   ├── embeddings.py    #   Embedding 封装（重试+缓存+批量+异步）
│   │   ├── retriever.py     #   混合检索（向量+BM25）+ Reranker 重排序
│   │   └── ingest.py        #   文档入库（解析+分块+向量化）
│   ├── services/            # 服务层
│   │   ├── llm.py           #   大模型封装
│   │   ├── query_rewrite.py #   Query 重写
│   │   ├── feedback.py      #   用户反馈 + 自主学习
│   │   └── security.py      #   安全检查（注入检测+内容审核）
│   ├── data/                # 数据层
│   │   └── repository.py    #   数据访问接口（模拟数据，可替换为真实API）
│   ├── common/              # 公共层
│   │   ├── logger.py        #   结构化日志（控制台+文件，按天轮转）
│   │   └── metrics.py       #   监控指标收集
│   ├── static/              # 前端页面
│   └── tests/
│       └── evaluate.py      # 评估测试（35条用例）
├── docs/                    # 知识库文档（Markdown）
├── .env / .env.example      # 环境变量
├── requirements.txt
├── run.py                   # 命令封装
├── Dockerfile
├── docker-compose.yml       # 含 Redis 的生产部署
└── start.sh                 # 容器启动脚本
```

## API 接口

### 聊天接口（SSE 流式）

```
POST /api/chat
Content-Type: application/json

{
  "message": "我买的手机壳不合适能退吗",
  "session_id": "sess_xxx"
}
```

响应为 SSE 流，事件类型：
- `rewrite`：Query 重写结果
- `tool`：工具调用
- `tool_result`：工具返回结果
- `token`：流式回答内容
- `done`：回答结束

### 其他接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/metrics` | GET | 监控指标（QPS、延迟、错误率等） |
| `/api/feedback` | POST | 用户反馈（👍/👎） |
| `/api/feedback/stats` | GET | 反馈统计 |

## 评估结果

```
RAG 综合召回率:   93.8% (15/16)
Agent 工具准确率:  100% (11/11)
HITL 安全通过率:   100% (2/2)
安全拦截准确率:    100% (6/6)
综合评级:         优秀 (98.4%)
```

运行评估：`python run.py test`

## Docker 部署

```bash
# 构建并启动（含 Redis）
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

生产环境通过 `REDIS_URL` 环境变量自动切换到 Redis 会话存储，未配置时回退 SQLite。

## 配置参考

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `APP_ENV` | dev | 运行环境（dev/test/prod） |
| `LLM_TIMEOUT` | 30 | 大模型请求超时（秒） |
| `RATE_LIMIT_PER_MINUTE` | 30 | 每 IP 每分钟请求上限 |
| `MAX_MESSAGE_LENGTH` | 2000 | 用户消息最大长度 |
| `TOP_K` | 3 | 检索返回条数 |
| `REDIS_URL` | 空 | Redis 连接地址，配置后启用 Redis 会话存储 |

完整配置见 `.env.example`。

## 架构设计

### 分层架构

```
用户请求 → API层（路由/校验/限流/SSE）
         → Agent层（LangGraph工作流/工具/Prompt）
         → 服务层（LLM/重写/反馈/安全）
         → RAG层（检索/向量化/入库）
         → 数据层（订单/商品数据访问）
         → 公共层（日志/指标，被所有层复用）
```

每层单一职责，层间通过接口通信，可独立拆分微服务。

### HITL 两阶段提交

删除/退款等危险操作不直接执行：
1. 用户请求 → 调用 `delete_order`（仅申请，返回确认请求）
2. 前端弹出确认框 → 用户确认
3. 调用 `execute_delete_order`（真正执行）

未确认时绝对不会调用执行工具，由 System Prompt 强制约束。

## License

MIT
