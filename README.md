# 电商客服机器人

基于 LangChain + LangGraph 的企业级电商客服 AI 机器人。支持 RAG 知识库检索、Agent 工具调用、三级意图路由、流式输出、多轮对话记忆压缩、HITL 人工确认、端到端评估等完整能力。

## 功能特性

### 核心能力
- **RAG 知识库**：向量检索（Chroma）+ BM25 关键词检索 + Reranker 重排序，混合检索提升准确率
- **Agent 工具调用**：5 个工具（知识库查询、订单查询、商品搜索、删除订单、退款申请），LangGraph ReAct 编排
- **HITL 人工确认**：删除/退款危险操作两阶段执行，硬门禁隔离执行工具
- **流式输出**：SSE 逐字返回，前端实时渲染
- **多轮对话**：Redis（生产）/ SQLite（开发）会话存储 + 滑动窗口摘要压缩

### 智能路由
- **三级意图分类**：规则前置过滤（零成本）→ 小模型 Qwen2.5-7B（便宜）→ 主模型 DeepSeek（兜底）
- **意图上下文注入**：分类结果注入 Agent，减少 LLM 判断负担，工具选择更准
- **直接回复路由**：问候/投诉直接返回固定回复，跳过 Agent 省 LLM 调用

### RAG 增强
- **增量更新**：基于 MD5 哈希的 manifest 机制，只处理变更文件，支持 `--force` 全量重建
- **文本噪音清理**：零宽字符、控制字符、多余空白、重复标点自动清洗
- **metadata 完整性**：分块时继承 source + file_type，检索结果可追溯来源
- **结构化分块**：Markdown 标题层级分块，chunk_size=400，overlap=40

### 工程化
- **生产级健康检查**：`/health/live`（存活）+ `/health/ready`（就绪，检测 LLM/Embedding/向量库/会话存储）
- **对话记忆压缩**：超过 10 轮自动用 LLM 摘要旧对话，防止上下文溢出和 token 爆炸
- **Query Rewrite**：多轮对话自动改写省略/代词问题
- **用户反馈 + 自主学习**：好评问答自动加入知识库，系统越用越准
- **安全防护**：Prompt 注入检测、内容审核、输出泄露过滤、系统提示加固
- **错误分类重试**：`[错误:超时]` 可重试、`[错误:不存在]` 不重试、`[错误:系统异常]` 转人工
- **限流**：按 IP 限制请求频率，防止滥用和 API 费用爆炸
- **监控指标**：QPS、延迟（P50/P99）、错误率、工具调用分布、Token 消耗
- **结构化日志**：控制台 + 文件，按天轮转，请求 ID 全链路追踪

### 评估体系
- **RAG 召回率**：16 条用例，验证混合检索 + 重排序效果
- **Agent 工具准确率**：11 条用例，验证工具选择正确性
- **HITL 安全流程**：2 条用例，验证危险操作不直接执行
- **安全拦截**：6 条用例，验证注入攻击和违规内容拦截
- **端到端回答质量（LLM 裁判）**：8 条用例，5 维度打分（准确性/相关性/完整性/语气/格式）

## 技术栈

| 层级 | 技术 |
|---|---|
| Agent 编排 | LangGraph（ReAct 循环 + 条件路由） |
| 主模型 | DeepSeek（deepseek-chat，ChatOpenAI 兼容） |
| 小模型（意图分类） | 硅基流动 Qwen/Qwen2.5-7B-Instruct |
| Embedding | 硅基流动 BAAI/bge-large-zh-v1.5（1024维） |
| Reranker | 硅基流动 BAAI/bge-reranker-v2-m3 |
| 向量库 | Chroma（持久化） |
| 关键词检索 | rank_bm25 + jieba 分词 |
| Web 框架 | FastAPI + Uvicorn（全异步） |
| 会话存储 | Redis（生产）/ SQLite（开发），LangGraph checkpointer |
| 前端 | 原生 HTML + marked.js + SSE 流式渲染 |
| 重试/缓存 | tenacity + 本地文件缓存（Embedding 结果） |
| 容器化 | Docker + docker-compose（含 Redis） |

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

# Embedding / Reranker / 小模型（硅基流动，同一个 Key）
EMBED_API_KEY=sk-xxx
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_MODEL=BAAI/bge-large-zh-v1.5
RERANK_MODEL=BAAI/bge-reranker-v2-m3
SMALL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### 3. 构建知识库

```bash
python run.py build          # 增量更新（默认）
python run.py build --force  # 全量重建
```

### 4. 启动服务

```bash
python run.py dev
```

浏览器自动打开 `http://127.0.0.1:8000`

## 命令一览

```bash
python run.py dev           # 启动开发服务器（自动打开浏览器）
python run.py build         # 增量构建向量库（知识库文档变更后执行）
python run.py build --force # 全量重建向量库
python run.py test          # 运行评估测试（RAG + Agent + HITL + 安全 + E2E）
python run.py docker-up     # Docker 构建并启动（含 Redis）
python run.py docker-down   # Docker 停止
python run.py docker-logs   # Docker 实时日志
python run.py clean         # 清理缓存和运行时数据
```

## 项目结构

```
ecom-bot/
├── src/
│   ├── main.py              # 入口层：FastAPI 路由、SSE 流式输出、安全检查、限流
│   ├── config.py            # 配置管理（多环境 .env 加载）
│   ├── agent/               # Agent 层
│   │   ├── prompt.py        #   System Prompt（安全规则 + 工具调用规则 + HITL约束）
│   │   ├── tools.py         #   工具定义（5个暴露工具 + 2个执行工具硬门禁）
│   │   └── workflow.py      #   LangGraph 工作流 + 流式调用 + 记忆压缩
│   ├── rag/                 # RAG 层
│   │   ├── embeddings.py    #   Embedding 封装（重试+缓存+批量+异步）
│   │   ├── retriever.py     #   混合检索（向量+BM25）+ Reranker 重排序
│   │   └── ingest.py        #   文档入库（解析+清洗+分块+向量化+增量更新）
│   ├── services/            # 服务层
│   │   ├── llm.py           #   大模型封装（DeepSeek）
│   │   ├── intent.py        #   三级意图路由（规则→小模型→主LLM）
│   │   ├── memory.py        #   对话记忆压缩（滑动窗口+LLM摘要）
│   │   ├── query_rewrite.py #   Query 重写（多轮对话上下文补全）
│   │   ├── feedback.py      #   用户反馈 + 自主学习（好评入库）
│   │   ├── security.py      #   安全检查（注入检测+内容审核+输出过滤）
│   │   └── health.py        #   生产级健康检查（LLM/Embedding/向量库/会话存储）
│   ├── data/                # 数据层
│   │   └── repository.py    #   数据访问接口（模拟数据，可替换为真实API）
│   ├── common/              # 公共层
│   │   ├── logger.py        #   结构化日志（控制台+文件，按天轮转，请求ID追踪）
│   │   └── metrics.py       #   监控指标收集（QPS/延迟/错误率/工具分布）
│   ├── static/              # 前端页面
│   └── tests/
│       └── evaluate.py      #   评估测试（RAG+Agent+HITL+安全+E2E，共43条用例）
├── docs/                    # 知识库文档（Markdown/PDF/DOCX）
├── .env / .env.example      # 环境变量
├── requirements.txt
├── run.py                   # 命令封装
├── Dockerfile
├── docker-compose.yml       # 含 Redis 的生产部署
└── start.sh                 # 容器启动脚本
```

## 请求处理流程

```
用户消息
  ↓
① 安全检查（Prompt注入 + 内容审核）→ 不通过则 403 拦截
  ↓
② 三级意图分类
   ├─ 规则匹配（问候/投诉）→ 直接回复，跳过 Agent
   ├─ 小模型 Qwen2.5-7B（置信度≥0.7）→ 使用分类结果
   └─ 主模型 DeepSeek（兜底）
  ↓
③ 读取会话历史（SQLite/Redis checkpointer）
  ↓
④ Query Rewrite（多轮对话补全上下文）
  ↓
⑤ 注入意图上下文（帮助 Agent 选对工具）
  ↓
⑥ Agent 执行（LangGraph ReAct 循环）
   ├─ LLM 决定调工具还是回答
   ├─ 调工具 → 执行 → 结果回传 LLM → 循环
   └─ 回答 → 输出过滤（检测系统提示泄露）
  ↓
⑦ SSE 流式返回前端
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
- `intent`：意图分类结果（意图 + 置信度）
- `rewrite`：Query 重写结果
- `tool`：工具调用
- `tool_result`：工具返回结果
- `token`：流式回答内容
- `done`：回答结束

### HITL 确认接口

```
POST /api/hitl/confirm
Content-Type: application/json

{
  "action": "refund",        // refund 或 delete
  "order_id": "ORD001",
  "session_id": "sess_xxx"
}
```

危险操作的硬门禁：`execute_delete_order` / `execute_refund_order` 不暴露给 LLM，仅由此接口直接调用。

### 其他接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health/live` | GET | 存活探针（进程是否运行） |
| `/health/ready` | GET | 就绪探针（所有依赖是否可用，失败返回 503） |
| `/health` | GET | 兼容旧接口，返回完整检查结果 |
| `/metrics` | GET | 监控指标（QPS、延迟、错误率等） |
| `/api/feedback` | POST | 用户反馈（👍/👎） |
| `/api/feedback/stats` | GET | 反馈统计 |

## 评估结果

```
RAG 综合召回率:     100% (16/16)
Agent 工具准确率:    100% (11/11)
HITL 安全通过率:     100% (2/2)
安全拦截准确率:      100% (6/6)
端到端回答质量:      95.6% (4.78/5, LLM裁判)
综合评级:           优秀
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
| `SMALL_LLM_MODEL` | Qwen/Qwen2.5-7B-Instruct | 意图分类小模型 |
| `SMALL_LLM_CONFIDENCE_THRESHOLD` | 0.7 | 小模型置信度阈值，低于则升级主 LLM |
| `MAX_RECENT_TURNS` | 10 | 对话记忆压缩阈值，超过则摘要旧对话 |

完整配置见 `.env.example`。

## 架构设计

### 分层架构

```
用户请求 → API层（路由/校验/限流/SSE/安全）
         → 意图路由层（规则→小模型→主LLM）
         → Agent层（LangGraph工作流/工具/Prompt/记忆压缩）
         → 服务层（LLM/重写/反馈/安全/健康检查）
         → RAG层（混合检索/向量化/增量入库）
         → 数据层（订单/商品数据访问）
         → 公共层（日志/指标，被所有层复用）
```

每层单一职责，层间通过接口通信，可独立拆分微服务。

### HITL 两阶段提交（硬门禁）

删除/退款等危险操作不直接执行：
1. 用户请求 → LLM 调用 `delete_order` / `refund_order`（仅返回确认请求，不执行）
2. 前端弹出确认框 → 用户明确确认
3. 前端调用 `/api/hitl/confirm` → 后端直接执行 `execute_delete_order` / `execute_refund_order`

执行工具**不在 LLM 可用工具列表中**，由代码强制隔离，不依赖模型自觉。

### 三级意图路由

```
规则匹配（零成本）→ 命中直接返回
  ↓ 未命中
小模型 Qwen2.5-7B（便宜，约主模型1/10成本）
  ├─ 置信度 ≥ 0.7 → 使用结果
  └─ 置信度 < 0.7 或 API 失败 → 升级
  ↓
主模型 DeepSeek（兜底，最准但最贵）
```

90%+ 的请求止步于前两级，主模型只处理复杂/模糊意图。

## License

MIT
