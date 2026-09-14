"""用户反馈 + 自主学习模块
- 好评(👍)：问答对自动加入知识库（文件 + Chroma），即时生效
- 差评(👎)：保存到待分析文件，识别知识库缺口
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import PROJECT_ROOT, DOCS_DIR, CHROMA_PATH, COLLECTION_NAME
from rag.embeddings import SiliconFlowEmbeddings
from common.logger import get_logger

logger = get_logger("feedback")

# 好评问答文件（持久化，重启后重建索引也能包含）
GOOD_QA_FILE = os.path.join(DOCS_DIR, "用户好评问答.md")
# 差评待分析文件
BAD_QA_FILE = os.path.join(PROJECT_ROOT, "feedback", "bad_questions.jsonl")

# 确保目录存在
os.makedirs(os.path.dirname(BAD_QA_FILE), exist_ok=True)


def _format_good_qa(question: str, answer: str) -> str:
    """把好评问答格式化成知识库条目"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
### Q: {question}

{answer}

> 来源：用户好评收录 | {timestamp}

---
"""


def save_good_feedback(question: str, answer: str):
    """
    好评处理：
    1. 追加到 docs/用户好评问答.md（持久化）
    2. 增量写入 Chroma（即时生效）
    """
    # 1. 追加到文件
    qa_text = _format_good_qa(question, answer)
    try:
        with open(GOOD_QA_FILE, "a", encoding="utf-8") as f:
            # 如果文件不存在，先写标题
            if not os.path.exists(GOOD_QA_FILE) or os.path.getsize(GOOD_QA_FILE) == 0:
                f.write("# 用户好评问答库\n\n> 用户确认有用的问答对，自动收录，持续积累\n\n")
            f.write(qa_text)
        logger.info(f"好评已写入文件 | Q={question[:30]}")
    except Exception as e:
        logger.error(f"写入好评文件失败: {e}")

    # 2. 增量写入 Chroma（即时生效，不用重建整个库）
    try:
        embeddings = SiliconFlowEmbeddings()
        vs = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        # 构造文档：标题路径 + 内容，和 ingest.py 的分块格式一致
        content = f"用户好评问答 > Q: {question}\n### Q: {question}\n\n{answer}"
        doc = Document(
            page_content=content,
            metadata={"source": GOOD_QA_FILE},
        )
        vs.add_documents([doc])
        logger.info(f"好评已写入 Chroma | Q={question[:30]}")
    except Exception as e:
        logger.error(f"写入 Chroma 失败: {e}")


def save_bad_feedback(question: str, answer: str):
    """差评处理：保存到 jsonl 文件，供后续分析知识库缺口"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
    }
    try:
        with open(BAD_QA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"差评已记录 | Q={question[:30]}")
    except Exception as e:
        logger.error(f"写入差评文件失败: {e}")


def get_bad_feedback_count() -> int:
    """获取待分析差评数量"""
    if not os.path.exists(BAD_QA_FILE):
        return 0
    with open(BAD_QA_FILE, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_good_feedback_count() -> int:
    """获取已收录好评数量"""
    if not os.path.exists(GOOD_QA_FILE):
        return 0
    with open(GOOD_QA_FILE, "r", encoding="utf-8") as f:
        return f.read().count("### Q:")
