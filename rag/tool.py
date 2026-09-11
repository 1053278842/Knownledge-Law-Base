import json
from rag.retriever import get_retriever
from rag.query_rewrite import rewrite_query

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "从中文知识库检索相关文档片段。涉及公司制度、法律条文、技术文档时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题，中文"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
}


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    results = get_retriever().retrieve(query, top_k=top_k)
    if not results:
        return "知识库中未找到相关内容。"
    return "\n\n".join(
        f"[片段{i} | 来源:{r['source']} | 相关度:{r.get('rerank_score', r.get('score', 0)):.3f}]\n{r['text']}"
        for i, r in enumerate(results, 1)
    )


TOOL_FUNCTIONS = {"search_knowledge_base": search_knowledge_base}