from typing import Optional

from llm_client import LLMClient, invoke


REWRITE_PROMPT = """你是一个查询改写助手。根据对话历史，把用户最新的问题改写成一个独立、完整、可脱离上下文理解的检索查询。
只输出改写后的查询，不要解释。

对话历史：
{history}

用户最新问题：{question}

改写后的查询："""


def rewrite_query(
    question: str,
    history: list[dict],
    client: LLMClient,
    model: Optional[str] = None,
    max_turns: int = 4,
) -> str:
    """
    没有历史 → 原样返回（省一次 LLM 调用）
    有历史 → 结合上文改写
    例如：
      历史: 用户问"年假多少天"，回答"5天"
      新问: "怎么申请？"
      改写: "公司年假的申请流程是什么？"
    """
    if not history:
        return question

    recent = history[-max_turns * 2:]
    history_text = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}：{m['content']}"
        for m in recent
    )

    rewritten = invoke(
        client,
        messages=[{
            "role": "user",
            "content": REWRITE_PROMPT.format(
                history=history_text,
                question=question,
            ),
        }],
        model=model,
        temperature=0,
    )
    rewritten = rewritten.strip()
    print(f"[rewrite] '{question}' → '{rewritten}'")
    return rewritten
