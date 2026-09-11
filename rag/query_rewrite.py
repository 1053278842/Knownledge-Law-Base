from openai import OpenAI

_client = None


def _get_client(api_key: str, base_url: str) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


REWRITE_PROMPT = """你是一个查询改写助手。根据对话历史，把用户最新的问题改写成一个独立、完整、可脱离上下文理解的检索查询。
只输出改写后的查询，不要解释。

对话历史：
{history}

用户最新问题：{question}

改写后的查询："""


def rewrite_query(
    question: str,
    history: list[dict],
    api_key: str,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-chat",
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

    client = _get_client(api_key, base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": REWRITE_PROMPT.format(history=history_text, question=question)}],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    print(f"[rewrite] '{question}' → '{rewritten}'")
    return rewritten