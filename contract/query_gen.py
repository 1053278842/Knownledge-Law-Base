import json
from openai import OpenAI

PROMPT = """你是合同审查专家。请阅读下面这条合同条款，提取出它涉及的所有法律问题。
对每个法律问题，生成一句用于检索法条的查询。

要求：
- 每个查询要像法律问题，用词要接近法条语言（"当事人"、"合同解除"、"违约金"）
- 覆盖条款涉及的各个法律维度（效力、履行、违约、解除、争议解决等）
- 输出 3-5 个查询
- 只输出 JSON 数组，不要解释

合同条款：
{clause}

输出格式：
["查询1", "查询2", "查询3"]
"""


def gen_queries(clause: str, client: OpenAI, model: str = "deepseek-flash") -> list[str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(clause=clause)}],
        temperature=0.2,
        response_format={"type": "json_object"} if "deepseek" in model else None,
    )
    text = resp.choices[0].message.content.strip()
    # 兼容各种返回格式
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        if isinstance(data, list):
            return data
    except Exception:
        # 兜底：按行拆
        return [l.strip(' -"[]') for l in text.split('\n') if l.strip()][:5]
    return []