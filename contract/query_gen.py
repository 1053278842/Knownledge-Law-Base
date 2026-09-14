import json
from typing import Optional

from llm_client import LLMClient, invoke

PROMPT = """你是合同审查专家。请阅读下面这条合同条款，提取出它涉及的所有法律问题。
对每个法律问题，生成一句用于检索法条的查询。

要求：
- 每个查询要像法律问题，用词要接近法条语言（"当事人"、"合同解除"、"违约金"）
- 覆盖条款涉及的各个法律维度（效力、履行、违约、解除、争议解决等）
- 如果条款没有涉及法律问题、或无实际意义，则输出空数组
- 输出 0-5 个查询
- 只输出 JSON 数组，不要解释

合同条款：
{clause}

输出格式：
["查询1", "查询2", "查询3"]
"""


def gen_queries(
    clause: str,
    client: LLMClient,
    model: Optional[str] = None,
) -> list[str]:
    """为单个合同条款生成法条检索问题。

    :param clause: 合同条款正文
    :param client: 通用大模型客户端
    :param model: 可选的模型名称
    :return: 0-5 个检索问题
    """
    text = invoke(
        client,
        messages=[{"role": "user", "content": PROMPT.format(clause=clause)}],
        model=model,
        temperature=0.2,
    )
    text = text.strip()
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
