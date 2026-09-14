from typing import Optional

from llm_client import LLMClient, invoke


REVIEW_PROMPT = """你是资深合同律师。请审查下面这条合同条款，判断是否存在法律风险。

【合同条款】
{clause}

【相关法条】
{articles}

请输出：
1. 【风险等级】高 / 中 / 低 / 无
2. 【问题分析】条款中存在的问题（如果有）
3. 【法律依据】引用的具体法条（写明"《XX法》第X条"）
4. 【修改建议】建议改成什么

要求：
- 如果相关法条不能支持某个判断，不要强行引用
- 如果条款合规，明确说明"未发现明显法律风险"
- 引用法条时必须写明条号
"""


def format_articles(articles: list[dict]) -> str:
    lines = []
    for a in articles:
        src = a.get("source", "").replace(".txt", "")
        no = a.get("article_no", "")
        lines.append(f"《{src}》{no}\n{a['text']}")
    return "\n\n".join(lines)


def review_clause(
    clause: str,
    articles: list[dict],
    client: LLMClient,
    model: Optional[str] = None,
) -> str:
    prompt = REVIEW_PROMPT.format(
        clause=clause,
        articles=format_articles(articles),
    )
    return invoke(
        client,
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
    )


CONTRACT_REVIEW_PROMPT = """你是资深合同律师。请依据下列合同条款及每段召回的关联法条，
对整份合同进行一次综合法律评审。没有召回法条的条款已被排除，不要臆测未提供的内容。

【待评审合同材料】
{materials}

请输出：
1. 【整体风险等级】高 / 中 / 低 / 无
2. 【重点问题】逐项说明条款、问题和风险
3. 【法律依据】引用支持判断的具体法条（写成《XX法》第X条）
4. 【整体修改建议】给出可执行的修改方向

要求：
- 只能依据已提供的合同内容和法条进行判断
- 法条不足以支持时不要强行引用或下结论
- 如果整体未发现明显法律风险，请明确说明
"""


def format_review_materials(records: list[dict]) -> str:
    """格式化各段合同及其召回法条。

    :param records: 包含 title、clause 和 articles 的检索记录
    :return: 供整份合同综合评审使用的文本
    """
    sections = []
    for index, record in enumerate(records, 1):
        title = record.get("title") or f"第{index}段"
        clause = record.get("clause") or record.get("text") or ""
        articles = record.get("articles") or []
        sections.append(
            f"【条款{index}：{title}】\n{clause}\n\n"
            f"【该条款召回法条】\n{format_articles(articles)}"
        )
    return "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(sections)


def review_entire_contract(
    records: list[dict],
    client: LLMClient,
    model: Optional[str] = None,
) -> str:
    """对全部有效条款进行一次综合法律评审。

    :param records: 已召回法条的合同条款记录
    :param client: 通用大模型客户端
    :param model: 综合评审模型
    :return: 整份合同的综合评审结果
    """
    if not records:
        return ""

    prompt = CONTRACT_REVIEW_PROMPT.format(
        materials=format_review_materials(records),
    )
    return invoke(
        client,
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
    )
