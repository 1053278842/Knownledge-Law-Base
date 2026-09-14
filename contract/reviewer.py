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


def review_clause(clause: str, articles: list[dict], client, model="deepseek-chat") -> str:
    prompt = REVIEW_PROMPT.format(
        clause=clause,
        articles=format_articles(articles),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content