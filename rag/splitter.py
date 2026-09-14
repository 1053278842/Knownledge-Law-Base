import re

# 匹配 "第X条"，X 是中文数字
ARTICLE_PAT = re.compile(r'(?:^|\n)[　\s]*(第[一二三四五六七八九十百千零]+条)')
# 匹配 "第X章"
CHAPTER_PAT = re.compile(r'第[一二三四五六七八九十百千零]+章\s*[^\n]*')


def _clean(text: str) -> str:
    """去掉修订历史、目录，从'第一条'开始才是正文"""
    # 找第一个"第X条"的位置（目录里没有"第X条"，只有"第X章"）
    m = ARTICLE_PAT.search(text)
    if m:
        text = text[m.start():]
    # 统一空白
    text = text.replace("\u3000", " ").strip()
    return text


def _find_chapter(text_before: str) -> str:
    """找当前条文所属的章标题"""
    matches = CHAPTER_PAT.findall(text_before)
    return matches[-1].strip() if matches else ""


def split_law_text(text: str, max_len: int = 800) -> list[dict]:
    """
    法律文档专用切分：
    - 去掉目录/修订历史
    - 以"第X条"为最小单元
    - 单条超过 max_len 时按句号二次切分
    返回：[{text, article_no, chapter, sub_id}, ...]
    """
    text = _clean(text)

    # 按"第X条"切：切点保留在结果里
    parts = ARTICLE_PAT.split(text)
    # parts = ['', '第一条', '内容...', '第二条', '内容...', ...]
    #          ↑parts[0]是"第一条"之前的残留，一般为空

    chunks = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        article_no = parts[i]           # "第一条"
        body = parts[i + 1].strip()
        full = f"{article_no}　{body}"

        # 记录这条属于哪一章（用"第X条"之前的文本找最近的"第X章"）
        pos = text.find(article_no)
        chapter = _find_chapter(text[:pos])

        if len(full) <= max_len:
            chunks.append({
                "text": full,
                "article_no": article_no,
                "chapter": chapter,
                "sub_id": 0,
            })
        else:
            # 太长：按句号二次切
            sentences = re.split(r'(?<=[。；])', body)
            buf = article_no + "　"
            sub = 0
            for s in sentences:
                if len(buf) + len(s) > max_len and buf.strip() != article_no:
                    chunks.append({
                        "text": buf.strip(),
                        "article_no": article_no,
                        "chapter": chapter,
                        "sub_id": sub,
                    })
                    sub += 1
                    buf = f"{article_no}（续）　"
                buf += s
            if buf.strip():
                chunks.append({
                    "text": buf.strip(),
                    "article_no": article_no,
                    "chapter": chapter,
                    "sub_id": sub,
                })

    return chunks


# 匹配合同条款开头：第X条、一、1.、(1) 等多种形式
CLAUSE_PATTERNS = [
    r'^第[一二三四五六七八九十百]+条\s*[^\n]*',       # 第X条 标题
    r'^[一二三四五六七八九十]+[、．.]\s*[^\n]*',      # 一、二、
    r'^\d+[、．.]\d*\s*[^\n]*',                     # 1. 1.1
    r'^（[一二三四五六七八九十\d]+）[^\n]*',          # （一）
]


def split_contract(text: str, max_len: int = 600) -> list[dict]:
    """把合同切成条款单元"""
    lines = text.split('\n')
    clauses = []
    buf = []
    cur_title = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_new = any(re.match(p, line) for p in CLAUSE_PATTERNS)

        if is_new and buf:
            body = '\n'.join(buf).strip()
            if body:
                # 太长再按句号切
                if len(body) <= max_len:
                    clauses.append({"title": cur_title, "text": body})
                else:
                    for sent_group in _split_long(body, max_len):
                        clauses.append({"title": cur_title, "text": sent_group})
            buf = []

        if is_new:
            cur_title = line
        buf.append(line)

    if buf:
        clauses.append({"title": cur_title, "text": '\n'.join(buf).strip()})

    return clauses


def _split_long(text: str, max_len: int) -> list[str]:
    """长条款按句号二次切分"""
    sents = re.split(r'(?<=[。；])', text)
    result, buf = [], ""
    for s in sents:
        if len(buf) + len(s) > max_len and buf:
            result.append(buf.strip())
            buf = s
        else:
            buf += s
    if buf.strip():
        result.append(buf.strip())
    return result