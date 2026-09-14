import sys
from openai import OpenAI
from rag.splitter import split_contract
from contract.query_gen import gen_queries
from contract.reviewer import review_clause
from rag.tool import search_knowledge_base

API_KEY = "sk-58d8cc12c864406ba193619651886597"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")


def main(contract_path: str):
    raw = open(contract_path, encoding="utf-8").read()
    clauses = split_contract(raw)
    print(f"合同共 {len(clauses)} 条\n")
    # print(f"合同条款：" + "\n".join([f"  - {c.get('title', '')}" for c in clauses]))

    for i, c in enumerate(clauses, 1):
        title = c.get("title", f"条款{i}")
        print("=" * 70)
        print(f"[{i}/{len(clauses)}] {title}")
        print("=" * 70)

        # 1) 生成查询
        queries = gen_queries(c["text"], client)
        print(f"检索查询：{queries}")

        # 2) 多路检索
        articles = search_knowledge_base(queries,top_k=5)
        # articles = retrieve_for_clause(queries, per_query_k=8, max_articles=15)
        print(f"召回法条：{len(articles)} 条")
        for a in articles[:5]:
            print(f"  - 《{a['source']}》{a.get('article_no','')}")

        # 3) LLM 评审
        review = review_clause(c["text"], articles, client)
        print("\n【评审结果】")
        print(review)
        print("\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "template.txt")