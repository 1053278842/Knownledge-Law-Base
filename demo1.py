# ---------- 示例 1：直接检索 ----------
print("=" * 60)
print("示例1：直接调用检索工具")
print("=" * 60)
from rag.tool import search_knowledge_base
print(search_knowledge_base("甲方应在货物交付后30日内支付全部价款，逾期支付按日万分之五支付违约金。", top_k=2))