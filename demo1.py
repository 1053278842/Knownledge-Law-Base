# ---------- 示例 1：直接检索 ----------
print("=" * 60)
print("示例1：直接调用检索工具")
print("=" * 60)
from rag.tool import search_knowledge_base

# 逾期付款违约金比例上限
# 逾期付款的违约责任
# 甲方应在货物交付后30日内支付全部价款，逾期支付按日万分之五支付违约金。

print(search_knowledge_base("逾期付款的违约责任", top_k=5))