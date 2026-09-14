import os
from pathlib import Path

from contract.review_contract import main
from dotenv import load_dotenv
from llm_client import DeepSeekLLMClient

load_dotenv(Path(__file__).with_name(".env"))

# 逾期付款违约金比例上限
# 逾期付款的违约责任
# 甲方应在货物交付后30日内支付全部价款，逾期支付按日万分之五支付违约金。

api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")


client = DeepSeekLLMClient(
    api_key=api_key,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
)
main("template.txt", client)
