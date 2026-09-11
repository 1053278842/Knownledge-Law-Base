import json
from openai import OpenAI
from rag.tool import TOOL_SCHEMA, TOOL_FUNCTIONS
from rag.query_rewrite import rewrite_query

API_KEY = "你的KEY"
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat_once(user_input: str, history: list[dict]) -> str:
    # ① 多轮上下文改写
    search_query = rewrite_query(user_input, history, API_KEY, BASE_URL)

    messages = [
        {"role": "system", "content": "你是公司知识助手，回答前必须先查知识库。"},
        *history,
        {"role": "user", "content": user_input},
    ]

    # ② 第一次调用：LLM 决定调工具
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[TOOL_SCHEMA],
    )
    msg = resp.choices[0].message

    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        # 用改写后的 query 去检索（覆盖 LLM 原始 query）
        args["query"] = search_query
        result = TOOL_FUNCTIONS[call.function.name](**args)
        messages.append({
            "role": "tool", "tool_call_id": call.id, "content": result,
        })

    # ③ 第二次调用：生成答案
    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    history = []
    print("输入 'quit' 退出\n")
    while True:
        q = input("你：").strip()
        if q in ("quit", "exit"):
            break
        answer = chat_once(q, history)
        print(f"助手：{answer}\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})