## 准备工作
1. 安装依赖
    ```bash
    pip3 install -r .\requirements.txt     
    ```

2. 模型下载
    ```bash
    # 设置镜像（一次即可，当前窗口有效）
    $env:HF_ENDPOINT = "https://hf-mirror.com"

    # 下载向量模型（~95MB）
    hf download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5

    # 下载精排模型（~1.1GB）
    hf download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
    ```

3. 文档向量化
    ```bash
    python index_docs.py 
    ```

## 运行
1. 分词demo
    ```bash
    python demo1.py 
    ```

## 接入 DeepSeek
项目根目录的 `.env` 用于保存本地敏感配置，`.env` 已加入 `.gitignore`。
将密钥填写到 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的 API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

然后运行：

```powershell
python demo.py
```

环境变量优先级高于 `.env`，也可以手动创建客户端：

```python
from llm_client import DeepSeekLLMClient

client = DeepSeekLLMClient(api_key="你的 API Key")
```

## 接入其他内网 AI
如果内网 AI 使用自定义请求头、请求体或响应结构，可以使用 `HttpLLMClient`：

```python
import json
import os

from contract.review_contract import main
from llm_client import HttpLLMClient


def build_headers(messages, model, temperature):
    return {
        "X-API-Key": os.environ["INTERNAL_LLM_API_KEY"],
        "X-Tenant-Id": os.environ["INTERNAL_LLM_TENANT_ID"],
    }


def build_body(messages, model, temperature):
    return {
        "model": model or "internal-model",
        "input": messages,
        "temperature": temperature,
    }


def parse_response(raw: str) -> str:
    return json.loads(raw)["data"]["answer"]


client = HttpLLMClient(
    endpoint=os.environ["INTERNAL_LLM_ENDPOINT"],
    header_builder=build_headers,
    body_builder=build_body,
    response_parser=parse_response,
)
main("template.txt", client)
```

`endpoint` 必须是完整请求地址，不会自动追加 `/chat/completions`。
三个构造器参数按顺序接收 `messages`、`model`、`temperature`。
业务层仍只依赖统一的 `LLMClient.chat` 协议。
