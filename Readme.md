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
实现统一的 `LLMClient.chat` 协议，然后注入 `main`：

```python
from contract.review_contract import main
from llm_client import LLMClient


class InternalLLMClient:
    def chat(self, messages: list[dict], *, model=None, temperature=0.0) -> str:
        # 调用内网 AI SDK 或 HTTP 接口，返回生成的文本
        ...


main("template.txt", InternalLLMClient())
```

`messages` 使用通用的 `{"role": ..., "content": ...}` 格式，业务层不依赖任何厂商 SDK。
