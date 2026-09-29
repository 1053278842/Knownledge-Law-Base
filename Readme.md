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

## 模型配置

项目根目录的 `.env` 用于保存本地敏感配置，`.env` 已加入 `.gitignore`。
对话、向量化和精排分别使用独立配置。

### Demo 模式

- 对话模型使用 DeepSeek
- 向量化默认使用本地 `bge-small-zh-v1.5`
- 精排默认使用本地 `bge-reranker-base`

对话模型必须配置完整的 `CHAT_URL`：

```dotenv
CHAT_URL=https://api.deepseek.com/chat/completions
CHAT_API_KEY=你的对话模型 API Key
CHAT_CHANNEL=deepseek-chat
```

然后运行：

```powershell
python demo.py
```

### BOCAI 模式

三个模型必须分别填写 URL、API Key 和通道名：

```dotenv
CHAT_URL=https://your-host/bocai-chat/mtp/v1/chat/completions
CHAT_API_KEY=对话模型 ChannelAPIkey
CHAT_CHANNEL=llm-chat-core

EMBEDDING_PROVIDER=bocai
EMBEDDING_URL=https://your-host/bocai-chat/agent/v1/embeddings
EMBEDDING_API_KEY=向量模型 ChannelAPIkey
EMBEDDING_CHANNEL=embedding
EMBEDDING_DIM=1024

RERANK_PROVIDER=bocai
RERANK_URL=https://your-host/bocai-chat/agent/v1/rerank
RERANK_API_KEY=精排模型 ChannelAPIkey
RERANK_CHANNEL=rerank
```

三个 `*_URL` 都必须填写完整请求地址，程序不会自动追加接口路径。
如果示例中的 `mtp(agent)` 表示二选一，请按实际环境填写 `/mtp/v1/...`
或 `/agent/v1/...`。

也可以只切换其中一个能力。例如向量化使用 BOCAI，精排暂时保留本地：

```dotenv
EMBEDDING_PROVIDER=bocai
EMBEDDING_URL=https://your-host/bocai-chat/agent/v1/embeddings
EMBEDDING_API_KEY=向量模型 ChannelAPIkey
EMBEDDING_CHANNEL=embedding
```

### 重新建立向量索引

不同向量模型的向量空间不能混用。切换到 BOCAI 向量模型后，需要重新执行：

```powershell
python index_docs.py
```

程序会按 embedding provider 和通道配置生成独立的索引目录，并校验向量维度。
如果 `EMBEDDING_DIM` 留空，首次调用时会自动识别远程向量维度；
如果填写了维度但与接口返回值不一致，程序会直接报错。

对话、向量化和精排客户端统一由 `model_clients.py` 创建，业务层仍只依赖
`chat`、`embed_texts` 和 `rerank` 接口。自定义 HTTP 协议仍可通过
`llm_client.HttpLLMClient` 手工注入。
