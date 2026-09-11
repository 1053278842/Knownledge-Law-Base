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