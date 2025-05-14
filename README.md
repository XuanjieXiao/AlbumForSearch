# 智能相册项目 (基于 Chinese-CLIP 与 Qwen-VL)

本项目是一个基于 Flask 的智能相册应用，允许用户上传图片，并通过中文文本描述进行搜索。它利用 Chinese-CLIP 进行图文特征提取，FAISS进行向量检索，并可选地使用 Qwen-VL (通过云API) 对图片进行详细描述和关键词提取，以增强搜索效果。元数据存储在 SQLite3 数据库中。

## 项目结构


smart_album_project/
├── app.py                     # 主 Flask 应用
├── bce_service.py             # BCE 文本向量化服务
├── qwen_service.py            # Qwen-VL 图片分析服务
├── database_utils.py          # SQLite3 数据库辅助函数
├── faiss_utils.py             # FAISS 向量索引管理
├── Chinese_CLIP/              # Chinese-CLIP 代码库
├── static/                    # 前端静态文件 (style.css, script.js)
├── templates/                 # HTML 模板 (index.html)
├── uploads/                   # 用户上传的原始图片 (运行时创建)
├── thumbnails/                # 生成的缩略图 (运行时创建)
├── data/                      # 数据文件 (smart_album.db, album_faiss.index)
├── models/                    # CLIP模型和未来BCE本地模型
└── README.md                  # 本文件


## 技术栈

* **后端**: Python, Flask
* **模型**:
    * Chinese-CLIP (您提供的本地版本, 例如 ViT-H-14)
    * Qwen-VL (通过 `qwen_service.py` 调用云API)
    * BCE 文本向量化 (当前使用 `sentence-transformers` 作为替代，输出512维)
* **向量数据库**: FAISS
* **元数据数据库**: SQLite3
* **前端**: HTML, CSS, JavaScript (原生)

## 环境准备与依赖安装

1.  **Python 环境**: 建议使用 Python 3.9+。创建一个虚拟环境是个好习惯：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

2.  **安装依赖**:
    项目根目录下创建一个 `requirements.txt` 文件，内容如下：
    ```txt
    flask
    flask-cors
    numpy
    Pillow
    faiss-cpu  # 或者 faiss-gpu 如果您有NVIDIA GPU并配置了CUDA
    torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) # 根据您的CUDA版本调整，或使用CPU版
    # sentence-transformers # 用于BCE替代模型
    openai # 用于qwen_service.py
    # 以下是Chinese_CLIP可能需要的依赖，请根据其原始requirements添加
    # ftfy
    # regex
    # tqdm
    ```
    然后运行:
    ```bash
    pip install -r requirements.txt
    # 如果 sentence-transformers 未自动安装，单独安装:
    pip install sentence-transformers
    ```
    **注意**: `Chinese_CLIP` 库本身可能有其特定的依赖，请参考其文档或 `requirements.txt`。您提供的代码中已包含 `cn_clip`，确保其能被正确导入。`torch` 的安装请根据您的环境（CPU或特定CUDA版本）选择合适的命令。

3.  **Chinese-CLIP 模型**:
    * 确保 `Chinese_CLIP` 目录位于项目根目录下。
    * `app.py` 中的 `CLIP_MODEL_DOWNLOAD_ROOT` 设置为 `./models/`。当应用首次启动并加载CLIP模型时，如果模型文件不存在，`cn_clip` 库会尝试从默认源下载到此路径下（例如 `models/ViT-H-14/`）。

4.  **Qwen-VL API 配置**:
    * 打开 `qwen_service.py` 文件。
    * 修改 `QWEN_API_KEY` 和 `QWEN_BASE_URL` 为您的实际 Qwen-VL 服务API密钥和地址。**强烈建议**通过环境变量设置API密钥，而不是硬编码。
        ```python
        # 示例:
        # QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "YOUR_ACTUAL_API_KEY")
        ```
        运行时设置环境变量：`export QWEN_API_KEY="your_key_here"` (Linux/macOS) 或在 `.env` 文件中管理。

5.  **BCE 模型 (替代)**:
    * `bce_service.py` 使用 `sentence-transformers` 库并尝试下载 `shibing624/text2vec-base-chinese` 模型。首次运行时会自动下载。如果您希望使用其他模型或本地模型，请修改 `bce_service.py` 中的 `MODEL_NAME` 和加载逻辑。

## 运行项目

1.  **启动后端服务**:
    在项目根目录 (`smart_album_project/`) 下运行:
    ```bash
    python app.py
    ```
    服务默认会在 `http://0.0.0.0:5000` 启动。

2.  **访问前端**:
    打开浏览器，访问 `http://localhost:5000`。

## 功能说明

* **图片上传**:
    * 点击 "选择文件夹" 按钮（实际是文件选择框，但设置了 `webkitdirectory` 属性，在支持的浏览器中可以选择文件夹）。
    * 选择包含图片的文件夹后，点击 "上传选中文件夹内的图片" 按钮。
    * 后端会处理上传的图片：生成缩略图、提取CLIP特征、存入数据库和FAISS索引。
    * 如果 "Qwen-VL图片分析" 开关（在控制面板）是开启状态（默认），上传后会自动对图片进行Qwen-VL分析，并将描述和关键词存入数据库，同时更新FAISS中的向量（拼接了BCE描述向量）。
* **图片搜索**:
    * 在搜索框中输入中文描述。
    * 点击 "搜索" 按钮。
    * 后端会计算查询文本的CLIP特征和BCE特征，拼接后在FAISS中进行相似度检索。
    * 结果会以缩略图形式展示，并显示相似度（如果是搜索结果）。
* **图片浏览与增强**:
    * 图片库会分页展示所有已上传的图片。
    * 点击图片缩略图可以查看大图、文件名、相似度（如果是搜索结果）、Qwen描述和关键词（如果已增强）。
    * 如果图片尚未进行Qwen-VL增强分析，详情弹窗中会显示 "对此图片进行增强分析" 按钮，点击可手动触发分析。
* **控制面板**:
    * **Qwen-VL图片分析 (上传时自动)**: 控制新图片上传时是否自动进行Qwen-VL分析。此设置会实时更新到后端。
    * **使用增强搜索**: 此开关目前主要为前端UI占位，您可以扩展其功能，例如在选中时，前端更侧重展示Qwen描述，或者未来可以传递给后端影响排序策略。

## 注意事项与未来优化

* **SOC设备优化**:
    * 当前代码主要为CPU环境设计（FAISS和PyTorch）。要在SOC设备上高效运行，您需要：
        * 将PyTorch模型 (CLIP, BCE替代模型) 转换为适用于您SOC上TPU的格式，如 TensorFlow Lite (`.tflite`)。
        * 修改模型加载和推理部分，使用相应的TPU运行时API (如 TFLite Interpreter for Python/C++)。
        * FAISS在端侧通常使用CPU版本。
* **错误处理**: 当前错误处理比较基础，生产环境需要更完善的错误捕获和用户反馈。
* **性能**:
    * 批量上传大量图片时，Qwen-VL分析（尤其是云API调用）和特征提取可能会比较耗时。同步处理可能导致请求超时或用户体验不佳。可以考虑引入后台任务队列 (如 Celery, RQ, 或 Python 的 `concurrent.futures`) 进行异步处理。
    * 数据库和FAISS的查询优化。
* **BCE模型**: 当前使用的是 `sentence-transformers` 作为BCE的替代。请替换为真实的、性能更优的BCE模型，并确保其输出维度与配置一致。
* **FAISS ID管理**: 当前 `faiss_id` 的分配策略较为简单（基于数据库最大ID递增）。在并发高或分布式场景下需要更健壮的ID生成和同步机制。
* **安全性**: API密钥管理、输入验证等。
* **前端体验**: 当前前端比较基础，可以进一步美化和增强交互。

## 目录与文件说明

* `app.py`: Flask应用主文件，包含API路由和核心业务逻辑。
* `bce_service.py`: 提供文本BCE向量化功能。
* `qwen_service.py`: 调用Qwen-VL云API进行图片分析。
* `database_utils.py`: SQLite数据库的初始化、增删改查操作。
* `faiss_utils.py`: FAISS索引的初始化、向量添加、搜索、保存等操作。
* `Chinese_CLIP/`: 存放您提供的Chinese-CLIP模型代码。
* `static/`: 存放CSS和JavaScript文件。
* `templates/`: 存放HTML模板。
* `uploads/`: 上传
* `thumbnails/`: 图片缩略图存储目录。
* `data/`: 数据库文件 (`smart_album.db`) 和FAISS索引文件 (`album_faiss.index`) 存储目录。
* `models/`: 下载的预训练模型 (如CLIP的ViT-H-14) 和未来可能存放的本地BCE模型。

