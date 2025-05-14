# bce_service.py
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os

# --- 配置 ---
# 尝试加载一个预训练的中文模型。您可以替换为更适合的BCE模型。
# 'shibing624/text2vec-base-chinese' 输出 768 维
# 'uer/sbert-base-chinese-nli' 输出 768 维
# 'nghuyong/ernie-3.0-base-zh' (需要 PaddlePaddle)
# 为了演示，我们使用一个常见的 sentence-transformers 模型
# maidalun1020/bce-embedding-base_v1 这个模型是输出512维度，算能适配了
# 如果您有特定的BCE模型（例如 .tflite 格式），加载方式会有所不同
MODEL_NAME = 'shibing624/text2vec-base-chinese' 
# 目标输出维度，根据您的方案是512。如果模型原生维度不同，需要处理。
TARGET_DIM = 768 

bce_model = None
model_actual_dim = 0

def load_bce_model():
    global bce_model, model_actual_dim
    try:
        # 检查模型是否已下载或指定本地路径
        # model_path = os.path.join("models", "bce_model_placeholder", MODEL_NAME) # 示例本地路径
        # if os.path.exists(model_path):
        #     bce_model = SentenceTransformer(model_path)
        # else:
        bce_model = SentenceTransformer(MODEL_NAME) # 从Hugging Face Hub下载
        
        # 获取模型原生维度
        dummy_emb = bce_model.encode(["测试文本"])
        model_actual_dim = dummy_emb.shape[1]
        logging.info(f"BCE (替代) 模型 '{MODEL_NAME}' 加载成功。原生维度: {model_actual_dim}, 目标维度: {TARGET_DIM}")
        
        if model_actual_dim < TARGET_DIM:
            logging.warning(f"模型原生维度 {model_actual_dim} 小于目标维度 {TARGET_DIM}。将进行零填充。")
        elif model_actual_dim > TARGET_DIM:
            logging.warning(f"模型原生维度 {model_actual_dim} 大于目标维度 {TARGET_DIM}。将进行截断。")

    except Exception as e:
        logging.error(f"加载 BCE (替代) 模型 '{MODEL_NAME}' 失败: {e}")
        bce_model = None

def get_bce_embedding(text: str) -> np.ndarray:
    """
    获取文本的BCE特征向量 (numpy array, float32, 维度 TARGET_DIM)。
    """
    if bce_model is None:
        logging.warning("BCE (替代) 模型未加载。返回零向量。")
        return np.zeros(TARGET_DIM, dtype=np.float32)

    try:
        # (1, model_actual_dim)
        embedding = bce_model.encode([text], convert_to_numpy=True, normalize_embeddings=True) 
        
        # 调整维度到 TARGET_DIM
        if model_actual_dim == TARGET_DIM:
            final_embedding = embedding[0]
        elif model_actual_dim < TARGET_DIM:
            # 零填充
            padding = np.zeros(TARGET_DIM - model_actual_dim, dtype=np.float32)
            final_embedding = np.concatenate((embedding[0], padding))
        else: # model_actual_dim > TARGET_DIM
            # 截断
            final_embedding = embedding[0][:TARGET_DIM]
            # 截断后最好重新归一化
            norm = np.linalg.norm(final_embedding)
            if norm > 1e-6 : # 避免除以零
                 final_embedding = final_embedding / norm

        return final_embedding.astype(np.float32)
    except Exception as e:
        logging.error(f"计算文本 '{text[:20]}...' 的 BCE (替代) embedding 失败: {e}")
        return np.zeros(TARGET_DIM, dtype=np.float32)

# 初始化时加载模型
load_bce_model()

if __name__ == "__main__":
    if bce_model:
        sample_text = "这是一段用于测试BCE向量化服务的中文文本。"
        embedding = get_bce_embedding(sample_text)
        print(f"示例文本: '{sample_text}'")
        print(f"Embedding 维度: {embedding.shape}")
        print(f"Embedding (前10): {embedding[:10]}")
        print(f"Embedding L2 范数: {np.linalg.norm(embedding)}")

        sample_text_2 = "另一段不同的文本。"
        embedding_2 = get_bce_embedding(sample_text_2)
        print(f"示例文本2: '{sample_text_2}'")
        print(f"Embedding 2 维度: {embedding_2.shape}")
        
        # 简单相似度计算示例
        similarity = np.dot(embedding, embedding_2)
        print(f"文本1与文本2的余弦相似度: {similarity}")
    else:
        print("BCE (替代) 模型加载失败，无法进行测试。")


