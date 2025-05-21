# faiss_utils.py
import faiss
import numpy as np
import os
import logging

FAISS_INDEX_PATH = os.path.join("data", "album_faiss.index")
# 假设Chinese-CLIP输出768维 (ViT-H-14), BCE输出512维
# 您需要根据实际使用的CLIP模型调整这里的维度
CLIP_EMBEDDING_DIM = 1024 # 请根据您的 Chinese-CLIP ViT-H-14 模型输出维度确认 (通常是768或1024)
BCE_EMBEDDING_DIM = 768
TOTAL_EMBEDDING_DIM = CLIP_EMBEDDING_DIM + BCE_EMBEDDING_DIM

faiss_index = None
use_gpu = False # 在SOC设备上通常不使用GPU进行FAISS，除非有特定硬件支持
# gpu_res = None # 如果要用GPU，取消注释

def init_faiss_index():
    global faiss_index #, gpu_res
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            logging.info(f"正在从 {FAISS_INDEX_PATH} 加载现有的FAISS索引...")
            cpu_index = faiss.read_index(FAISS_INDEX_PATH)
            if use_gpu:
                # gpu_res = faiss.StandardGpuResources()
                # faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                pass # 当前禁用GPU
            else:
                faiss_index = cpu_index
            logging.info(f"FAISS索引加载成功。索引中向量数量: {faiss_index.ntotal}, 维度: {faiss_index.d}")
            if faiss_index.d != TOTAL_EMBEDDING_DIM:
                logging.warning(f"警告: 加载的FAISS索引维度 ({faiss_index.d}) 与期望维度 ({TOTAL_EMBEDDING_DIM}) 不符! 可能需要重建索引。")
                # 考虑是否在此处强制重建或抛出错误
                faiss_index = None # 清除不匹配的索引
        
        if faiss_index is None: # 如果加载失败或文件不存在或维度不匹配
            logging.info(f"未找到兼容的FAISS索引或需要重建。正在创建新的FAISS索引，维度: {TOTAL_EMBEDDING_DIM}")
            # 使用 IndexFlatIP (内积) 进行相似度搜索，因为CLIP和BCE向量通常已归一化
            # IndexFlatL2 对应欧氏距离
            cpu_quantizer = faiss.IndexFlatIP(TOTAL_EMBEDDING_DIM)
            # IndexIDMap2 允许我们使用自定义的ID
            cpu_index = faiss.IndexIDMap2(cpu_quantizer)
            if use_gpu:
                # gpu_res = faiss.StandardGpuResources()
                # faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                pass # 当前禁用GPU
            else:
                faiss_index = cpu_index
            logging.info("新的FAISS索引创建成功。")

    except Exception as e:
        logging.error(f"初始化或加载FAISS索引失败: {e}")
        # 创建一个空的备用索引
        cpu_quantizer = faiss.IndexFlatIP(TOTAL_EMBEDDING_DIM)
        faiss_index = faiss.IndexIDMap2(cpu_quantizer)
        logging.info("已创建一个空的备用FAISS索引。")


def add_vector_to_index(vector: np.ndarray, vector_id: int):
    global faiss_index
    if faiss_index is None:
        logging.error("FAISS索引未初始化。无法添加向量。")
        return False
    try:
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0) # (1, dim)
        if vector.shape[1] != TOTAL_EMBEDDING_DIM:
            logging.error(f"向量维度 ({vector.shape[1]}) 与索引期望维度 ({TOTAL_EMBEDDING_DIM}) 不符。")
            return False
        
        vector_id_np = np.array([vector_id], dtype='int64')
        faiss_index.add_with_ids(vector.astype(np.float32), vector_id_np)
        logging.info(f"向量 (ID: {vector_id}) 已添加到FAISS索引。当前索引大小: {faiss_index.ntotal}")
        return True
    except Exception as e:
        logging.error(f"添加向量到FAISS索引失败 (ID: {vector_id}): {e}")
        return False

def update_vector_in_index(vector: np.ndarray, vector_id: int):
    """更新索引中的向量。FAISS通常通过先删除再添加实现。"""
    global faiss_index
    if faiss_index is None:
        logging.error("FAISS索引未初始化。无法更新向量。")
        return False
    try:
        # 1. 删除旧向量 (如果存在)
        faiss_index.remove_ids(np.array([vector_id], dtype='int64'))
        logging.debug(f"从FAISS中移除了旧向量 (ID: {vector_id}) 以便更新。")
        
        # 2. 添加新向量
        return add_vector_to_index(vector, vector_id)
    except Exception as e:
        logging.error(f"更新FAISS索引中的向量失败 (ID: {vector_id}): {e}")
        return False


def search_vectors_in_index(query_vector: np.ndarray, top_k: int = 10):
    global faiss_index
    if faiss_index is None or faiss_index.ntotal == 0:
        logging.warning("FAISS索引未初始化或为空。")
        return [], []
    try:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        actual_top_k = min(top_k, faiss_index.ntotal)
        if actual_top_k == 0: return [], []

        distances, indices = faiss_index.search(query_vector.astype(np.float32), actual_top_k)
        return distances[0].tolist(), indices[0].tolist() # 返回单个查询的结果
    except Exception as e:
        logging.error(f"在FAISS中搜索向量失败: {e}")
        return [], []

def save_faiss_index():
    global faiss_index
    if faiss_index is None:
        logging.error("FAISS索引未初始化。无法保存。")
        return
    try:
        if use_gpu:
            # cpu_index = faiss.index_gpu_to_cpu(faiss_index)
            pass # 当前禁用GPU
        else:
            cpu_index = faiss_index
        faiss.write_index(cpu_index, FAISS_INDEX_PATH)
        logging.info(f"FAISS索引已保存到 {FAISS_INDEX_PATH}")
    except Exception as e:
        logging.error(f"保存FAISS索引失败: {e}")

def get_faiss_index_ntotal():
    if faiss_index:
        return faiss_index.ntotal
    return 0

# 初始化时加载/创建索引
init_faiss_index()

if __name__ == '__main__':
    print(f"FAISS索引初始化完毕。当前向量数: {get_faiss_index_ntotal()}, 维度: {TOTAL_EMBEDDING_DIM}")
    
    # 测试添加
    # test_id_1 = 1001
    # test_vec_1 = np.random.rand(TOTAL_EMBEDDING_DIM).astype(np.float32)
    # test_vec_1 /= np.linalg.norm(test_vec_1) # 归一化
    # add_vector_to_index(test_vec_1, test_id_1)

    # test_id_2 = 1002
    # test_vec_2 = np.random.rand(TOTAL_EMBEDDING_DIM).astype(np.float32)
    # test_vec_2 /= np.linalg.norm(test_vec_2)
    # add_vector_to_index(test_vec_2, test_id_2)
    
    # print(f"添加后向量数: {get_faiss_index_ntotal()}")

    # # 测试搜索
    # if get_faiss_index_ntotal() > 0:
    #     query_vec = test_vec_1 # 用自身搜索
    #     distances, indices = search_vectors_in_index(query_vec, top_k=5)
    #     print("\n搜索结果:")
    #     for dist, idx in zip(distances, indices):
    #         print(f"  ID: {idx}, 相似度 (内积): {dist:.4f}")
    
    # # 测试更新
    # test_vec_1_updated = np.random.rand(TOTAL_EMBEDDING_DIM).astype(np.float32)
    # test_vec_1_updated /= np.linalg.norm(test_vec_1_updated)
    # update_vector_in_index(test_vec_1_updated, test_id_1)
    # print(f"更新后向量数: {get_faiss_index_ntotal()}")

    # if get_faiss_index_ntotal() > 0:
    #     distances_updated, indices_updated = search_vectors_in_index(test_vec_1_updated, top_k=5)
    #     print("\n更新后用新向量搜索结果:")
    #     for dist, idx in zip(distances_updated, indices_updated):
    #         print(f"  ID: {idx}, 相似度: {dist:.4f}")
    
    # # 保存索引
    # save_faiss_index()


