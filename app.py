# AlbumForSearch/app.py
import os
import sys
import logging
import json
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
import time
# --- 项目路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHINESE_CLIP_DIR = os.path.join(CURRENT_DIR, "Chinese-CLIP")
if CHINESE_CLIP_DIR not in sys.path:
    sys.path.append(CHINESE_CLIP_DIR)

# --- 服务导入 ---
try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name, available_models
    import torch
    logging.info("Chinese_CLIP 包导入成功。")
except ImportError as e:
    logging.error(f"导入 Chinese_CLIP 包失败: {e}. 请确保 Chinese_CLIP 目录结构正确且在PYTHONPATH中。")
    sys.exit(1)

import bce_service
import qwen_service
import database_utils as db
import faiss_utils as fu

# --- Flask 应用初始化 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- 全局配置和常量 ---
# ... (保持不变) ...
CLIP_MODEL_NAME = "ViT-H-14"
CLIP_MODEL_DOWNLOAD_ROOT = os.path.join(CURRENT_DIR, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BCE_OUTPUT_DIM = fu.BCE_EMBEDDING_DIM
FAISS_TOTAL_DIM = fu.TOTAL_EMBEDDING_DIM

UPLOADS_DIR = os.path.join(CURRENT_DIR, "uploads")
THUMBNAILS_DIR = os.path.join(CURRENT_DIR, "thumbnails")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "data"), exist_ok=True)
os.makedirs(CLIP_MODEL_DOWNLOAD_ROOT, exist_ok=True)

clip_model = None
clip_preprocess = None


# --- 应用配置 (可持久化或从配置文件加载) ---
APP_CONFIG_FILE = os.path.join(CURRENT_DIR, "data", "app_config.json")

default_app_config = {
    "qwen_vl_analysis_enabled": True,
    "use_enhanced_search": True # 新增：使用增强搜索的默认值
}

app_config = {}

def load_app_config():
    global app_config
    if os.path.exists(APP_CONFIG_FILE):
        try:
            with open(APP_CONFIG_FILE, 'r') as f:
                app_config = json.load(f)
            #确保所有默认键都存在
            for key, value in default_app_config.items():
                app_config.setdefault(key, value)
            logging.info(f"应用配置已从 {APP_CONFIG_FILE} 加载。")
        except Exception as e:
            logging.error(f"从 {APP_CONFIG_FILE} 加载配置失败: {e}。使用默认配置。")
            app_config = default_app_config.copy()
    else:
        logging.info("未找到应用配置文件。使用默认配置并创建新文件。")
        app_config = default_app_config.copy()
        save_app_config()

def save_app_config():
    global app_config
    try:
        with open(APP_CONFIG_FILE, 'w') as f:
            json.dump(app_config, f, indent=4)
        logging.info(f"应用配置已保存到 {APP_CONFIG_FILE}。")
    except Exception as e:
        logging.error(f"保存配置到 {APP_CONFIG_FILE} 失败: {e}")


# --- 模型加载与核心功能函数 (保持不变) ---
# ... (load_clip_model_on_startup, compute_clip_image_embedding, etc. 保持不变) ...
def load_clip_model_on_startup():
    global clip_model, clip_preprocess
    try:
        logging.info(f"正在加载 Chinese-CLIP 模型: {CLIP_MODEL_NAME} (设备: {DEVICE})")
        clip_model, clip_preprocess = load_from_name(
            CLIP_MODEL_NAME,
            device=DEVICE,
            download_root=CLIP_MODEL_DOWNLOAD_ROOT
        )
        clip_model.eval()

        dummy_img = Image.new('RGB', (224, 224))
        dummy_tensor = clip_preprocess(dummy_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dummy_feat = clip_model.encode_image(dummy_tensor)
        actual_clip_dim = dummy_feat.shape[1]

        if actual_clip_dim != fu.CLIP_EMBEDDING_DIM:
            logging.error(f"致命错误: CLIP模型 '{CLIP_MODEL_NAME}' 的实际输出维度 ({actual_clip_dim}) "
                          f"与 faiss_utils.py 中配置的 CLIP_EMBEDDING_DIM ({fu.CLIP_EMBEDDING_DIM}) 不符。请修正配置。")
        else:
             logging.info(f"Chinese-CLIP 模型 '{CLIP_MODEL_NAME}' 加载成功。图像特征维度: {actual_clip_dim} (与配置一致)")
    except Exception as e:
        logging.error(f"加载 Chinese-CLIP 模型失败: {e}")
        clip_model = None

def compute_clip_image_embedding(image_path: str) -> np.ndarray | None:
    if not clip_model or not clip_preprocess:
        logging.error("CLIP模型未加载，无法计算图像 embedding。")
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
        arr = feat.cpu().numpy().astype(np.float32)
        return arr[0]
    except Exception as e:
        logging.error(f"计算图像 '{image_path}' 的CLIP embedding失败: {e}")
        return None

def compute_clip_text_embedding(text: str) -> np.ndarray | None:
    if not clip_model:
        logging.error("CLIP模型未加载，无法计算文本 embedding。")
        return None
    try:
        tokens = clip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        arr = feat.cpu().numpy().astype(np.float32)
        return arr[0]
    except Exception as e:
        logging.error(f"计算文本 '{text[:20]}...' 的CLIP embedding失败: {e}")
        return None

def generate_thumbnail(image_path: str, thumbnail_path: str, size=(256, 256)):
    try:
        img = Image.open(image_path)
        img.thumbnail(size)
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        img.save(thumbnail_path)
        return True
    except Exception as e:
        logging.error(f"生成缩略图失败 for {image_path}: {e}")
        return False

def process_single_image_upload(file_storage):
    original_filename = file_storage.filename
    logging.info(f"[process_single_image_upload] 开始处理文件: {original_filename}")
    file_extension = os.path.splitext(original_filename)[1].lower()
    unique_filename_base = str(uuid.uuid4())

    if not file_extension or file_extension not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        logging.warning(f"不支持的文件扩展名: {file_extension} for file {original_filename}. 跳过。")
        return None

    saved_original_filename = unique_filename_base + file_extension
    original_path = os.path.join(UPLOADS_DIR, saved_original_filename)
    saved_thumbnail_filename = unique_filename_base + "_thumb" + file_extension
    thumbnail_path = os.path.join(THUMBNAILS_DIR, saved_thumbnail_filename)
    image_db_id = None 
    actual_faiss_id = None

    try:
        file_storage.save(original_path)
        logging.info(f"图片 '{original_filename}' 已保存到 '{original_path}'")

        if not generate_thumbnail(original_path, thumbnail_path):
            thumbnail_path = None

        clip_img_emb = compute_clip_image_embedding(original_path)
        if clip_img_emb is None:
            logging.error(f"无法为图片 '{original_filename}' 生成CLIP embedding。跳过此图片。")
            if os.path.exists(original_path): os.remove(original_path)
            if thumbnail_path and os.path.exists(thumbnail_path): os.remove(thumbnail_path)
            return None

        zeros_bce_emb = np.zeros(BCE_OUTPUT_DIM, dtype=np.float32)
        concatenated_emb = np.concatenate((clip_img_emb, zeros_bce_emb))

        image_db_id = db.add_image_to_db(
            original_filename=original_filename,
            original_path=original_path,
            thumbnail_path=thumbnail_path,
            clip_embedding=clip_img_emb
        )

        if image_db_id:
            actual_faiss_id = image_db_id 
            db.update_faiss_id_for_image(image_db_id, actual_faiss_id)

            if fu.add_vector_to_index(concatenated_emb, actual_faiss_id):
                logging.info(f"图片 '{original_filename}' (DB ID: {image_db_id}, FAISS ID: {actual_faiss_id}) 处理完成并入库。")
                if app_config.get("qwen_vl_analysis_enabled"): # 使用 app_config.get()
                    logging.info(f"全局Qwen-VL分析已开启，开始分析图片 ID: {image_db_id}")
                    qwen_result = qwen_service.analyze_image_content(original_path)
                    if qwen_result and (qwen_result["description"] or qwen_result["keywords"]):
                        db.update_image_enhancement(image_db_id, qwen_result["description"], qwen_result["keywords"])
                        bce_desc_emb = bce_service.get_bce_embedding(qwen_result["description"])
                        updated_concatenated_emb = np.concatenate((clip_img_emb, bce_desc_emb))
                        fu.update_vector_in_index(updated_concatenated_emb, actual_faiss_id)
                        logging.info(f"图片 ID: {image_db_id} Qwen-VL分析完成并更新了FAISS向量。")
                    else:
                        logging.warning(f"图片 ID: {image_db_id} Qwen-VL分析未返回有效结果。")
                return {"id": image_db_id, "faiss_id": actual_faiss_id, "filename": original_filename, "status": "success"}
            else:
                logging.error(f"图片 '{original_filename}' 添加到FAISS失败 (FAISS ID: {actual_faiss_id})。正在回滚数据库记录。")
                db.delete_image_from_db(image_db_id)
        else:
            logging.error(f"图片 '{original_filename}' 存入数据库失败。")

        if os.path.exists(original_path): os.remove(original_path)
        if thumbnail_path and os.path.exists(thumbnail_path): os.remove(thumbnail_path)
        return None
    except Exception as e:
        logging.error(f"处理上传图片 '{original_filename}' 时发生严重错误: {e}", exc_info=True)
        if image_db_id:
            db.delete_image_from_db(image_db_id)
            if actual_faiss_id and fu.faiss_index is not None:
                try: fu.faiss_index.remove_ids(np.array([actual_faiss_id], dtype='int64'))
                except Exception as fe: logging.error(f"处理错误后FAISS回滚失败: {fe}")
        if os.path.exists(original_path): os.remove(original_path)
        if thumbnail_path and os.path.exists(thumbnail_path): os.remove(thumbnail_path)
        return None

# --- API 路由 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/controls')
def controls_page():
    return render_template('controls.html')

@app.route('/upload_images', methods=['POST'])
def upload_images_api():
    # ... (保持不变, process_single_image_upload 已修改)
    if 'files' not in request.files:
        return jsonify({"error": "请求中未找到文件部分(files key missing)"}), 400
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "未选择任何文件"}), 400

    processed_results = []
    failed_count = 0
    for file_storage in files:
        if file_storage and file_storage.filename:
            result = process_single_image_upload(file_storage) 
            if result:
                processed_results.append(result)
            else:
                failed_count += 1
        else:
            failed_count +=1 
    
    if processed_results:
        fu.save_faiss_index()
        return jsonify({
            "message": f"成功处理 {len(processed_results)} 张图片，失败 {failed_count} 张。",
            "processed_files": processed_results
        }), 200
    else:
        return jsonify({"error": f"所有有效图片处理均失败。"}), 500


@app.route('/search_images', methods=['POST'])
def search_images_api():
    # ... (如果 "use_enhanced_search" 需要影响后端，可以在这里读取 app_config['use_enhanced_search'])
    if not clip_model or fu.faiss_index is None:
        return jsonify({"error": "模型或FAISS索引未初始化。"}), 503
    data = request.get_json()
    query_text = data.get('query_text', '').strip()
    top_k = int(data.get('top_k', 200)) 
    if not query_text:
        return jsonify({"error": "查询文本不能为空"}), 400

    # 示例：如果 use_enhanced_search 为 false，可以考虑只用CLIP特征或调整BCE权重
    # if not app_config.get("use_enhanced_search", True):
    #     logging.info("增强搜索已禁用，将调整搜索策略 (示例：可能仅使用CLIP或降低BCE影响)")
        # query_bce_emb = np.zeros(BCE_OUTPUT_DIM, dtype=np.float32) # 例如，如果禁用则BCE部分为0
    # else:
    #     query_bce_emb = bce_service.get_bce_embedding(query_text)
    
    query_clip_emb = compute_clip_text_embedding(query_text)
    if query_clip_emb is None:
        return jsonify({"error": "无法计算查询文本的CLIP embedding"}), 500
    
    query_bce_emb = bce_service.get_bce_embedding(query_text) # 当前始终计算BCE

    concatenated_query_emb = np.concatenate((query_clip_emb, query_bce_emb))
    distances, faiss_ids = fu.search_vectors_in_index(concatenated_query_emb, top_k=top_k)
    results = []
    # ... (其余部分不变)
    if not faiss_ids:
        return jsonify({"query": query_text, "results": [], "message": "未找到匹配图片。"}), 200

    for i in range(len(faiss_ids)):
        faiss_id = int(faiss_ids[i])
        similarity = float(distances[i])
        image_data = db.get_image_by_faiss_id(faiss_id)
        if image_data:
            try:
                keywords_list = json.loads(image_data["qwen_keywords"]) if image_data["qwen_keywords"] else []
            except json.JSONDecodeError:
                keywords_list = []

            results.append({
                "id": image_data["id"],
                "faiss_id": image_data["faiss_id"],
                "filename": image_data["original_filename"],
                "thumbnail_url": f"/thumbnails/{os.path.basename(image_data['thumbnail_path'])}" if image_data["thumbnail_path"] else None,
                "original_url": f"/uploads/{os.path.basename(image_data['original_path'])}" if image_data["original_path"] else None,
                "similarity": similarity,
                "qwen_description": image_data["qwen_description"],
                "qwen_keywords": keywords_list,
                "is_enhanced": image_data["is_enhanced"]
            })
        else:
            logging.warning(f"在数据库中未找到FAISS ID为 {faiss_id} 的图片记录。")
    return jsonify({"query": query_text, "results": results}), 200


@app.route('/image_details/<int:image_db_id>', methods=['GET'])
def get_image_details_api(image_db_id):
    # ... (保持不变)
    image_data = db.get_image_by_id(image_db_id)
    if not image_data:
        return jsonify({"error": f"图片 ID {image_db_id} 未找到"}), 404

    try:
        keywords_list = json.loads(image_data["qwen_keywords"]) if image_data["qwen_keywords"] else []
    except json.JSONDecodeError:
        keywords_list = []
        logging.warning(f"图片ID {image_db_id} 的 qwen_keywords 字段无法解析为JSON: {image_data['qwen_keywords']}")
    
    details = {
        "id": image_data["id"],
        "filename": image_data["original_filename"],
        "original_url": f"/uploads/{os.path.basename(image_data['original_path'])}" if image_data['original_path'] else None,
        "qwen_description": image_data["qwen_description"] or "无",
        "qwen_keywords": keywords_list,
        "is_enhanced": image_data["is_enhanced"],
    }
    return jsonify(details), 200


@app.route('/config/settings', methods=['GET', 'POST'])
def handle_app_settings():
    global app_config
    if request.method == 'GET':
        return jsonify(app_config), 200
    elif request.method == 'POST':
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的JSON数据"}), 400
        
        updated_any = False
        if 'qwen_vl_analysis_enabled' in data and isinstance(data['qwen_vl_analysis_enabled'], bool):
            app_config['qwen_vl_analysis_enabled'] = data['qwen_vl_analysis_enabled']
            logging.info(f"Qwen-VL全局分析状态已更新为: {app_config['qwen_vl_analysis_enabled']}")
            updated_any = True
            
        if 'use_enhanced_search' in data and isinstance(data['use_enhanced_search'], bool):
            app_config['use_enhanced_search'] = data['use_enhanced_search']
            logging.info(f"使用增强搜索状态已更新为: {app_config['use_enhanced_search']}")
            updated_any = True
        
        if updated_any:
            save_app_config()
            return jsonify({"message": "应用设置已更新。", "settings": app_config}), 200
        else:
            return jsonify({"message": "未提供有效设置进行更新。", "settings": app_config}), 200

# 移除旧的 /config/qwen_analysis 端点，因为它被 /config/settings 取代
# @app.route('/config/qwen_analysis', methods=['GET', 'POST'])
# def toggle_qwen_analysis_api(): ...

@app.route('/enhance_image/<int:image_db_id>', methods=['POST'])
def enhance_single_image_api(image_db_id):
    # ... (保持不变)
    image_data = db.get_image_by_id(image_db_id)
    if not image_data:
        return jsonify({"error": f"图片 ID {image_db_id} 未找到"}), 404
    if image_data["is_enhanced"]:
        return jsonify({"message": f"图片 ID {image_db_id} 已经分析过了。"}), 200

    original_path = image_data["original_path"]
    if not original_path or not os.path.exists(original_path): # Check path validity
        return jsonify({"error": f"图片 ID {image_db_id} 的原始文件路径无效或文件不存在。"}), 404

    clip_img_emb = db.get_clip_embedding_for_image(image_db_id)
    if clip_img_emb is None:
        logging.warning(f"图片ID {image_db_id} 在数据库中未找到CLIP embedding，尝试重新计算...")
        clip_img_emb = compute_clip_image_embedding(original_path)
        if clip_img_emb is None:
            return jsonify({"error": f"无法获取或计算图片 ID {image_db_id} 的CLIP embedding"}), 500

    logging.info(f"手动触发对图片 ID: {image_db_id} ({original_path}) 的Qwen-VL分析。")
    qwen_result = qwen_service.analyze_image_content(original_path)

    if qwen_result and (qwen_result["description"] or qwen_result["keywords"]):
        update_success = db.update_image_enhancement(image_db_id, qwen_result["description"], qwen_result["keywords"])
        if not update_success:
             return jsonify({"error": f"图片 ID {image_db_id} 分析结果存入数据库失败。"}), 500

        bce_desc_emb = bce_service.get_bce_embedding(qwen_result["description"])
        updated_concatenated_emb = np.concatenate((clip_img_emb, bce_desc_emb))

        if image_data["faiss_id"] is None: 
            logging.error(f"图片 ID {image_db_id} 在数据库中没有FAISS ID，无法更新FAISS向量。")
            return jsonify({"error": f"图片 ID {image_db_id} 数据不一致，缺少FAISS ID。"}), 500

        if fu.update_vector_in_index(updated_concatenated_emb, image_data["faiss_id"]):
             logging.info(f"图片 ID: {image_db_id} 手动Qwen-VL分析完成并更新了FAISS向量。")
             fu.save_faiss_index()
             return jsonify({
                 "message": f"图片 ID {image_db_id} 分析增强成功。",
                 "qwen_description": qwen_result["description"],
                 "qwen_keywords": qwen_result["keywords"],
                 "is_enhanced": True
                 }), 200
        else: 
            logging.error(f"图片 ID: {image_db_id} FAISS向量更新失败，但DB可能已更新增强状态。")
            return jsonify({
                "error": f"图片 ID {image_db_id} 分析信息已存DB，但FAISS更新失败。",
                "qwen_description": qwen_result["description"],
                "qwen_keywords": qwen_result["keywords"],
                "is_enhanced": True 
                }), 500
    else:
        logging.warning(f"图片 ID: {image_db_id} 手动Qwen-VL分析未返回有效结果。")
        return jsonify({"error": f"图片 ID {image_db_id} 分析未产生有效结果。"}), 500

@app.route('/images', methods=['GET'])
def get_images_list_api():
    # ... (保持不变)
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    images, total_count = db.get_all_images(page, limit)
    results = []
    for img_row in images:
        original_path_value = img_row['original_path']
        thumbnail_path_value = img_row['thumbnail_path']
        results.append({
            "id": img_row["id"],
            "filename": img_row["original_filename"],
            "thumbnail_url": f"/thumbnails/{os.path.basename(thumbnail_path_value)}" if thumbnail_path_value else None,
            "original_url": f"/uploads/{os.path.basename(original_path_value)}" if original_path_value else None,
            "is_enhanced": img_row["is_enhanced"],
        })
    return jsonify({
        "images": results,
        "total_count": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit if limit > 0 else 0
    })

# --- 静态文件服务 ---
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    return send_from_directory(THUMBNAILS_DIR, filename)

# --- 应用启动 ---
if __name__ == '__main__':
    load_app_config() # 加载应用配置
    db.init_db()
    fu.init_faiss_index()
    load_clip_model_on_startup()
    if not bce_service.bce_model:
        logging.warning("BCE模型在bce_service中未能加载。请检查日志。")
    if not clip_model:
        logging.warning("CLIP模型未能加载。请检查日志。")

    logging.info("智能相册后端服务准备启动...")
    app.run(host="0.0.0.0", port=5000, debug=True)