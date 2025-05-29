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
import tempfile # Added for temporary file handling

logging.info("welcome using sophgo smart album!")

# --- 项目路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHINESE_CLIP_DIR = os.path.join(CURRENT_DIR, "Chinese-CLIP")
if CHINESE_CLIP_DIR not in sys.path:
    sys.path.append(CHINESE_CLIP_DIR)

# --- 服务导入 ---
try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name # available_models (not used)
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
CORS(app) # Allow all origins by default

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- 全局配置和常量 ---
CLIP_MODEL_NAME = "ViT-H-14" 
CLIP_MODEL_DOWNLOAD_ROOT = os.path.join(CURRENT_DIR, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"PLEASE NOTE USING {DEVICE} NOW!")

UPLOADS_DIR = os.path.join(CURRENT_DIR, "uploads")
THUMBNAILS_DIR = os.path.join(CURRENT_DIR, "thumbnails")
TEMP_SEARCH_UPLOADS_DIR = os.path.join(UPLOADS_DIR, "temp_search") # For temporary search images
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
os.makedirs(TEMP_SEARCH_UPLOADS_DIR, exist_ok=True) # Create temp dir for search image
os.makedirs(os.path.join(CURRENT_DIR, "data"), exist_ok=True) 
os.makedirs(CLIP_MODEL_DOWNLOAD_ROOT, exist_ok=True)

clip_model = None
clip_preprocess = None


# --- 应用配置 (可持久化或从配置文件加载) ---
APP_CONFIG_FILE = os.path.join(CURRENT_DIR, "data", "app_config.json")
default_app_config = {
    "qwen_vl_analysis_enabled": True,
    "use_enhanced_search": True
}
app_config = {}

def load_app_config():
    global app_config
    if os.path.exists(APP_CONFIG_FILE):
        try:
            with open(APP_CONFIG_FILE, 'r', encoding='utf-8') as f:
                app_config = json.load(f)
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
        os.makedirs(os.path.dirname(APP_CONFIG_FILE), exist_ok=True)
        with open(APP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(app_config, f, indent=4, ensure_ascii=False)
        logging.info(f"应用配置已保存到 {APP_CONFIG_FILE}。")
    except Exception as e:
        logging.error(f"保存配置到 {APP_CONFIG_FILE} 失败: {e}")


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
        logging.error(f"加载 Chinese-CLIP 模型失败: {e}", exc_info=True)
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
        return feat.cpu().numpy().astype(np.float32)[0] 
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
        return feat.cpu().numpy().astype(np.float32)[0] 
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
    if not file_extension or file_extension not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        logging.warning(f"不支持的文件扩展名: {file_extension} for file {original_filename}. 跳过。")
        return None

    unique_filename_base = str(uuid.uuid4())
    saved_original_filename = unique_filename_base + file_extension
    original_path = os.path.join(UPLOADS_DIR, saved_original_filename)
    
    saved_thumbnail_filename = unique_filename_base + "_thumb.jpg" 
    thumbnail_path_abs = os.path.join(THUMBNAILS_DIR, saved_thumbnail_filename)
    
    image_db_id = None
    actual_faiss_id = None 

    try:
        file_storage.save(original_path)
        logging.info(f"图片 '{original_filename}' 已保存到 '{original_path}'")

        if not generate_thumbnail(original_path, thumbnail_path_abs):
            thumbnail_path_abs = None 

        clip_img_emb = compute_clip_image_embedding(original_path)
        if clip_img_emb is None:
            logging.error(f"无法为图片 '{original_filename}' 生成CLIP embedding。跳过此图片。")
            if os.path.exists(original_path): os.remove(original_path)
            if thumbnail_path_abs and os.path.exists(thumbnail_path_abs): os.remove(thumbnail_path_abs)
            return None

        zeros_bce_emb = np.zeros(fu.BCE_EMBEDDING_DIM, dtype=np.float32)
        concatenated_emb = np.concatenate((clip_img_emb, zeros_bce_emb))
        
        image_db_id = db.add_image_to_db(
            original_filename=original_filename,
            original_path=original_path, 
            thumbnail_path=thumbnail_path_abs, 
            clip_embedding=clip_img_emb 
        )

        if image_db_id:
            actual_faiss_id = image_db_id 
            db.update_faiss_id_for_image(image_db_id, actual_faiss_id) 

            if fu.add_vector_to_index(concatenated_emb, actual_faiss_id):
                logging.info(f"图片 '{original_filename}' (DB ID: {image_db_id}, FAISS ID: {actual_faiss_id}) 处理完成并入库。")
                
                if app_config.get("qwen_vl_analysis_enabled", True):
                    logging.info(f"全局Qwen-VL分析已开启，开始分析图片 ID: {image_db_id}")
                    qwen_result = qwen_service.analyze_image_content(original_path)
                    if qwen_result and (qwen_result.get("description") or qwen_result.get("keywords")):
                        db.update_image_enhancement(image_db_id, qwen_result["description"], qwen_result["keywords"])
                        
                        bce_desc_emb = bce_service.get_bce_embedding(qwen_result["description"]) 
                        if bce_desc_emb is not None and bce_desc_emb.shape[0] == fu.BCE_EMBEDDING_DIM:
                            updated_concatenated_emb = np.concatenate((clip_img_emb, bce_desc_emb))
                            fu.update_vector_in_index(updated_concatenated_emb, actual_faiss_id) 
                            logging.info(f"图片 ID: {image_db_id} Qwen-VL分析完成并更新了FAISS向量。")
                        else:
                            logging.warning(f"图片 ID: {image_db_id} Qwen-VL分析的BCE embedding生成失败或维度不符。FAISS向量未更新BCE部分。")
                    else:
                        logging.warning(f"图片 ID: {image_db_id} Qwen-VL分析未返回有效结果。")
                
                return {
                    "id": image_db_id, 
                    "faiss_id": actual_faiss_id, 
                    "filename": original_filename, 
                    "status": "success"
                }
            else: 
                logging.error(f"图片 '{original_filename}' 添加到FAISS失败。回滚数据库记录。")
                db.hard_delete_image_from_db(image_db_id) 
        else: 
            logging.error(f"图片 '{original_filename}' 存入数据库失败。")

        if os.path.exists(original_path): os.remove(original_path)
        if thumbnail_path_abs and os.path.exists(thumbnail_path_abs): os.remove(thumbnail_path_abs)
        return None

    except Exception as e:
        logging.error(f"处理上传图片 '{original_filename}' 时发生严重错误: {e}", exc_info=True)
        if image_db_id: 
            if actual_faiss_id and fu.faiss_index is not None: 
                try: 
                    fu.faiss_index.remove_ids(np.array([actual_faiss_id], dtype='int64'))
                    logging.info(f"FAISS ID {actual_faiss_id} removed during error cleanup.")
                except Exception as fe: logging.error(f"处理错误后FAISS回滚失败: {fe}")
            db.hard_delete_image_from_db(image_db_id) 
        
        if os.path.exists(original_path): os.remove(original_path)
        if thumbnail_path_abs and os.path.exists(thumbnail_path_abs): os.remove(thumbnail_path_abs)
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
    elif failed_count > 0 and not processed_results: 
         return jsonify({"error": f"所有 {failed_count} 张有效图片处理均失败。"}), 500
    else: 
        return jsonify({"error": "未提供有效文件进行处理。"}), 400


@app.route('/search_images', methods=['POST'])
def search_images_api():
    if not clip_model or fu.faiss_index is None:
        return jsonify({"error": "模型或FAISS索引未初始化。"}), 503
    
    data = request.get_json()
    if not data: return jsonify({"error": "请求数据为空"}), 400

    query_text = data.get('query_text', '').strip()
    top_k = int(data.get('top_k', 200)) 
    if not query_text:
        return jsonify({"error": "查询文本不能为空"}), 400

    query_clip_emb = compute_clip_text_embedding(query_text) 
    if query_clip_emb is None:
        return jsonify({"error": "无法计算查询文本的CLIP embedding"}), 500

    use_enhanced = app_config.get("use_enhanced_search", True)
    logging.info(f"搜索模式: {'增强搜索' if use_enhanced else '仅CLIP搜索'}")

    if use_enhanced:
        query_bce_emb = bce_service.get_bce_embedding(query_text)
        if query_bce_emb is None or query_bce_emb.shape[0] != fu.BCE_EMBEDDING_DIM:
            logging.warning(f"增强搜索的BCE embedding生成失败或维度不符。将使用零向量代替BCE部分。")
            query_bce_emb = np.zeros(fu.BCE_EMBEDDING_DIM, dtype=np.float32)
    else:
        query_bce_emb = np.zeros(fu.BCE_EMBEDDING_DIM, dtype=np.float32)
    
    concatenated_query_emb = np.concatenate((query_clip_emb, query_bce_emb))
    
    distances, faiss_ids = fu.search_vectors_in_index(concatenated_query_emb, top_k=top_k)
    results = []
    if not faiss_ids: 
        return jsonify({
            "query": query_text, 
            "results": [], 
            "message": "未找到匹配图片。",
            "search_mode_is_enhanced": use_enhanced
        }), 200

    for i in range(len(faiss_ids)):
        faiss_id_val = int(faiss_ids[i])
        similarity = float(distances[i]) 
        
        image_data = db.get_image_by_faiss_id(faiss_id_val) 
        if image_data:
            try:
                keywords_list = json.loads(image_data["qwen_keywords"]) if image_data["qwen_keywords"] else []
            except (json.JSONDecodeError, TypeError):
                keywords_list = []
            
            try:
                user_tags_list = json.loads(image_data["user_tags"]) if image_data["user_tags"] else []
            except (json.JSONDecodeError, TypeError):
                user_tags_list = []


            results.append({
                "id": image_data["id"],
                "faiss_id": image_data["faiss_id"],
                "filename": image_data["original_filename"],
                "thumbnail_url": f"/thumbnails/{os.path.basename(image_data['thumbnail_path'])}" if image_data["thumbnail_path"] and os.path.exists(image_data['thumbnail_path']) else None,
                "original_url": f"/uploads/{os.path.basename(image_data['original_path'])}" if image_data["original_path"] and os.path.exists(image_data['original_path']) else None,
                "similarity": similarity, 
                "qwen_description": image_data["qwen_description"],
                "qwen_keywords": keywords_list,
                "user_tags": user_tags_list, 
                "is_enhanced": image_data["is_enhanced"]
            })
        else:
            logging.warning(f"在数据库中未找到FAISS ID为 {faiss_id_val} 的图片记录 (可能已被删除或不同步)。")
    
    return jsonify({
        "query": query_text, 
        "results": results,
        "search_mode_is_enhanced": use_enhanced
    }), 200

# --- NEW: Image-to-Image Search API ---
@app.route('/search_by_uploaded_image', methods=['POST'])
def search_by_uploaded_image_api():
    if not clip_model:
        return jsonify({"error": "CLIP模型未初始化。"}), 503
    
    if 'image_query_file' not in request.files:
        return jsonify({"error": "请求中未找到图片文件(image_query_file key missing)"}), 400
    
    uploaded_file = request.files['image_query_file']
    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({"error": "未选择图片文件进行搜索"}), 400

    logging.info(f"开始图搜图处理，上传文件名: {uploaded_file.filename}")

    tmp_file_path = None
    query_clip_emb = None
    try:
        # Save to a temporary file to pass its path to compute_clip_image_embedding
        # Suffix helps Pillow determine file type, though compute_clip_image_embedding converts to RGB
        suffix = os.path.splitext(uploaded_file.filename)[1] if os.path.splitext(uploaded_file.filename)[1] else '.tmp'
        with tempfile.NamedTemporaryFile(dir=TEMP_SEARCH_UPLOADS_DIR, suffix=suffix, delete=False) as tmp_file:
            uploaded_file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        logging.info(f"图搜图: 临时文件已保存到 {tmp_file_path}")
        query_clip_emb = compute_clip_image_embedding(tmp_file_path)

    except Exception as e:
        logging.error(f"图搜图: 处理上传图片时发生错误: {e}", exc_info=True)
        return jsonify({"error": f"处理上传图片失败: {e}"}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                logging.info(f"图搜图: 临时文件 {tmp_file_path} 已删除。")
            except Exception as e_remove:
                logging.warning(f"图搜图: 无法删除临时文件 {tmp_file_path}: {e_remove}")

    if query_clip_emb is None:
        return jsonify({"error": "无法为上传的图片计算CLIP embedding"}), 500

    # Get all valid CLIP embeddings from the database
    # This list contains dicts: {'id': db_id, 'faiss_id': faiss_id, 'clip_embedding': np.array}
    all_db_images_data = db.get_all_valid_images_clip_embeddings()
    if not all_db_images_data:
        return jsonify({"query_filename": uploaded_file.filename, "results": [], "message": "数据库中没有可比较的图片。"}), 200

    db_clip_embeddings_list = [item['clip_embedding'] for item in all_db_images_data]
    # Stack into a matrix for batch dot product: (N, D)
    db_clip_embeddings_matrix = np.array(db_clip_embeddings_list)

    # Calculate cosine similarities (dot product, since embeddings are normalized)
    # query_clip_emb is (D,), db_clip_embeddings_matrix is (N, D)
    # Result of dot product will be (N,)
    similarities = np.dot(db_clip_embeddings_matrix, query_clip_emb) # Already normalized

    results = []
    SIMILARITY_THRESHOLD = 0.6 # As per requirement

    for i, image_data_item in enumerate(all_db_images_data):
        similarity_score = float(similarities[i])
        
        if similarity_score > SIMILARITY_THRESHOLD:
            # Fetch full image details from DB using its ID
            image_full_details = db.get_image_by_id(image_data_item["id"])
            if image_full_details:
                try:
                    keywords_list = json.loads(image_full_details["qwen_keywords"]) if image_full_details["qwen_keywords"] else []
                except (json.JSONDecodeError, TypeError):
                    keywords_list = []
                try:
                    user_tags_list = json.loads(image_full_details["user_tags"]) if image_full_details["user_tags"] else []
                except (json.JSONDecodeError, TypeError):
                    user_tags_list = []
                
                results.append({
                    "id": image_full_details["id"],
                    "faiss_id": image_full_details["faiss_id"], # Though not used for this search, good to return
                    "filename": image_full_details["original_filename"],
                    "thumbnail_url": f"/thumbnails/{os.path.basename(image_full_details['thumbnail_path'])}" if image_full_details["thumbnail_path"] and os.path.exists(image_full_details['thumbnail_path']) else None,
                    "original_url": f"/uploads/{os.path.basename(image_full_details['original_path'])}" if image_full_details["original_path"] and os.path.exists(image_full_details['original_path']) else None,
                    "similarity": similarity_score, 
                    "qwen_description": image_full_details["qwen_description"],
                    "qwen_keywords": keywords_list,
                    "user_tags": user_tags_list,
                    "is_enhanced": image_full_details["is_enhanced"]
                })
            else:
                logging.warning(f"图搜图: 在数据库中未找到图片ID为 {image_data_item['id']} 的详细记录，尽管其embedding存在。")

    # Sort results by similarity, descending
    results.sort(key=lambda x: x["similarity"], reverse=True)

    logging.info(f"图搜图: 为 '{uploaded_file.filename}' 找到 {len(results)} 张相似图片 (阈值 > {SIMILARITY_THRESHOLD})。")
    
    return jsonify({
        "query_filename": uploaded_file.filename,
        "results": results,
        "search_mode_is_enhanced": False # This is pure CLIP image search
    }), 200


@app.route('/image_details/<int:image_db_id>', methods=['GET'])
def get_image_details_api(image_db_id):
    image_data = db.get_image_by_id(image_db_id) 
    if not image_data:
        return jsonify({"error": f"图片 ID {image_db_id} 未找到"}), 404
    
    try:
        keywords_list = json.loads(image_data["qwen_keywords"]) if image_data["qwen_keywords"] else []
    except (json.JSONDecodeError, TypeError):
        keywords_list = []
    
    try:
        user_tags_list = json.loads(image_data["user_tags"]) if image_data["user_tags"] else []
    except (json.JSONDecodeError, TypeError): 
        user_tags_list = []

    details = {
        "id": image_data["id"],
        "filename": image_data["original_filename"],
        "original_url": f"/uploads/{os.path.basename(image_data['original_path'])}" if image_data['original_path'] and os.path.exists(image_data['original_path']) else None,
        "qwen_description": image_data["qwen_description"] or "无",
        "qwen_keywords": keywords_list,
        "user_tags": user_tags_list, 
        "is_enhanced": image_data["is_enhanced"],
    }
    return jsonify(details), 200


@app.route('/config/settings', methods=['GET', 'POST'])
def handle_app_settings():
    global app_config
    if request.method == 'GET':
        for key, value in default_app_config.items():
            app_config.setdefault(key, value)
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


@app.route('/enhance_image/<int:image_db_id>', methods=['POST'])
def enhance_single_image_api(image_db_id):
    image_data = db.get_image_by_id(image_db_id)
    if not image_data:
        return jsonify({"error": f"图片 ID {image_db_id} 未找到"}), 404
    if image_data["is_enhanced"]: 
        return jsonify({
            "message": f"图片 ID {image_db_id} 已经分析过了。",
            "qwen_description": image_data["qwen_description"],
            "qwen_keywords": json.loads(image_data["qwen_keywords"] or "[]"),
            "is_enhanced": True
            }), 200

    original_path = image_data["original_path"]
    if not original_path or not os.path.exists(original_path):
        return jsonify({"error": f"图片 ID {image_db_id} 的原始文件路径无效或文件不存在。"}), 404

    clip_img_emb = db.get_clip_embedding_for_image(image_db_id) 
    if clip_img_emb is None:
        logging.warning(f"图片ID {image_db_id} 在数据库中未找到纯CLIP embedding，尝试重新计算...")
        clip_img_emb = compute_clip_image_embedding(original_path)
        if clip_img_emb is None: 
            return jsonify({"error": f"无法获取或计算图片 ID {image_db_id} 的CLIP embedding"}), 500

    logging.info(f"手动触发对图片 ID: {image_db_id} ({original_path}) 的Qwen-VL分析。")
    qwen_result = qwen_service.analyze_image_content(original_path)

    if qwen_result and (qwen_result.get("description") or qwen_result.get("keywords")):
        update_success = db.update_image_enhancement(image_db_id, qwen_result["description"], qwen_result["keywords"])
        if not update_success:
             return jsonify({"error": f"图片 ID {image_db_id} 分析结果存入数据库失败。"}), 500

        bce_desc_emb = bce_service.get_bce_embedding(qwen_result["description"])
        if bce_desc_emb is not None and bce_desc_emb.shape[0] == fu.BCE_EMBEDDING_DIM:
            updated_concatenated_emb = np.concatenate((clip_img_emb, bce_desc_emb))
        else: 
            logging.warning(f"BCE embedding for description failed or wrong dim for image {image_db_id}. Using zero vector for BCE part in FAISS.")
            zeros_bce_emb = np.zeros(fu.BCE_EMBEDDING_DIM, dtype=np.float32)
            updated_concatenated_emb = np.concatenate((clip_img_emb, zeros_bce_emb))


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
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int) 
    images, total_count = db.get_all_images(page, limit) 
    results = []
    for img_row in images:
        original_path_value = img_row['original_path']
        thumbnail_path_value = img_row['thumbnail_path']
        try:
            user_tags_list = json.loads(img_row["user_tags"]) if img_row["user_tags"] else []
        except (json.JSONDecodeError, TypeError):
            user_tags_list = []

        results.append({
            "id": img_row["id"],
            "filename": img_row["original_filename"],
            "thumbnail_url": f"/thumbnails/{os.path.basename(thumbnail_path_value)}" if thumbnail_path_value and os.path.exists(thumbnail_path_value) else None,
            "original_url": f"/uploads/{os.path.basename(original_path_value)}" if original_path_value and os.path.exists(original_path_value) else None,
            "is_enhanced": img_row["is_enhanced"],
            "user_tags": user_tags_list, 
        })
    return jsonify({
        "images": results,
        "total_count": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit if limit > 0 else 0
    })

@app.route('/delete_images_batch', methods=['POST'])
def delete_images_batch_api():
    data = request.get_json()
    if not data or 'image_ids' not in data or not isinstance(data['image_ids'], list):
        return jsonify({"error": "无效的请求数据，需要包含 image_ids 列表。"}), 400

    image_ids_to_delete = data['image_ids']
    if not image_ids_to_delete:
        return jsonify({"error": "未提供要删除的图片ID。"}), 400

    deleted_count = 0
    failed_ids = []
    faiss_ids_to_remove_from_index = []

    for image_id in image_ids_to_delete:
        try:
            image_id_int = int(image_id)
            original_path, thumbnail_path, faiss_id = db.get_image_paths_and_faiss_id(image_id_int)

            if original_path and os.path.exists(original_path):
                os.remove(original_path)
                logging.info(f"已删除原始图片文件: {original_path}")
            else:
                logging.warning(f"原始图片文件未找到或路径无效: {original_path} for ID {image_id_int}")
            
            if thumbnail_path and os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                logging.info(f"已删除缩略图文件: {thumbnail_path}")
            else:
                logging.warning(f"缩略图文件未找到或路径无效: {thumbnail_path} for ID {image_id_int}")

            if faiss_id is not None:
                faiss_ids_to_remove_from_index.append(faiss_id)
            
            if db.hard_delete_image_from_db(image_id_int):
                deleted_count += 1
            else:
                failed_ids.append(image_id_int)
                logging.error(f"从数据库删除图片 ID {image_id_int} 失败。")

        except ValueError:
            logging.error(f"无效的图片ID格式: {image_id}")
            failed_ids.append(image_id)
        except Exception as e:
            logging.error(f"删除图片 ID {image_id} 时发生错误: {e}", exc_info=True)
            failed_ids.append(image_id)
    
    if faiss_ids_to_remove_from_index:
        if fu.faiss_index is not None and fu.faiss_index.ntotal > 0:
            try:
                logging.info(f"准备从FAISS中删除ID列表: {faiss_ids_to_remove_from_index}")
                ids_to_remove_np = np.array(faiss_ids_to_remove_from_index, dtype=np.int64)
                num_removed = fu.faiss_index.remove_ids(ids_to_remove_np)
                logging.info(f"从FAISS索引中移除了 {num_removed} 个向量。")
                if num_removed > 0 : fu.save_faiss_index()
            except Exception as e_faiss_remove:
                logging.error(f"从FAISS索引移除向量时出错: {e_faiss_remove}")
        else:
            logging.warning("FAISS索引为空或未初始化，跳过FAISS删除。")


    if deleted_count > 0:
        return jsonify({
            "success": True, 
            "message": f"成功删除 {deleted_count} 张图片。",
            "failed_ids": failed_ids
        }), 200
    else:
        return jsonify({
            "success": False, 
            "error": "未能删除任何图片。",
            "failed_ids": failed_ids
        }), 500

@app.route('/add_user_tags_batch', methods=['POST'])
def add_user_tags_batch_api():
    data = request.get_json()
    if not data or 'image_ids' not in data or not isinstance(data['image_ids'], list) \
            or 'user_tags' not in data or not isinstance(data['user_tags'], list):
        return jsonify({"error": "无效的请求数据，需要包含 image_ids (列表) 和 user_tags (字符串列表)。"}), 400

    image_ids_to_tag = data['image_ids']
    user_tags_to_add = [str(tag).strip() for tag in data['user_tags'] if str(tag).strip()] 

    if not image_ids_to_tag:
        return jsonify({"error": "未提供要标记的图片ID。"}), 400
    if not user_tags_to_add:
        return jsonify({"error": "未提供有效的用户标签。"}), 400
    
    updated_count = 0
    failed_ids = []

    for image_id in image_ids_to_tag:
        try:
            image_id_int = int(image_id)
            if db.update_user_tags_for_image(image_id_int, user_tags_to_add):
                updated_count += 1
            else:
                failed_ids.append(image_id_int)
        except ValueError:
            logging.error(f"无效的图片ID格式: {image_id} for tagging.")
            failed_ids.append(image_id)
        except Exception as e:
            logging.error(f"为图片 ID {image_id} 添加标签时发生错误: {e}", exc_info=True)
            failed_ids.append(image_id)
            
    if updated_count > 0:
        return jsonify({
            "success": True, 
            "message": f"成功为 {updated_count} 张图片添加/更新了用户标签。",
            "failed_ids": failed_ids
        }), 200
    else:
        return jsonify({
            "success": False, 
            "error": "未能为任何图片添加/更新用户标签。",
            "failed_ids": failed_ids
        }), 500


# --- 静态文件服务 ---
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "非法路径"}), 400
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "非法路径"}), 400
    return send_from_directory(THUMBNAILS_DIR, filename)

# --- 应用启动 ---
if __name__ == '__main__':
    load_app_config()
    db.init_db() 
    fu.init_faiss_index() 
    load_clip_model_on_startup() 
    
    if not bce_service.bce_model: 
        logging.warning("BCE模型在bce_service中未能加载。请检查日志。")
    if not clip_model:
        logging.warning("CLIP模型未能加载。请检查日志。")

    logging.info("智能相册后端服务准备启动...")
    app.run(host="0.0.0.0", port=5000, debug=True)