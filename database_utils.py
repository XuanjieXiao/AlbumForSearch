# database_utils.py
import sqlite3
import json
import logging
import os
import numpy as np
from datetime import datetime

DATABASE_PATH = os.path.join("data", "smart_album.db")
# UPLOADS_DIR = "uploads" # 不在此文件直接使用
# THUMBNAILS_DIR = "thumbnails" # 不在此文件直接使用

# --- 辅助函数，用于Numpy数组和BLOB的转换 ---
def adapt_array(arr):
    return json.dumps(arr.tolist())

def convert_array(text):
    return np.array(json.loads(text), dtype=np.float32)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)


def get_db_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            original_path TEXT NOT NULL UNIQUE,
            thumbnail_path TEXT UNIQUE,
            faiss_id INTEGER UNIQUE, -- 允许为NULL，后续更新
            clip_embedding ARRAY,
            qwen_description TEXT,
            qwen_keywords TEXT,
            user_tags TEXT,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_enhanced BOOLEAN DEFAULT FALSE,
            last_enhanced_timestamp TIMESTAMP,
            deleted BOOLEAN DEFAULT FALSE
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faiss_id ON images (faiss_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_original_path ON images (original_path)")
    conn.commit()
    conn.close()
    logging.info("数据库初始化/检查完毕。")

def add_image_to_db(original_filename: str, original_path: str, thumbnail_path: str | None, clip_embedding: np.ndarray):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # faiss_id 初始可以不设置或设置为NULL，后续通过 update_faiss_id_for_image 更新
        cursor.execute('''
            INSERT INTO images (original_filename, original_path, thumbnail_path, clip_embedding, is_enhanced)
            VALUES (?, ?, ?, ?, ?)
        ''', (original_filename, original_path, thumbnail_path, clip_embedding, False))
        conn.commit()
        image_id = cursor.lastrowid
        logging.info(f"图片 '{original_filename}' (ID: {image_id}) 已初步添加到数据库。FAISS ID 待更新。")
        return image_id
    except sqlite3.IntegrityError as e:
        logging.error(f"添加图片 '{original_filename}' 到数据库失败 (路径可能已存在): {original_path}. Error: {e}")
        return None
    finally:
        conn.close()

def update_faiss_id_for_image(image_id: int, faiss_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET faiss_id = ? WHERE id = ?", (faiss_id, image_id))
        conn.commit()
        logging.info(f"数据库中图片 ID {image_id} 的 FAISS ID 已更新为 {faiss_id}")
        return True
    except Exception as e:
        logging.error(f"更新图片 ID {image_id} 的 FAISS ID 失败: {e}")
        return False
    finally:
        conn.close()

def delete_image_from_db(image_id: int):
    """ 真实删除数据库记录 (用于上传失败时的回滚) """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
        logging.info(f"已从数据库中删除图片 ID: {image_id} (硬删除)。")
        return True
    except Exception as e:
        logging.error(f"从数据库删除图片 ID: {image_id} 失败: {e}")
        return False
    finally:
        conn.close()


def update_image_enhancement(image_id: int, description: str, keywords: list):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        keywords_json = json.dumps(keywords, ensure_ascii=False)
        cursor.execute('''
            UPDATE images
            SET qwen_description = ?, qwen_keywords = ?, is_enhanced = TRUE, last_enhanced_timestamp = ?
            WHERE id = ?
        ''', (description, keywords_json, datetime.now(), image_id))
        conn.commit()
        logging.info(f"图片 ID: {image_id} 的增强信息已更新。")
        return True
    except Exception as e:
        logging.error(f"更新图片 ID: {image_id} 增强信息失败: {e}")
        return False
    finally:
        conn.close()

def get_image_by_id(image_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images WHERE id = ? AND deleted = FALSE", (image_id,))
    image_data = cursor.fetchone()
    conn.close()
    return image_data

def get_image_by_faiss_id(faiss_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images WHERE faiss_id = ? AND deleted = FALSE", (faiss_id,))
    image_data = cursor.fetchone()
    conn.close()
    return image_data

def get_all_images(page: int = 1, limit: int = 20):
    conn = get_db_connection()
    cursor = conn.cursor()
    offset = (page - 1) * limit
    cursor.execute("""
        SELECT id, original_filename, original_path, thumbnail_path, qwen_description, is_enhanced
        FROM images
        WHERE deleted = FALSE
        ORDER BY upload_timestamp DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))
    images = cursor.fetchall()

    cursor.execute("SELECT COUNT(id) FROM images WHERE deleted = FALSE")
    total_count = cursor.fetchone()[0]

    conn.close()
    return images, total_count

def get_images_for_enhancement(limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, original_path, clip_embedding FROM images WHERE is_enhanced = FALSE AND deleted = FALSE ORDER BY upload_timestamp ASC LIMIT ?", (limit,))
    images = cursor.fetchall()
    conn.close()
    return images

def get_clip_embedding_for_image(image_id: int) -> np.ndarray | None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT clip_embedding FROM images WHERE id = ? AND deleted = FALSE", (image_id,))
    result = cursor.fetchone()
    conn.close()
    if result and result['clip_embedding'] is not None:
        return result['clip_embedding']
    return None

def get_next_available_db_id(): # 重命名以更准确反映其作用
    conn = get_db_connection()
    cursor = conn.cursor()
    # SQLite的AUTOINCREMENT特性保证ID是递增的。
    # 如果需要精确的下一个ID，可以查询 sqlite_sequence 表
    # 不过，让数据库自己处理自增通常是最好的。
    # 此函数主要用于预估，或者在某些ID管理策略中使用。
    # 对于当前的上传逻辑，我们让DB自己生成ID。
    cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='images'")
    result = cursor.fetchone()
    conn.close()
    if result and result['seq'] is not None:
        return result['seq'] + 1
    # 如果表为空或从未插入过，则下一个ID是1
    # 更简单的做法是依赖 lastrowid, 这里我们不直接用 get_next_faiss_id 来分配ID给新图片了
    return 1 # Fallback, 但实际不应依赖此返回值来插入

# 注意：get_next_faiss_id 在当前实现下不再用于预分配ID给新图片，
# 因为我们改为先插入DB获取ID，再用此ID作为FAISS ID。
# 如果FAISS ID需要独立于DB ID管理，则此函数需要不同的逻辑。

if __name__ == '__main__':
    init_db()
    print("数据库已初始化。")