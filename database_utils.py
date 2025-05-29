# database_utils.py
import sqlite3
import json
import logging
import os
import numpy as np
from datetime import datetime

DATABASE_PATH = os.path.join("data", "smart_album.db")

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
    cursor.execute("PRAGMA table_info(images)")
    columns = [column['name'] for column in cursor.fetchall()]
    if 'user_tags' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN user_tags TEXT")
        logging.info("Added 'user_tags' column to 'images' table.")
    if 'deleted' not in columns: 
        cursor.execute("ALTER TABLE images ADD COLUMN deleted BOOLEAN DEFAULT FALSE")
        logging.info("Added 'deleted' column to 'images' table.")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            original_path TEXT NOT NULL UNIQUE,
            thumbnail_path TEXT UNIQUE,
            faiss_id INTEGER UNIQUE, 
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON images (deleted)")

    conn.commit()
    conn.close()
    logging.info("数据库初始化/检查完毕。")

def add_image_to_db(original_filename: str, original_path: str, thumbnail_path: str | None, clip_embedding: np.ndarray):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO images (original_filename, original_path, thumbnail_path, clip_embedding, is_enhanced, user_tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (original_filename, original_path, thumbnail_path, clip_embedding, False, json.dumps([]))) 
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

def get_image_paths_and_faiss_id(image_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_path, thumbnail_path, faiss_id FROM images WHERE id = ?", (image_id,))
    data = cursor.fetchone()
    conn.close()
    if data:
        return data['original_path'], data['thumbnail_path'], data['faiss_id']
    return None, None, None

def hard_delete_image_from_db(image_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
        logging.info(f"已从数据库中硬删除图片 ID: {image_id}。")
        return True
    except Exception as e:
        logging.error(f"从数据库硬删除图片 ID: {image_id} 失败: {e}")
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
            WHERE id = ? AND deleted = FALSE
        ''', (description, keywords_json, datetime.now(), image_id))
        conn.commit()
        logging.info(f"图片 ID: {image_id} 的增强信息已更新。")
        return True
    except Exception as e:
        logging.error(f"更新图片 ID: {image_id} 增强信息失败: {e}")
        return False
    finally:
        conn.close()

def update_user_tags_for_image(image_id: int, user_tags: list[str]):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        tags_json = json.dumps(user_tags, ensure_ascii=False)
        cursor.execute("UPDATE images SET user_tags = ? WHERE id = ? AND deleted = FALSE", (tags_json, image_id))
        conn.commit()
        logging.info(f"图片 ID: {image_id} 的用户标签已更新为: {tags_json}")
        return True
    except Exception as e:
        logging.error(f"更新图片 ID: {image_id} 用户标签失败: {e}")
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
        SELECT id, original_filename, original_path, thumbnail_path, qwen_description, is_enhanced, user_tags
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
    cursor.execute("""
        SELECT id, original_path, clip_embedding 
        FROM images 
        WHERE is_enhanced = FALSE AND deleted = FALSE 
        ORDER BY upload_timestamp ASC 
        LIMIT ?
    """, (limit,))
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

# --- NEW: Function to get all valid CLIP embeddings for image-to-image search ---
def get_all_valid_images_clip_embeddings():
    """
    Fetches ID, FAISS ID, and CLIP embedding for all non-deleted images 
    that have a CLIP embedding.
    Used for image-to-image search.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Ensure clip_embedding is not NULL and image is not deleted
        cursor.execute("""
            SELECT id, faiss_id, clip_embedding 
            FROM images 
            WHERE deleted = FALSE AND clip_embedding IS NOT NULL
        """)
        images_embeddings_data = cursor.fetchall()
        
        # The 'clip_embedding' column is already converted to np.ndarray by the type converter
        # if it was registered with detect_types=sqlite3.PARSE_DECLTYPES.
        # So, we can directly use it.
        results = [{'id': row['id'], 
                    'faiss_id': row['faiss_id'], 
                    'clip_embedding': row['clip_embedding']} 
                   for row in images_embeddings_data if row['clip_embedding'] is not None]
        
        logging.info(f"Fetched {len(results)} CLIP embeddings from DB for image-to-image search.")
        return results
    except Exception as e:
        logging.error(f"Error fetching all CLIP embeddings: {e}", exc_info=True)
        return []
    finally:
        conn.close()


if __name__ == '__main__':
    init_db()
    print("数据库已初始化。")
    # Example: Test the new function
    # all_embeddings = get_all_valid_images_clip_embeddings()
    # if all_embeddings:
    #     print(f"Found {len(all_embeddings)} images with CLIP embeddings.")
    #     print(f"First item: ID {all_embeddings[0]['id']}, Embedding shape: {all_embeddings[0]['clip_embedding'].shape}")
    # else:
    #     print("No images with CLIP embeddings found or an error occurred.")