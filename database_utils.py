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
    # Check if 'user_tags' column exists, add if not (for backward compatibility)
    cursor.execute("PRAGMA table_info(images)")
    columns = [column['name'] for column in cursor.fetchall()]
    if 'user_tags' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN user_tags TEXT")
        logging.info("Added 'user_tags' column to 'images' table.")
    if 'deleted' not in columns: # Should exist, but good practice
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
            user_tags TEXT, -- Stores JSON array of strings, e.g., '["tag1", "tag2"]'
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_enhanced BOOLEAN DEFAULT FALSE,
            last_enhanced_timestamp TIMESTAMP,
            deleted BOOLEAN DEFAULT FALSE 
        )
    ''')
    # Ensure indexes exist
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
        ''', (original_filename, original_path, thumbnail_path, clip_embedding, False, json.dumps([]))) # Initialize user_tags as empty JSON list
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
    """ Helper to get paths and faiss_id for deletion """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_path, thumbnail_path, faiss_id FROM images WHERE id = ?", (image_id,))
    data = cursor.fetchone()
    conn.close()
    if data:
        return data['original_path'], data['thumbnail_path'], data['faiss_id']
    return None, None, None

def hard_delete_image_from_db(image_id: int):
    """ Performs a hard delete from the database. """
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
    """ Updates user_tags for a single image. Tags are stored as a JSON string list. """
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
        # Assuming convert_array is correctly registered and handles the conversion
        return result['clip_embedding'] 
    return None


if __name__ == '__main__':
    init_db()
    print("数据库已初始化。")
    # Example: Add a test image and then update its tags
    # test_image_id = add_image_to_db("test.jpg", "/path/to/test.jpg", "/path/to/thumb_test.jpg", np.random.rand(768).astype(np.float32))
    # if test_image_id:
    #     print(f"Added test image with ID: {test_image_id}")
    #     update_user_tags_for_image(test_image_id, ["风景", "测试"])
    #     img_data = get_image_by_id(test_image_id)
    #     if img_data:
    #         print(f"Image data after tagging: {dict(img_data)}")
    #         user_tags_loaded = json.loads(img_data['user_tags']) if img_data['user_tags'] else []
    #         print(f"Loaded user tags: {user_tags_loaded}")
