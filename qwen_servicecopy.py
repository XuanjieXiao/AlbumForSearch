# qwen_service.py
from openai import OpenAI
import base64
import os
import logging
import json
import re # 引入正则表达式库
from PIL import Image # 用于图片处理
import io # 用于内存中的字节流操作

# 配置日志格式和等级
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', # 添加了module
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("analyze_images.log", encoding='utf-8')
    ]
)

# 配置 Qwen API (保持用户原有的配置)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "HoUbVVd_L1Z0uLJJiq5ND13yfDreU4pkTHwoTbU_EMp31G_OLx_ONh5fIoa37cNM4mRfAvst7bR_9VUfi4-QXg")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://www.sophnet.com/api/open-apis/v1")

client = None
if QWEN_API_KEY and QWEN_API_KEY != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" and QWEN_API_KEY != "YOUR_QWEN_API_KEY": # 确保不是占位符
    try:
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL
        )
        logging.info("Qwen-VL 服务已使用API Key初始化。")
    except Exception as e:
        logging.error(f"初始化 Qwen-VL OpenAI 客户端失败: {e}")
else:
    logging.warning("QWEN_API_KEY 未配置或为默认占位符。Qwen-VL 服务可能无法正常工作。")

# Base64编码后图片字符串的最大允许字符数 (约7MB，为8MB API限制留出余量)
MAX_BASE64_IMAGE_CHARS = 7 * 1024 * 1024

def _prepare_image_data_for_qwen(image_path: str, max_chars: int = MAX_BASE64_IMAGE_CHARS) -> str | None:
    """
    打开图片，尝试将其编码为Base64 (JPEG格式)。
    如果Base64字符串过大，则迭代调整图片大小和质量，直到符合限制。
    成功则返回 "data:image/jpeg;base64,..." 格式的字符串，否则返回 None。
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        logging.error(f"图片文件未找到: {image_path}")
        return None
    except Exception as e:
        logging.error(f"使用Pillow打开图片失败 '{image_path}': {e}")
        return None

    # 确保图片是RGB模式，这是JPEG编码的常用模式
    if img.mode not in ('RGB', 'L'): # L代表灰度图，也可以保存为JPEG
        try:
            img = img.convert('RGB')
            logging.info(f"图片 '{image_path}' 已转换为RGB模式。")
        except Exception as e:
            logging.error(f"图片 '{image_path}' 转换为RGB模式失败: {e}")
            return None

    # 初始JPEG保存质量
    current_quality = 90
    # 最大尝试缩放次数
    max_resize_attempts = 5
    # 最小图片边长，防止过度缩放
    min_dimension = 100

    original_dims = img.size

    for attempt in range(max_resize_attempts):
        buffer = io.BytesIO()
        try:
            # 将图片以JPEG格式保存到内存缓冲区
            img.save(buffer, format="JPEG", quality=current_quality)
            logging.debug(f"尝试保存图片到内存: '{image_path}', 尺寸: {img.size}, 质量: {current_quality}")
        except Exception as e:
            logging.error(f"保存图片到内存缓冲区失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
            return None # 如果保存操作失败，则停止

        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        current_base64_size_mb = len(base64_str) / (1024 * 1024)

        if len(base64_str) <= max_chars:
            logging.info(f"图片 '{image_path}' (原始尺寸: {original_dims}, 当前尺寸: {img.size}, 质量: {current_quality}, 尝试 {attempt + 1}) "
                         f"成功编码为Base64，大小: {current_base64_size_mb:.2f} MB。")
            return f"data:image/jpeg;base64,{base64_str}"

        logging.info(f"图片 '{image_path}' (原始尺寸: {original_dims}, 当前尺寸: {img.size}, 质量: {current_quality}, 尝试 {attempt + 1}) "
                     f"Base64大小 ({current_base64_size_mb:.2f} MB) 超过限制 ({max_chars / (1024*1024):.2f} MB). 尝试缩减...")

        if attempt < max_resize_attempts - 1:
            # 计算新的目标尺寸 (按比例缩小20%)
            new_width = int(img.width * 0.8)
            new_height = int(img.height * 0.8)

            # 如果缩放后尺寸过小，则停止尝试
            if new_width < min_dimension or new_height < min_dimension:
                logging.warning(f"图片 '{image_path}' 尺寸已过小 ({new_width}x{new_height})，停止缩减。")
                break

            try:
                # 使用 thumbnail 方法进行等比缩放，它会直接修改img对象
                img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                logging.debug(f"图片已缩放至: {img.size}")
            except Exception as e:
                logging.error(f"图片缩放失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
                break # 如果缩放失败，则停止

            # 如果图片仍然过大，可以考虑在后续尝试中略微降低JPEG质量
            if current_quality > 65: # JPEG质量不要低于65
                current_quality -= 5
        else:
            logging.warning(f"图片 '{image_path}' 达到最大缩减尝试次数，但仍超过大小限制。")

    logging.error(f"图片 '{image_path}' (原始尺寸: {original_dims}) 无法在 {max_resize_attempts} 次尝试内缩减到符合API大小限制。")
    return None


def analyze_image_content(image_path: str):
    """
    分析单个图片内容，返回结构化的描述和关键词。
    包含多种解析回退机制和图片大小处理。
    """
    if not client:
        logging.error("Qwen-VL client 未初始化。无法分析图片。")
        return {"description": "", "keywords": []}

    logging.info(f"准备使用Qwen-VL分析图片: {image_path}")

    # 准备图片数据，如果过大则进行缩放
    data_url = _prepare_image_data_for_qwen(image_path)

    if not data_url:
        logging.error(f"为Qwen API准备图片数据失败: {image_path}. 跳过分析。")
        return {"description": "", "keywords": []} # 返回空结果，避免后续出错

    prompt_text = """请你用中文详细描述这张图片的内容和物品，要求内容简洁明了，突出图片的主要元素和场景，从图片内容中提取出最多10个最具代表性的中文关键词，关键词之间请用单个英文逗号“,”隔开。
最后，请按照如下JSON格式输出，不要包含任何JSON格式之外的额外解释或文字，总字数不要超过490字：
{
  "description": "在此详细描述图片内容，突出主要元素、场景、动作、氛围等信息。",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}"""

    try:
        logging.info(f"向Qwen-VL API发送图片分析请求: {image_path} (处理后图片数据长度: {len(data_url)})")
        response = client.chat.completions.create(
            model="Qwen2.5-VL-7B-Instruct", # 保持用户原有的模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            temperature=0.7,
            # max_tokens=600 # 用户原有代码中注释掉了，保持一致
        )
        result_content = response.choices[0].message.content.strip()
        logging.info(f"图片分析成功: {image_path}")
        logging.debug(f"Qwen-VL原始输出 (处理前): '{result_content}'")

        # ---后续的JSON解析逻辑保持用户原样---
        try:
            if result_content.startswith("```json"):
                result_content = result_content[7:]
                if result_content.endswith("```"):
                    result_content = result_content[:-3]
                result_content = result_content.strip()

            parsed_result = json.loads(result_content)
            description = parsed_result.get("description", "")
            keywords = parsed_result.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            logging.info("通过直接JSON解析成功。")
            return {"description": description, "keywords": keywords[:10]}
        except json.JSONDecodeError as je:
            logging.warning(f"直接JSON解析失败: {je}. 返回内容: '{result_content}'. 尝试其他解析方法...")
            try:
                match = re.search(r'\{.*\}', result_content, re.DOTALL)
                if match:
                    potential_json_str = match.group(0)
                    logging.info(f"提取到潜在JSON子字符串: '{potential_json_str}'")
                    parsed_result = json.loads(potential_json_str)
                    description = parsed_result.get("description", "")
                    keywords = parsed_result.get("keywords", [])
                    if isinstance(keywords, str):
                        keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                    logging.info("通过提取子字符串JSON解析成功。")
                    return {"description": description, "keywords": keywords[:10]}
                else:
                    logging.warning("未在返回内容中找到JSON对象模式 ( {.*} )。")
            except json.JSONDecodeError as sub_je:
                logging.warning(f"从子字符串解析JSON也失败: {sub_je}.")
            except Exception as e_sub_extract:
                logging.error(f"提取子字符串JSON时发生其他错误: {e_sub_extract}")

            logging.info("尝试解析 '描述...换行...关键词：' 格式。")
            lines = result_content.splitlines()
            extracted_description_parts = []
            extracted_keywords = []
            keywords_line_found = False
            keyword_prefixes = ["关键词：", "关键词:", "keywords:", "Keywords:"]

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line: continue
                is_keyword_line = False
                for prefix in keyword_prefixes:
                    if stripped_line.lower().startswith(prefix.lower()):
                        kw_data = stripped_line[len(prefix):].strip()
                        if '、' in kw_data:
                            extracted_keywords = [kw.strip() for kw in kw_data.split('、') if kw.strip()]
                        elif ',' in kw_data:
                            extracted_keywords = [kw.strip() for kw in kw_data.split(',') if kw.strip()]
                        else:
                            extracted_keywords = [kw.strip() for kw in kw_data.split(' ') if kw.strip()]
                        keywords_line_found = True
                        is_keyword_line = True
                        break
                if is_keyword_line: continue
                if not keywords_line_found:
                    extracted_description_parts.append(stripped_line)

            final_description = " ".join(extracted_description_parts).strip()
            if final_description or extracted_keywords:
                logging.info(f"通过解析'描述...关键词：'格式获得: desc='{final_description[:50]}...', keywords={extracted_keywords}")
                return {"description": final_description, "keywords": extracted_keywords[:10]}
            else:
                logging.error(f"所有解析尝试均失败。原始返回内容 (已strip): '{result_content}'")
                # Fallback: 如果解析完全失败，但有内容，则将原始内容作为描述返回
                if result_content:
                    logging.info("将整个无法解析的响应作为描述返回。")
                    return {"description": result_content, "keywords": []}
                return {"description": "无法解析描述", "keywords": []}

    except Exception as e:
        logging.error(f"Qwen-VL分析图片 '{image_path}' 失败 (API调用或未知错误): {e}", exc_info=True)
        return {"description": "", "keywords": []}


if __name__ == "__main__":
    # --- 测试部分保持用户原样，但确保测试图片路径有效 ---
    # 测试单个图片分析
    test_image_dir = "test_uploads_qwen" # 建议为测试图片创建一个单独目录
    os.makedirs(test_image_dir, exist_ok=True)

    # 使用一个相对路径，并确保这个图片在执行脚本时存在于该路径
    # 或者提供一个绝对路径
    test_image_file = os.path.join(test_image_dir, "example_large_image.jpg") # 修改为您实际的测试图片路径

    if not os.path.exists(test_image_file):
        try:
            # 创建一个较大的虚拟测试图片 (例如，模拟一个可能需要缩放的图片)
            # 注意：这个虚拟图片可能不会真正达到需要多次缩放的程度，
            # 真实测试最好使用一个实际的、Base64编码后会超过7MB的大图片文件。
            img_large = Image.new('RGB', (4000, 3000), color = 'skyblue')
            # 可以添加一些噪点或复杂图案让JPEG文件稍大一些
            # from PIL import ImageDraw
            # draw = ImageDraw.Draw(img_large)
            # for i in range(0, img_large.width, 50):
            #     draw.line([(i,0), (img_large.width-i, img_large.height)], fill="red", width=5)
            # for i in range(0, img_large.height, 50):
            #     draw.line([(0,i), (img_large.width, img_large.height-i)], fill="blue", width=5)
            img_large.save(test_image_file, "JPEG", quality=95) # 以较高质量保存
            logging.info(f"创建了大型虚拟测试图片: {test_image_file} (大小: {os.path.getsize(test_image_file)/(1024*1024):.2f} MB)")
        except Exception as e_create_img:
            logging.error(f"创建大型虚拟测试图片失败: {e_create_img}")

    if os.path.exists(test_image_file):
        if client: # 只有在client初始化成功时才进行API调用测试
            analysis_result = analyze_image_content(test_image_file)
            print("\n单张图片分析结果:")
            print(json.dumps(analysis_result, ensure_ascii=False, indent=4))
        else:
            print(f"Qwen-VL client 未初始化，跳过对 {test_image_file} 的 API 分析测试。")
            print("请检查QWEN_API_KEY和QWEN_BASE_URL配置。")
    else:
        logging.warning(f"测试图片 {test_image_file} 不存在。请创建或修改路径以进行测试。")

    # --- 用户原有的模拟解析测试部分保持不变 ---
    print("\n--- 测试非JSON格式解析 ---")
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    class MockMessage:
        def __init__(self, content):
            self.content = content
    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    original_create_method = None
    if client and hasattr(client.chat.completions, 'create'):
        original_create_method = client.chat.completions.create

    def mock_create(*args, **kwargs):
        return MockResponse(kwargs.get('mock_return_text', ''))

    if client:
        client.chat.completions.create = mock_create
        # 定义一个虚拟的kwargs，因为在实际调用analyze_image_content时它不会被传递
        # 这个kwargs仅用于模拟测试，实际API调用不需要它
        mock_kwargs_for_test = {}


    test_texts_for_parsing = {
        "test_text_1": """这是一个美丽的湖泊，湖边有绿树环绕。\n关键词：湖泊、绿树、风景、自然""",
        "test_text_2": """```json
{
  "description": "城市夜景，灯火辉煌。",
  "keywords": ["城市", "夜景", "灯光"]
}
```""",
        "test_text_3": """完全没有按格式要求。就是一段话。""",
        "test_text_4": """{"description":"JSON对象在文本中间","keywords":["嵌入式","测试"]} 其他无关文字。"""
    }

    for test_name, test_text in test_texts_for_parsing.items():
        print(f"\n测试文本 {test_name}:\n{test_text}")
        if client:
            # 在mock_create中，我们从kwargs获取mock_return_text
            # 因此，在调用analyze_image_content前，需要确保mock_create能拿到这个值
            # 最简单的方式是直接修改mock_create的行为或通过一个全局变量传递，
            # 但为了保持analyze_image_content接口不变，这里我们假设mock_create能访问到它
            # 对于更复杂的mocking，可以使用unittest.mock
            # 这里我们简单地在调用前设置一个属性给mock_create函数对象（如果Python允许）
            # 或者，更简单地，让mock_create直接使用 test_text
            
            # 临时的解决方案：直接修改mock_create以使用外部变量
            # 这不是最佳实践，但在这个简单场景下可以工作
            def temp_mock_create_specific(text_to_return):
                def _mock(*args, **kwargs):
                    return MockResponse(text_to_return)
                return _mock

            client.chat.completions.create = temp_mock_create_specific(test_text)
            analysis = analyze_image_content(f"dummy_path_for_{test_name}.jpg")
            print(f"解析结果 {test_name}:", json.dumps(analysis, ensure_ascii=False, indent=4))
        else:
            print(f"Qwen-VL client 未初始化，跳过对 {test_name} 的模拟解析测试。")


    if client and original_create_method:
        client.chat.completions.create = original_create_method
        logging.info("已恢复原始的 client.chat.completions.create 方法。")
