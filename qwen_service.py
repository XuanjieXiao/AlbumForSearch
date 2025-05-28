# qwen_service.py
from openai import OpenAI
import base64
import os
import logging
import json
import re # 引入正则表达式库
from PIL import Image # 用于图片处理
import io # 用于内存中的字节流操作
# import time # Potentially for delays between retries, uncomment if used

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

    if img.mode not in ('RGB', 'L'):
        try:
            img = img.convert('RGB')
            logging.info(f"图片 '{image_path}' 已转换为RGB模式。")
        except Exception as e:
            logging.error(f"图片 '{image_path}' 转换为RGB模式失败: {e}")
            return None

    current_quality = 90
    max_resize_attempts = 5
    min_dimension = 100
    original_dims = img.size

    for attempt in range(max_resize_attempts):
        buffer = io.BytesIO()
        try:
            img.save(buffer, format="JPEG", quality=current_quality)
            logging.debug(f"尝试保存图片到内存: '{image_path}', 尺寸: {img.size}, 质量: {current_quality}")
        except Exception as e:
            logging.error(f"保存图片到内存缓冲区失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
            return None

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
            new_width = int(img.width * 0.8)
            new_height = int(img.height * 0.8)

            if new_width < min_dimension or new_height < min_dimension:
                logging.warning(f"图片 '{image_path}' 尺寸已过小 ({new_width}x{new_height})，停止缩减。")
                break
            try:
                img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                logging.debug(f"图片已缩放至: {img.size}")
            except Exception as e:
                logging.error(f"图片缩放失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
                break
            if current_quality > 65:
                current_quality -= 5
        else:
            logging.warning(f"图片 '{image_path}' 达到最大缩减尝试次数，但仍超过大小限制。")

    logging.error(f"图片 '{image_path}' (原始尺寸: {original_dims}) 无法在 {max_resize_attempts} 次尝试内缩减到符合API大小限制。")
    return None


def analyze_image_content(image_path: str):
    """
    分析单个图片内容，返回结构化的描述和关键词。
    包含多种解析回退机制、图片大小处理以及API调用重试逻辑。
    """
    if not client:
        logging.error("Qwen-VL client 未初始化。无法分析图片。")
        return {"description": "", "keywords": []}

    logging.info(f"准备使用Qwen-VL分析图片: {image_path}")
    data_url = _prepare_image_data_for_qwen(image_path)

    if not data_url:
        logging.error(f"为Qwen API准备图片数据失败: {image_path}. 跳过分析。")
        return {"description": "", "keywords": []}

    prompt_text = """请你用中文详细描述这张图片的内容和物品，要求内容简洁明了，突出图片的主要元素和场景，从图片内容中提取出最多10个最具代表性的中文关键词，关键词之间请用单个英文逗号“,”隔开。
最后，请按照如下JSON格式输出，不要包含任何JSON格式之外的额外解释或文字，总字数不要超过490字：
{
  "description": "在此详细描述图片内容，突出主要元素、场景、动作、氛围等信息。",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}"""

    MAX_API_ATTEMPTS = 3  # Total attempts: 1 initial + 2 retries
    last_successful_api_content_if_unparsed = None # Stores content of the last successful API call if it was unparsable
    last_api_call_exception_details = None # Stores details of the last API call exception

    for attempt_num in range(MAX_API_ATTEMPTS):
        current_attempt_str = f"Attempt {attempt_num + 1}/{MAX_API_ATTEMPTS}"
        logging.info(f"开始图片分析 - {current_attempt_str} for image: {image_path}")

        try:
            logging.info(f"向Qwen-VL API发送图片分析请求: {image_path} ({current_attempt_str}, 处理后图片数据长度: {len(data_url)})")
            response = client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct",
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
            )
            # Successfully got a response from API
            result_content = response.choices[0].message.content.strip()
            # This content is from a successful API call. Store it as a fallback candidate.
            last_successful_api_content_if_unparsed = result_content
            last_api_call_exception_details = None # Clear any previous API error since this call was successful

            logging.info(f"图片分析API调用成功: {image_path} ({current_attempt_str})")
            logging.debug(f"Qwen-VL原始输出 ({current_attempt_str}, 处理前): '{result_content}'")

            # --- Begin parsing logic for result_content ---
            try:
                # Attempt 1: Direct JSON parsing (after stripping potential markdown)
                content_to_parse = result_content
                if content_to_parse.startswith("```json"):
                    content_to_parse = content_to_parse[7:]
                    if content_to_parse.endswith("```"):
                        content_to_parse = content_to_parse[:-3]
                    content_to_parse = content_to_parse.strip()

                parsed_result = json.loads(content_to_parse)
                description = parsed_result.get("description", "")
                keywords = parsed_result.get("keywords", [])
                if isinstance(keywords, str): # Handle if keywords are a single comma-separated string
                    keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                logging.info(f"通过直接JSON解析成功 ({current_attempt_str}).")
                return {"description": description, "keywords": keywords[:10]} # Successfully parsed
            except json.JSONDecodeError as je:
                logging.warning(f"直接JSON解析失败 ({current_attempt_str}): {je}. 返回内容: '{result_content}'. 尝试其他解析方法...")
                # Attempt 2: Regex to find JSON object within the text
                try:
                    match = re.search(r'\{.*\}', result_content, re.DOTALL)
                    if match:
                        potential_json_str = match.group(0)
                        logging.info(f"提取到潜在JSON子字符串 ({current_attempt_str}): '{potential_json_str}'")
                        parsed_result = json.loads(potential_json_str)
                        description = parsed_result.get("description", "")
                        keywords = parsed_result.get("keywords", [])
                        if isinstance(keywords, str):
                            keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                        logging.info(f"通过提取子字符串JSON解析成功 ({current_attempt_str}).")
                        return {"description": description, "keywords": keywords[:10]} # Successfully parsed
                    else:
                        logging.warning(f"未在返回内容中找到JSON对象模式 ( {{.*}} ) ({current_attempt_str}).")
                except json.JSONDecodeError as sub_je:
                    logging.warning(f"从子字符串解析JSON也失败 ({current_attempt_str}): {sub_je}.")
                except Exception as e_sub_extract: # Catch any other error during regex extraction
                    logging.error(f"提取子字符串JSON时发生其他错误 ({current_attempt_str}): {e_sub_extract}")

                # Attempt 3: Line-based parsing for "description... keywords:" format
                logging.info(f"尝试解析 '描述...换行...关键词：' 格式 ({current_attempt_str}).")
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
                            if '、' in kw_data: # Chinese comma
                                extracted_keywords = [kw.strip() for kw in kw_data.split('、') if kw.strip()]
                            elif ',' in kw_data: # English comma
                                extracted_keywords = [kw.strip() for kw in kw_data.split(',') if kw.strip()]
                            else: # Space separated as a fallback
                                extracted_keywords = [kw.strip() for kw in kw_data.split(' ') if kw.strip()]
                            keywords_line_found = True
                            is_keyword_line = True
                            break
                    if is_keyword_line: continue
                    if not keywords_line_found: # Only append to description if keywords line not yet found
                        extracted_description_parts.append(stripped_line)

                final_description = " ".join(extracted_description_parts).strip()
                if final_description or extracted_keywords: # If any data was extracted
                    logging.info(f"通过解析'描述...关键词：'格式获得 ({current_attempt_str}): desc='{final_description[:50]}...', keywords={extracted_keywords}")
                    return {"description": final_description, "keywords": extracted_keywords[:10]} # Successfully parsed
                else:
                    # ALL PARSING METHODS FAILED for the current result_content
                    logging.error(f"所有解析尝试均失败 ({current_attempt_str}). 原始返回内容 (已strip): '{result_content}'")
                    if attempt_num < MAX_API_ATTEMPTS - 1:
                        logging.info(f"由于解析失败，将进行下一次尝试 (下一个是 {attempt_num + 2}/{MAX_API_ATTEMPTS}).")
                        # time.sleep(1) # Optional: introduce delay before retrying
                    # If this was the last attempt (attempt_num == MAX_API_ATTEMPTS - 1),
                    # 'last_successful_api_content_if_unparsed' holds 'result_content'.
                    # The loop will end, and the fallback logic after the loop will use this value.
            # --- End parsing logic ---
        except Exception as e: # Catches API call errors (from client.chat.completions.create)
            logging.error(f"Qwen-VL分析图片 '{image_path}' 时API调用失败 ({current_attempt_str}): {e}", exc_info=True)
            last_api_call_exception_details = str(e) # Store current API error
            # Since API call failed, there's no new "successful_api_content" from THIS attempt.
            # last_successful_api_content_if_unparsed will retain its value from a previous successful call, if any.

            if attempt_num < MAX_API_ATTEMPTS - 1:
                logging.info(f"由于API调用错误，将进行下一次尝试 (下一个是 {attempt_num + 2}/{MAX_API_ATTEMPTS}).")
                # time.sleep(1) # Optional: introduce delay before retrying
            # If this was the last attempt and API call failed, loop ends.
            # Fallback logic will use 'last_api_call_exception_details' if 'last_successful_api_content_if_unparsed' is None.

    # --- After the loop ---
    # If this point is reached, it means all MAX_API_ATTEMPTS were made,
    # and none of them resulted in a successful parse and return within the loop.
    logging.error(f"所有 {MAX_API_ATTEMPTS} 次分析尝试均已完成，但未能成功解析结果。")

    if last_successful_api_content_if_unparsed:
        # This implies at least one API call was successful and provided content,
        # but that content (from the latest such call) couldn't be parsed.
        # This is the primary fallback for "unparsable response".
        logging.info(f"将最后一次成功API调用但无法解析的响应作为描述返回。内容: '{last_successful_api_content_if_unparsed}'")
        return {"description": last_successful_api_content_if_unparsed, "keywords": []}
    elif last_api_call_exception_details:
        # This means all attempts that made it to API call stage resulted in API call failures,
        # and no attempt ever yielded successful (even if unparsable) content.
        logging.error(f"所有 {MAX_API_ATTEMPTS} 次API调用均失败。最后记录的API错误: {last_api_call_exception_details}")
        return {"description": f"API调用在所有{MAX_API_ATTEMPTS}次尝试后均失败", "keywords": []}
    else:
        # This is a less likely fallback, e.g., if MAX_API_ATTEMPTS was 0 (not possible here with fixed value),
        # or some other unexpected state where neither content nor an API error was recorded.
        logging.error("所有尝试后，既无成功获取但无法解析的内容，也无API调用错误记录。返回通用失败信息。")
        return {"description": "无法解析描述，且API未能在多次尝试中提供可解析内容或均调用失败", "keywords": []}


if __name__ == "__main__":
    # --- 测试部分保持用户原样，但确保测试图片路径有效 ---
    test_image_dir = "test_uploads_qwen" # 建议为测试图片创建一个单独目录
    os.makedirs(test_image_dir, exist_ok=True)

    test_image_file = os.path.join(test_image_dir, "example_large_image.jpg")

    if not os.path.exists(test_image_file):
        try:
            img_large = Image.new('RGB', (4000, 3000), color = 'skyblue')
            img_large.save(test_image_file, "JPEG", quality=95)
            logging.info(f"创建了大型虚拟测试图片: {test_image_file} (大小: {os.path.getsize(test_image_file)/(1024*1024):.2f} MB)")
        except Exception as e_create_img:
            logging.error(f"创建大型虚拟测试图片失败: {e_create_img}")

    if os.path.exists(test_image_file):
        if client:
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

    # This mock_create will be replaced temporarily for each test text
    # The retry logic in analyze_image_content will mean this mock might be called multiple times
    # if a test text is designed to fail parsing.
    def mock_create_factory(text_to_return):
        call_count = 0
        def _mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            logging.info(f"Mock API call number {call_count} for this test, returning: '{text_to_return}'")
            # To test retry, you might want the mock to return different things on successive calls,
            # or simulate API errors. For this simple parsing test, it always returns the same text.
            return MockResponse(text_to_return)
        return _mock

    test_texts_for_parsing = {
        "test_text_1_parses_line_based": """这是一个美丽的湖泊，湖边有绿树环绕。\n关键词：湖泊、绿树、风景、自然""",
        "test_text_2_parses_json_markdown": """```json
{
  "description": "城市夜景，灯火辉煌。",
  "keywords": ["城市", "夜景", "灯光"]
}
```""",
        "test_text_3_fails_all_parsing": """完全没有按格式要求。就是一段话。""", # This will trigger retries with the mock
        "test_text_4_parses_json_embedded": """{"description":"JSON对象在文本中间","keywords":["嵌入式","测试"]} 其他无关文字。"""
    }

    for test_name, test_text in test_texts_for_parsing.items():
        print(f"\n测试文本 {test_name}:\n{test_text}")
        if client:
            client.chat.completions.create = mock_create_factory(test_text)
            # For a dummy path, _prepare_image_data_for_qwen would normally fail.
            # To properly test analyze_image_content's parsing part with a mock,
            # we might need to also mock _prepare_image_data_for_qwen or use a tiny valid dummy image.
            # However, if we assume data_url is obtained, analyze_image_content will proceed.
            # For simplicity, we'll use a dummy path and expect _prepare_image_data_for_qwen to handle it.
            # A better mock test would isolate the parsing logic more directly.
            # Let's assume _prepare_image_data_for_qwen is robust or we use a tiny placeholder.
            # To ensure the retry logic is tested for "test_text_3_fails_all_parsing":
            # The mock will be called 3 times. The final result should be the raw test_text_3.

            # Create a dummy file for _prepare_image_data_for_qwen to "succeed" without erroring out early
            dummy_image_path = os.path.join(test_image_dir, f"dummy_for_{test_name}.jpg")
            if not os.path.exists(dummy_image_path):
                try:
                    Image.new('RGB', (10,10), color='red').save(dummy_image_path, "JPEG")
                except Exception: pass # ignore if cannot create

            analysis = analyze_image_content(dummy_image_path) # Use dummy path
            print(f"解析结果 {test_name}:", json.dumps(analysis, ensure_ascii=False, indent=4))
        else:
            print(f"Qwen-VL client 未初始化，跳过对 {test_name} 的模拟解析测试。")

    if client and original_create_method:
        client.chat.completions.create = original_create_method
        logging.info("已恢复原始的 client.chat.completions.create 方法。")