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
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "HoUbVVd_L1Z0uLJJiq5ND13yfDreU4pkTHwoTbU_EMp31G_OLx_ONh5fIoa37cNM4mRfAvst7bR_9VUfi4-QXg") # 请替换为您的真实API Key或确保环境变量已设置
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://www.sophnet.com/api/open-apis/v1") # 示例URL，请确认

client = None
if QWEN_API_KEY and QWEN_API_KEY not in ["sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "YOUR_QWEN_API_KEY", ""]:
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
                # Changed from thumbnail to resize to ensure dimensions are explicitly set
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logging.debug(f"图片已缩放至: {img.size}")
            except Exception as e:
                logging.error(f"图片缩放失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
                break
            if current_quality > 65: # Keep quality reduction logic
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

    # Refined prompt with explicit instructions for escaping double quotes
    prompt_text = """请你用中文详细描述这张图片的内容和物品，要求内容简洁明了，突出图片的所有元素和场景。从图片内容中提取出最多10个最具代表性的中文关键词，关键词之间请用单个英文逗号“,”隔开。

最后，请严格按照如下JSON格式输出。JSON字符串内部的文本（尤其是"description"字段的值）如果包含英文双引号(")，必须将其转义为 \\"。不要包含任何JSON格式之外的额外解释或文字。总字数不要超过490字：
{
  "description": "在此详细描述图片内容，包含所有元素、场景、动作、氛围等信息。",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}"""

    MAX_API_ATTEMPTS = 3
    last_successful_api_content_if_unparsed = None
    last_api_call_exception_details = None

    for attempt_num in range(MAX_API_ATTEMPTS):
        current_attempt_str = f"Attempt {attempt_num + 1}/{MAX_API_ATTEMPTS}"
        logging.info(f"开始图片分析 - {current_attempt_str} for image: {image_path}")

        try:
            logging.info(f"向Qwen-VL API发送图片分析请求: {image_path} ({current_attempt_str}, 处理后图片数据长度: {len(data_url)})")
            response = client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct", # Or your preferred model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                temperature=0.7, # Or your preferred temperature
            )
            result_content = response.choices[0].message.content.strip()
            last_successful_api_content_if_unparsed = result_content
            last_api_call_exception_details = None

            logging.info(f"图片分析API调用成功: {image_path} ({current_attempt_str})")
            logging.debug(f"Qwen-VL原始输出 ({current_attempt_str}, 处理前): '{result_content}'")

            try:
                # Attempt 1: Direct JSON parsing (after stripping potential markdown)
                content_to_parse = result_content
                if content_to_parse.startswith("```json"):
                    content_to_parse = content_to_parse[len("```json"):]
                    if content_to_parse.endswith("```"):
                        content_to_parse = content_to_parse[:-len("```")]
                    content_to_parse = content_to_parse.strip()
                
                parsed_result = json.loads(content_to_parse)
                description = parsed_result.get("description", "")
                keywords = parsed_result.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                logging.info(f"通过直接JSON解析成功 ({current_attempt_str}).")
                return {"description": description, "keywords": keywords[:10]}
            except json.JSONDecodeError as je:
                logging.warning(f"直接JSON解析失败 ({current_attempt_str}): {je}. 返回内容: '{result_content}'. 尝试其他解析方法...")
                # Attempt 2: Regex to find JSON object within the text, then parse
                try:
                    # Using re.DOTALL to make . match newlines
                    match = re.search(r'\{[\s\S]*\}', result_content) 
                    if match:
                        potential_json_str = match.group(0)
                        logging.info(f"提取到潜在JSON子字符串 ({current_attempt_str}): '{potential_json_str}'")
                        
                        # Clean markdown from the extracted string as well
                        if potential_json_str.startswith("```json"):
                            potential_json_str = potential_json_str[len("```json"):]
                            if potential_json_str.endswith("```"):
                                potential_json_str = potential_json_str[:-len("```")]
                            potential_json_str = potential_json_str.strip()
                        
                        parsed_result = json.loads(potential_json_str)
                        description = parsed_result.get("description", "")
                        keywords = parsed_result.get("keywords", [])
                        if isinstance(keywords, str):
                            keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                        logging.info(f"通过提取子字符串JSON解析成功 ({current_attempt_str}).")
                        return {"description": description, "keywords": keywords[:10]}
                    else:
                        logging.warning(f"未在返回内容中找到JSON对象模式 ( {{.*}} ) ({current_attempt_str}).")
                except json.JSONDecodeError as sub_je:
                    logging.warning(f"从子字符串解析JSON也失败 ({current_attempt_str}): {sub_je}.")
                except Exception as e_sub_extract:
                    logging.error(f"提取子字符串JSON时发生其他错误 ({current_attempt_str}): {e_sub_extract}")

                # Attempt 3: Line-based parsing for "description... keywords:" format (your existing fallback)
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
                            # Handle various delimiters for keywords
                            if '、' in kw_data: # Chinese comma
                                extracted_keywords = [kw.strip() for kw in kw_data.split('、') if kw.strip()]
                            elif ',' in kw_data: # English comma
                                extracted_keywords = [kw.strip() for kw in kw_data.split(',') if kw.strip()]
                            else: # Space separated as a fallback
                                extracted_keywords = [kw.strip() for kw in kw_data.split(' ') if kw.strip()] # Changed from kw_data.split()
                            keywords_line_found = True
                            is_keyword_line = True
                            break
                    if is_keyword_line: continue
                    if not keywords_line_found:
                        extracted_description_parts.append(stripped_line)
                
                final_description = " ".join(extracted_description_parts).strip()
                if final_description or extracted_keywords:
                    logging.info(f"通过解析'描述...关键词：'格式获得 ({current_attempt_str}): desc='{final_description[:50]}...', keywords={extracted_keywords}")
                    return {"description": final_description, "keywords": extracted_keywords[:10]}
                else:
                    logging.error(f"所有解析尝试均失败 ({current_attempt_str}). 原始返回内容 (已strip): '{result_content}'")
                    if attempt_num < MAX_API_ATTEMPTS - 1:
                        logging.info(f"由于解析失败，将进行下一次API尝试 (下一个是 {attempt_num + 2}/{MAX_API_ATTEMPTS}).")
                        # if you want a delay: time.sleep(1) 
        except Exception as e:
            logging.error(f"Qwen-VL分析图片 '{image_path}' 时API调用失败 ({current_attempt_str}): {e}", exc_info=True)
            last_api_call_exception_details = str(e)
            if attempt_num < MAX_API_ATTEMPTS - 1:
                logging.info(f"由于API调用错误，将进行下一次API尝试 (下一个是 {attempt_num + 2}/{MAX_API_ATTEMPTS}).")
                # if you want a delay: time.sleep(1) 

    # After all attempts
    logging.error(f"所有 {MAX_API_ATTEMPTS} 次分析尝试均已完成，但未能成功解析结果。")
    if last_successful_api_content_if_unparsed:
        logging.info(f"将最后一次成功API调用但无法解析的响应作为描述返回。内容: '{last_successful_api_content_if_unparsed}'")
        return {"description": last_successful_api_content_if_unparsed, "keywords": []}
    elif last_api_call_exception_details:
        logging.error(f"所有 {MAX_API_ATTEMPTS} 次API调用均失败。最后记录的API错误: {last_api_call_exception_details}")
        return {"description": f"API调用在所有{MAX_API_ATTEMPTS}次尝试后均失败: {last_api_call_exception_details}", "keywords": []}
    else:
        logging.error("所有尝试后，既无成功获取但无法解析的内容，也无API调用错误记录。返回通用失败信息。")
        return {"description": "无法解析描述，且API未能在多次尝试中提供可解析内容或均调用失败", "keywords": []}


if __name__ == "__main__":
    test_image_dir = "test_uploads_qwen" 
    os.makedirs(test_image_dir, exist_ok=True)
    test_image_file = os.path.join(test_image_dir, "example_test_image.jpg") # Changed name for clarity

    # Create a dummy image if it doesn't exist for testing _prepare_image_data_for_qwen and full flow
    if not os.path.exists(test_image_file):
        try:
            img_test = Image.new('RGB', (800, 600), color = 'red')
            img_test.save(test_image_file, "JPEG", quality=90)
            logging.info(f"创建了虚拟测试图片: {test_image_file}")
        except Exception as e_create_img:
            logging.error(f"创建虚拟测试图片失败: {e_create_img}")

    if os.path.exists(test_image_file):
        if client:
            print(f"\n--- 测试对真实（或虚拟）图片 {test_image_file} 的完整分析流程 ---")
            analysis_result = analyze_image_content(test_image_file)
            print("\n单张图片分析结果:")
            print(json.dumps(analysis_result, ensure_ascii=False, indent=4))
        else:
            print(f"\nQwen-VL client 未初始化，跳过对 {test_image_file} 的 API 分析测试。")
            print("请检查QWEN_API_KEY和QWEN_BASE_URL配置。如果已配置但仍提示未初始化，请检查API Key值是否为占位符。")
    else:
        logging.warning(f"测试图片 {test_image_file} 不存在且无法创建。请创建或修改路径以进行完整流程测试。")

    # --- Mocking API responses for parsing logic tests ---
    print("\n--- 测试不同格式响应的解析逻辑 (使用Mock API) ---")
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

    def mock_create_factory(text_to_return):
        def _mock(*args, **kwargs):
            logging.info(f"Mock API call, returning: '{text_to_return}'")
            return MockResponse(text_to_return)
        return _mock

    # Create a tiny dummy image file that _prepare_image_data_for_qwen can process without error
    # This allows testing the parsing logic without actual API calls or large image processing.
    dummy_image_for_parsing_test = os.path.join(test_image_dir, "tiny_dummy_for_parser_test.jpg")
    if not os.path.exists(dummy_image_for_parsing_test):
        try:
            Image.new('RGB', (10,10), color='blue').save(dummy_image_for_parsing_test, "JPEG")
        except Exception: 
            logging.error(f"无法创建用于解析测试的微型虚拟图片 {dummy_image_for_parsing_test}")
            # If tiny dummy cannot be created, mock tests below might fail early at _prepare_image_data_for_qwen
            # For robust testing, ensure this file can be created or mock _prepare_image_data_for_qwen too.

    test_texts_for_parsing = {
        "test_valid_json": """{
  "description": "这是一张有效的JSON描述。",
  "keywords": ["有效", "JSON"]
}""",
        "test_json_with_markdown": """```json
{
  "description": "城市夜景，灯火辉煌，包含“引言”和结束。",
  "keywords": ["城市", "夜景", "灯光", "引言"]
}
```""",
        "test_json_with_escaped_quotes_in_prompt_example": """{
  "description": "图片展示了一个牌子，上面写着“特价商品\\"大甩卖”。",
  "keywords": ["牌子", "特价", "甩卖"]
}""",
        "test_line_based_fallback": """这是一个美丽的湖泊，湖边有绿树环绕。\n关键词：湖泊、绿树、风景、自然""",
        "test_malformed_json_should_retry_and_fallback": """{
  "description": "这是一个部分损坏的JSON，引号未闭合,
  "keywords": ["损坏", "部分"]""", # This will fail parsing
        "test_text_only_no_json_structure": """完全没有按格式要求。就是一段普通描述文字，不包含任何关键词标记。"""
    }

    if os.path.exists(dummy_image_for_parsing_test): # Proceed with mock tests only if dummy image is available
        for test_name, test_text in test_texts_for_parsing.items():
            print(f"\n--- 测试解析: {test_name} ---")
            print(f"模拟API返回内容:\n{test_text}")
            if client and original_create_method : # Ensure client was initialized to have original_create_method
                client.chat.completions.create = mock_create_factory(test_text)
                # analyze_image_content will call the mock_create via client.chat.completions.create
                analysis = analyze_image_content(dummy_image_for_parsing_test) 
                print(f"解析结果 ({test_name}):", json.dumps(analysis, ensure_ascii=False, indent=4))
            elif client is None:
                 print(f"Qwen-VL client 未初始化，跳过对 {test_name} 的模拟解析测试。")
            else: # client exists but original_create_method is None (should not happen if client init was successful)
                 print(f"Qwen-VL client 存在但原始方法未捕获，跳过对 {test_name} 的模拟解析测试。")
    else:
        print(f"\n无法创建/找到微型虚拟图片 {dummy_image_for_parsing_test}，跳过模拟API响应的解析逻辑测试。")


    # Restore original method if it was mocked
    if client and original_create_method:
        client.chat.completions.create = original_create_method
        logging.info("已恢复原始的 client.chat.completions.create 方法。")
