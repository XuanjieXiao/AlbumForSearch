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
import html # For unescaping HTML entities if necessary

# 配置日志格式和等级
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("analyze_images.log", encoding='utf-8')
    ]
)

# 配置 Qwen API
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

MAX_BASE64_IMAGE_CHARS = 7 * 1024 * 1024

def _prepare_image_data_for_qwen(image_path: str, max_chars: int = MAX_BASE64_IMAGE_CHARS) -> str | None:
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
        except Exception as e:
            logging.error(f"保存图片到内存缓冲区失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
            return None

        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        current_base64_size_mb = len(base64_str) / (1024 * 1024)

        if len(base64_str) <= max_chars:
            logging.info(f"图片 '{image_path}' (原始尺寸: {original_dims}, 当前尺寸: {img.size}, 质量: {current_quality}, 尝试 {attempt + 1}) "
                         f"成功编码为Base64，大小: {current_base64_size_mb:.2f} MB。")
            return f"data:image/jpeg;base64,{base64_str}"

        logging.info(f"图片 '{image_path}' Base64大小 ({current_base64_size_mb:.2f} MB) 超过限制. 尝试缩减...")
        if attempt < max_resize_attempts - 1:
            new_width = int(img.width * 0.8)
            new_height = int(img.height * 0.8)
            if new_width < min_dimension or new_height < min_dimension:
                logging.warning(f"图片 '{image_path}' 尺寸已过小 ({new_width}x{new_height})，停止缩减。")
                break
            try:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logging.error(f"图片缩放失败 (尝试 {attempt + 1}) for '{image_path}': {e}")
                break
            if current_quality > 65: current_quality -= 5
        else:
            logging.warning(f"图片 '{image_path}' 达到最大缩减尝试次数，但仍超过大小限制。")
    logging.error(f"图片 '{image_path}' 无法在 {max_resize_attempts} 次尝试内缩减到符合API大小限制。")
    return None

def strip_html(text: str) -> str:
    """移除文本中的HTML标签。"""
    if not text or not isinstance(text, str):
        return ""
    clean = re.compile('<.*?>')
    text_no_html = re.sub(clean, '', text)
    # Optionally unescape HTML entities like &amp;, &lt;, etc.
    # text_no_html = html.unescape(text_no_html)
    return text_no_html.strip()

def clean_and_format_keywords(raw_keywords) -> list:
    """
    清理和格式化从API获取的关键词。
    raw_keywords 可以是列表或单个字符串。
    """
    cleaned_keywords_intermediate = []
    
    items_to_process = []
    if isinstance(raw_keywords, list):
        items_to_process = raw_keywords
    elif isinstance(raw_keywords, str):
        # If it's a single string, it might contain multiple keywords separated by delimiters
        # First, try to split by JSON-like array delimiters if it's a stringified list
        # e.g. "[\"kw1\", \"kw2\"]" or "'kw1', 'kw2'"
        # This is a heuristic and might need refinement
        if raw_keywords.startswith('[') and raw_keywords.endswith(']'):
            try:
                # Attempt to parse it as a JSON list if it's a string representation of one
                potential_list = json.loads(raw_keywords)
                if isinstance(potential_list, list):
                    items_to_process = potential_list
                else: # Parsed but not a list, treat as single item
                    items_to_process = [raw_keywords]
            except json.JSONDecodeError: # Not a valid JSON list string, treat as single item
                 items_to_process = [raw_keywords]
        else:
            items_to_process = [raw_keywords] # Treat as a single string that might contain delimiters

    for item in items_to_process:
        if isinstance(item, str):
            # Remove leading/trailing whitespace
            temp_kw_str = item.strip()
            
            # Iteratively strip surrounding quotes (single or double)
            # Handles cases like ' "keyword" ' or "\"keyword\""
            while len(temp_kw_str) >= 2 and \
                  ((temp_kw_str.startswith('"') and temp_kw_str.endswith('"')) or \
                   (temp_kw_str.startswith("'") and temp_kw_str.endswith("'"))):
                temp_kw_str = temp_kw_str[1:-1].strip() # Strip again after removing quotes

            # Split by common delimiters (comma, Chinese comma, spaces)
            # This handles cases like "keyword1, keyword2" or "keyword1 keyword2"
            # even if they were a single element in the original list
            split_kws = re.split(r'[,\s、]+', temp_kw_str)
            for kw_part in split_kws:
                kw_final = kw_part.strip()
                if kw_final: # Ensure not empty after stripping
                    cleaned_keywords_intermediate.append(kw_final)
        # else: logging.debug(f"Skipping non-string item in keywords: {item}")

    # Deduplicate while preserving order and limit to 10
    final_keywords_unique = []
    for kw in cleaned_keywords_intermediate:
        if kw and kw not in final_keywords_unique: # Ensure not empty again and unique
            final_keywords_unique.append(kw)
    return final_keywords_unique[:10]


def analyze_image_content(image_path: str):
    if not client:
        logging.error("Qwen-VL client 未初始化。无法分析图片。")
        return {"description": "", "keywords": []}

    logging.info(f"准备使用Qwen-VL分析图片: {image_path}")
    data_url = _prepare_image_data_for_qwen(image_path)
    if not data_url:
        return {"description": "", "keywords": []}

    prompt_text = """请你严格作为图片分析JSON生成器运行。你的唯一输出必须是一个符合下述规范的JSON对象。

图片分析要求：
1.  **描述 (description)**: 用中文详细描述图片内容、物品、元素和场景。描述应简洁明了。此描述文本中绝对不允许包含任何HTML标签 (如 `<p>`, `<b>` 等)。如果描述中需要使用英文双引号 (`"`)，则必须将其转义为 `\\"`.
2.  **关键词 (keywords)**: 从图片内容中提取最多10个最具代表性的中文关键词。此字段必须是一个JSON数组，其中每个元素都是一个独立的、纯净的中文关键词字符串。例如：`["天空", "建筑", "城市"]`。单个关键词字符串内部不应包含任何引号、逗号或其他分隔符。

JSON输出格式 (严格遵守，不要添加任何额外字符、注释或Markdown标记如 \`\`\`json):
{
  "description": "图片描述文本。",
  "keywords": ["关键词示例1", "关键词示例2"]
}

字数限制：整个JSON响应（包括JSON结构本身和所有文本内容）的总字数不要超过490字。"""

    MAX_API_ATTEMPTS = 3
    last_successful_api_content_if_unparsed = None
    last_api_call_exception_details = None

    for attempt_num in range(MAX_API_ATTEMPTS):
        current_attempt_str = f"Attempt {attempt_num + 1}/{MAX_API_ATTEMPTS}"
        logging.info(f"开始图片分析 - {current_attempt_str} for image: {image_path}")

        try:
            response = client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]}
                ],
                temperature=0.7,
            )
            result_content = response.choices[0].message.content.strip()
            last_successful_api_content_if_unparsed = result_content
            last_api_call_exception_details = None
            logging.info(f"图片分析API调用成功: {image_path} ({current_attempt_str})")
            logging.debug(f"Qwen-VL原始输出 ({current_attempt_str}): '{result_content}'")

            json_str_candidate = None
            # Attempt 1: Extract from ```json ... ``` block
            match_md = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', result_content, re.DOTALL)
            if match_md:
                json_str_candidate = match_md.group(1).strip()
                logging.info(f"从Markdown JSON块提取候选JSON ({current_attempt_str}).")
            else:
                # Attempt 2: Extract from first '{' to last '}'
                # This is greedy and assumes the main JSON is the only or primary one.
                match_curly = re.search(r'(\{[\s\S]*\})', result_content, re.DOTALL)
                if match_curly:
                    json_str_candidate = match_curly.group(1).strip()
                    logging.info(f"从裸JSON对象结构提取候选JSON ({current_attempt_str}).")
                else:
                    logging.warning(f"在API响应中未找到可识别的JSON结构 ({current_attempt_str}). 响应片段: '{result_content[:200]}...'")
            
            if json_str_candidate:
                try:
                    # Pre-parsing cleaning
                    json_str_candidate = json_str_candidate.replace('\u00a0', ' ') # Replace non-breaking spaces
                    # Add any other specific pre-cleaning steps here if known issues persist

                    parsed_data = json.loads(json_str_candidate)
                    
                    description_raw = parsed_data.get("description", "")
                    description = strip_html(description_raw) if isinstance(description_raw, str) else ""

                    keywords_raw = parsed_data.get("keywords", [])
                    keywords = clean_and_format_keywords(keywords_raw)

                    logging.info(f"JSON成功解析和清理 ({current_attempt_str}).")
                    return {"description": description, "keywords": keywords}

                except json.JSONDecodeError as je:
                    logging.warning(f"JSON解析失败 ({current_attempt_str}) for candidate: '{json_str_candidate[:300]}...'. Error: {je}")
                    # Fall through to the next API attempt or final fallback
            
            # If json_str_candidate is None or parsing failed, and it's not the last API attempt
            if attempt_num < MAX_API_ATTEMPTS - 1 and (not json_str_candidate or (json_str_candidate and 'parsed_data' not in locals())):
                 logging.info(f"当前尝试解析失败，将进行下一次API调用 ({attempt_num + 2}/{MAX_API_ATTEMPTS}).")
                 # time.sleep(1) # Optional delay

        except Exception as e:
            logging.error(f"Qwen-VL分析图片API调用失败 ({current_attempt_str}): {e}", exc_info=True)
            last_api_call_exception_details = str(e)
            if attempt_num < MAX_API_ATTEMPTS - 1:
                logging.info(f"由于API调用错误，将进行下一次尝试 ({attempt_num + 2}/{MAX_API_ATTEMPTS}).")
                # time.sleep(1) # Optional delay
    
    # Fallback after all API attempts
    logging.error(f"所有 {MAX_API_ATTEMPTS} 次图片分析API尝试均已完成，但未能成功解析出有效JSON。")
    if last_successful_api_content_if_unparsed:
        # Attempt line-based parsing as a last resort on the last good API response
        logging.info(f"尝试对最后一次成功的API响应进行基于行的宽松解析: '{last_successful_api_content_if_unparsed[:200]}...'")
        lines = last_successful_api_content_if_unparsed.splitlines()
        extracted_description_parts = []
        extracted_keywords_parts = []
        keywords_line_found = False
        keyword_prefixes = ["关键词：", "关键词:", "keywords:", "Keywords:"]

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line: continue
            
            is_kw_line_segment = False
            if not keywords_line_found: # Only check for keyword prefix if not already found
                for prefix in keyword_prefixes:
                    if stripped_line.lower().startswith(prefix.lower()):
                        kw_data = stripped_line[len(prefix):].strip()
                        extracted_keywords_parts.append(kw_data)
                        keywords_line_found = True
                        is_kw_line_segment = True
                        break
            elif keywords_line_found and not any(stripped_line.lower().startswith(p.lower()) for p in keyword_prefixes):
                # If keywords line was found, subsequent lines might be continuations of keywords
                # unless they look like a new description or another keyword prefix.
                # This heuristic is simple; complex layouts might break it.
                 extracted_keywords_parts.append(stripped_line)
                 is_kw_line_segment = True


            if not is_kw_line_segment and not keywords_line_found: # Add to description if not part of keywords
                extracted_description_parts.append(stripped_line)
        
        final_desc_fallback = strip_html(" ".join(extracted_description_parts))
        # Join all keyword parts then process with the robust cleaner
        final_keywords_fallback_str = " ".join(extracted_keywords_parts)
        final_keywords_fallback = clean_and_format_keywords(final_keywords_fallback_str)

        if final_desc_fallback or final_keywords_fallback:
            logging.info(f"通过宽松行解析获得回退结果: desc='{final_desc_fallback[:50]}...', keywords={final_keywords_fallback}")
            return {"description": final_desc_fallback, "keywords": final_keywords_fallback}
        
        logging.warning("宽松行解析也未能提取有效信息。将最后一次API响应作为描述返回。")
        return {"description": strip_html(last_successful_api_content_if_unparsed), "keywords": []}
        
    elif last_api_call_exception_details:
        logging.error(f"所有API调用均失败。最后记录的API错误: {last_api_call_exception_details}")
        return {"description": f"API调用在所有{MAX_API_ATTEMPTS}次尝试后均失败: {last_api_call_exception_details}", "keywords": []}
    else:
        logging.error("所有尝试后均无有效内容或错误记录。返回通用失败信息。")
        return {"description": "图片分析失败，请稍后重试。", "keywords": []}


if __name__ == "__main__":
    test_image_dir = "test_uploads_qwen" 
    os.makedirs(test_image_dir, exist_ok=True)
    test_image_file = os.path.join(test_image_dir, "example_test_image.jpg")

    if not os.path.exists(test_image_file):
        try:
            Image.new('RGB', (800, 600), color = 'skyblue').save(test_image_file, "JPEG", quality=90)
            logging.info(f"创建了虚拟测试图片: {test_image_file}")
        except Exception as e_create_img:
            logging.error(f"创建虚拟测试图片失败: {e_create_img}")

    if os.path.exists(test_image_file) and client:
        print(f"\n--- 测试对图片 {test_image_file} 的完整分析流程 ---")
        analysis_result = analyze_image_content(test_image_file)
        print("\n单张图片分析结果:")
        print(json.dumps(analysis_result, ensure_ascii=False, indent=4))
    elif not client:
        print(f"\nQwen-VL client 未初始化，跳过对 {test_image_file} 的 API 分析测试。")
    else:
        logging.warning(f"测试图片 {test_image_file} 不存在。")

    print("\n--- 测试不同格式响应的解析逻辑 (使用Mock API) ---")
    class MockChoice:
        def __init__(self, content): self.message = MockMessage(content)
    class MockMessage:
        def __init__(self, content): self.content = content
    class MockResponse:
        def __init__(self, content): self.choices = [MockChoice(content)]

    original_create_method = client.chat.completions.create if client else None
    def mock_create_factory(text_to_return):
        def _mock(*args, **kwargs):
            logging.info(f"Mock API call, returning: '{text_to_return}'")
            return MockResponse(text_to_return)
        return _mock

    dummy_image_for_parsing_test = os.path.join(test_image_dir, "tiny_dummy_for_parser_test.jpg")
    if not os.path.exists(dummy_image_for_parsing_test):
        try: Image.new('RGB', (10,10), color='blue').save(dummy_image_for_parsing_test, "JPEG")
        except Exception: logging.error(f"无法创建用于解析测试的微型虚拟图片 {dummy_image_for_parsing_test}")

    test_texts_for_parsing = {
        "valid_json": """{
"description": "这是一张有效的JSON描述，包含<p>HTML</p>标签。",
"keywords": ["有效", "JSON", "  带空格  ", "带引号的\"关键词\""]
}""",
        "json_with_markdown_fences": """```json
{
  "description": "城市夜景，灯火辉煌。",
  "keywords": ["城市", "夜景", "灯光", "重复关键词", "重复关键词"]
}
```""",
        "text_before_json_block": """一些无关的文本信息在JSON块之前。
```json
{
  "description": "描述在无关文本之后。",
  "keywords": ["前缀文本", "解析"]
}
```""",
        "json_with_malformed_keywords_list_string": """{
"description": "关键词列表是字符串，且内部格式混乱",
"keywords": "['机场', '\\",\\"飞行器\\",', '  \\"地面设施\\"  ', '天桥', [], '']"
}""",
        "json_with_malformed_keywords_list_items": """{
"description": "关键词列表内元素格式混乱",
"keywords": ["机场", " , 飞行器, ", "地面设施, 、 天桥  ", ["嵌套列表"], {"对象":"不行"}, "  "]
}""",
        "only_curly_braces_json": """
{
  "description": "只有花括号包裹的JSON，没有markdown。天空是蓝色的。",
  "keywords": ["天空", "蓝色", "云彩", "户外", "自然", "风景", "清新", "宁静", "广阔", "明亮", "额外第11个"]
}""",
        "line_based_fallback_needed": """这是一个美丽的湖泊，湖边有绿树环绕，没有JSON结构。
关键词：湖泊、绿树、风景、自然、宁静祥和""",
        "completely_malformed_json_string": """{ description": "引号缺失或错位, keywords: ["测试", "损坏"] }""",
        "control_char_in_string_simulated": '{ "description": "描述中包含一个\n换行符和\t制表符。", "keywords": ["控制字符", "测试"] }',
         "error_example_1_from_user": """机场, 飞机跑道, 新加坡航空飞机, 航空站台, 停靠区, 黄色标记线, 天桥连接口

```json
{
  "description": "在繁忙的国际机场内拍摄的一张照片，在前景中有新加坡航空公司标志明显的白色客机停泊于黄色标示的地面上；背景则是其他不同型号与颜色的飞机以及远处模糊可见的大楼结构，天空呈现淡蓝色调。",
  "keywords": ["机场", "\",\"飞行器\", "\"地面设施\",\"地标性建筑\",\"空中交通控制","航向指示牌"]
}
```""",
        "error_example_2_from_user_invalid_control_char": """{
  "description": "这是一张展示在机场天际走廊Sky Bridge区域的照片。画面左侧有两名游客正在拍照或观看窗外景色；右侧坐着一位穿着紫色连帽衫的人正专注地使用笔记本电脑工作，并且旁边放着一个红色背包。背景墙上有大大的白色英文字母'Sky Bridge'以及对应的汉字翻译‘天際走廊’，并附有一个桥梁结构的设计图示。整个环境显得现代而整洁。”,
  "keywords": ["机场", "天际走廊", "Sky Bridge", "乘客", "摄影", "休息区", "设计图纸", "现代化建筑", "旅游景点", "户外活动"]
}"""
    }

    if os.path.exists(dummy_image_for_parsing_test) and client and original_create_method:
        for test_name, test_text in test_texts_for_parsing.items():
            print(f"\n--- 测试解析: {test_name} ---")
            # print(f"模拟API返回内容:\n{test_text}") # Optional: print full text
            client.chat.completions.create = mock_create_factory(test_text)
            analysis = analyze_image_content(dummy_image_for_parsing_test) 
            print(f"解析结果 ({test_name}):", json.dumps(analysis, ensure_ascii=False, indent=4))
    elif not client:
        print(f"Qwen-VL client 未初始化，跳过模拟API响应的解析逻辑测试。")
    else: # Client might be init, but dummy image failed
        print(f"无法创建/找到微型虚拟图片 {dummy_image_for_parsing_test} 或原始方法未正确备份，跳过模拟API响应的解析逻辑测试。")

    if client and original_create_method: # Restore original method
        client.chat.completions.create = original_create_method
        logging.info("已恢复原始的 client.chat.completions.create 方法。")
