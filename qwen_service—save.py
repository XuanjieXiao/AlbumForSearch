# qwen_service.py
from openai import OpenAI
import base64
import os
import logging
import json
import re # 引入正则表达式库

# 配置日志格式和等级
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', # 添加了module
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("analyze_images.log", encoding='utf-8')
    ]
)

# 配置 Qwen API
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "HoUbVVd_L1Z0uLJJiq5ND13yfDreU4pkTHwoTbU_EMp31G_OLx_ONh5fIoa37cNM4mRfAvst7bR_9VUfi4-QXg") 
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://www.sophnet.com/api/open-apis/v1")

client = None
if QWEN_API_KEY and QWEN_API_KEY != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
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


def file_to_base64(file_path):
    try:
        with open(file_path, 'rb') as file:
            file_data = file.read()
        base64_data = base64.b64encode(file_data).decode('utf-8')
        logging.debug(f"文件已编码为base64: {file_path}")
        return base64_data
    except Exception as e:
        logging.error(f"读取文件 '{file_path}' 出错: {e}")
        return None

def analyze_image_content(image_path: str):
    """
    分析单个图片内容，返回结构化的描述和关键词。
    包含多种解析回退机制。
    """
    if not client:
        logging.error("Qwen-VL client 未初始化。无法分析图片。")
        return {"description": "", "keywords": []}
        
    logging.info(f"开始使用Qwen-VL分析图片: {image_path}")
    base64_str = file_to_base64(image_path)
    if not base64_str:
        logging.error(f"图片读取或Base64编码失败: {image_path}")
        return {"description": "", "keywords": []}

    data_url = "data:image/jpeg;base64," + base64_str 
    
    prompt_text = """请你用中文详细描述这张图片的内容，要求内容简洁明了，突出图片的主要元素和场景，从图片内容中提取出最多10个最具代表性的中文关键词，关键词之间请用单个英文逗号“,”隔开。
最后，请按照如下JSON格式输出，不要包含任何JSON格式之外的额外解释或文字：
{
  "description": "在此详细描述图片内容，突出主要元素、场景、动作、氛围等信息。",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}"""

    try:
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
            temperature=0.7, # 较高的温度可能导致输出多样性，但也可能不严格遵循格式
            # max_tokens=600 # 限制输出长度，避免过长无法解析的文本
        )
        result_content = response.choices[0].message.content.strip() # 去除首尾空白
        # breakpoint()
        logging.info(f"图片分析成功: {image_path}")
        logging.debug(f"Qwen-VL原始输出 (处理前): '{result_content}'")

        # 解析策略：
        # 1. 尝试直接解析为JSON
        try:
            # 有时模型会在JSON前后添加 markdown ```json ... ``` 标记
            if result_content.startswith("```json"):
                result_content = result_content[7:] # 移除 ```json
                if result_content.endswith("```"):
                    result_content = result_content[:-3] # 移除 ```
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

            # 2. 尝试从字符串中提取JSON对象 (处理前后有无关文本的情况)
            try:
                # 使用正则表达式查找第一个 '{' 到最后一个 '}' 之间的内容
                # 这假设JSON对象是主体内容中唯一或最主要的JSON结构
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


            # 3. 尝试解析 "描述...\n关键词：关键字1、关键字2" 格式
            logging.info("尝试解析 '描述...换行...关键词：' 格式。")
            lines = result_content.splitlines()
            extracted_description_parts = []
            extracted_keywords = []
            keywords_line_found = False
            
            # 常见关键词前缀
            keyword_prefixes = ["关键词：", "关键词:", "keywords:", "Keywords:"] 

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line: # 跳过空行
                    continue

                # 检查是否为关键词行
                is_keyword_line = False
                for prefix in keyword_prefixes:
                    if stripped_line.lower().startswith(prefix.lower()):
                        kw_data = stripped_line[len(prefix):].strip()
                        # 优先使用中文顿号 '、' 分割，其次是英文逗号 ','
                        if '、' in kw_data:
                            extracted_keywords = [kw.strip() for kw in kw_data.split('、') if kw.strip()]
                        elif ',' in kw_data:
                            extracted_keywords = [kw.strip() for kw in kw_data.split(',') if kw.strip()]
                        else: # 如果都没有，尝试用空格分割（作为最后手段）
                            extracted_keywords = [kw.strip() for kw in kw_data.split(' ') if kw.strip()]
                        keywords_line_found = True
                        is_keyword_line = True
                        break 
                
                if is_keyword_line:
                    continue # 如果是关键词行，则不作为描述的一部分

                # 如果还未找到关键词行，则当前行可能是描述的一部分
                if not keywords_line_found:
                    extracted_description_parts.append(stripped_line)
                # 如果已经找到了关键词行，但当前行不是关键词行 (例如API在关键词后又输出了其他内容)，则忽略
            
            final_description = " ".join(extracted_description_parts).strip()

            if final_description or extracted_keywords:
                logging.info(f"通过解析'描述...关键词：'格式获得: desc='{final_description[:50]}...', keywords={extracted_keywords}")
                return {"description": final_description, "keywords": extracted_keywords[:10]}
            else:
                logging.error(f"所有解析尝试均失败。原始返回内容 (已strip): '{result_content}'")
                return {"description": "无法解析描述", "keywords": []}

    except Exception as e:
        logging.error(f"Qwen-VL分析图片 '{image_path}' 失败 (API调用或未知错误): {e}", exc_info=True)
        return {"description": "", "keywords": []}


if __name__ == "__main__":
    # 测试单个图片分析
    test_image_file = "./uploads/68c00011-2fe7-4e83-a4f1-97707c481fee.jpg" 
    if not os.path.exists(test_image_file):
        # 创建一个虚拟的测试图片，如果它不存在
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 100), color = 'red')
            draw = ImageDraw.Draw(img)
            draw.text((10,10), "Test", fill=(255,255,0))
            img.save(test_image_file)
            logging.info(f"创建了虚拟测试图片: {test_image_file}")
        except Exception as e_create_img:
            logging.error(f"创建虚拟测试图片失败: {e_create_img}")
            # logging.error(f"测试图片 {test_image_file} 不存在。请创建或修改路径。")
    
    if os.path.exists(test_image_file):
        analysis_result = analyze_image_content(test_image_file)
        print("\n单张图片分析结果:")
        print(json.dumps(analysis_result, ensure_ascii=False, indent=4))
    else:
        logging.error(f"测试图片 {test_image_file} 仍然不存在，跳过测试。")

    # 测试不同格式的解析
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

    # 模拟 client.chat.completions.create 方法
    original_create_method = None
    if client and hasattr(client.chat.completions, 'create'):
        original_create_method = client.chat.completions.create

    def mock_create(*args, **kwargs):
        # 从kwargs中获取messages，找到图片分析的文本内容
        # 这里我们直接返回预设的测试文本
        return MockResponse(kwargs.get('mock_return_text', ''))

    if client:
        client.chat.completions.create = mock_create

    test_text_1 = """这是一个美丽的湖泊，湖边有绿树环绕。
关键词：湖泊、绿树、风景、自然"""
    print(f"\n测试文本1:\n{test_text_1}")
    if client: kwargs = {'mock_return_text': test_text_1} # 传递给mock_create
    analysis1 = analyze_image_content("dummy_path_for_test1.jpg") # image_path不重要，因为API被mock了
    print("解析结果1:", json.dumps(analysis1, ensure_ascii=False, indent=4))

    test_text_2 = """```json
{
  "description": "城市夜景，灯火辉煌。",
  "keywords": ["城市", "夜景", "灯光"]
}
```"""
    print(f"\n测试文本2:\n{test_text_2}")
    if client: kwargs = {'mock_return_text': test_text_2}
    analysis2 = analyze_image_content("dummy_path_for_test2.jpg")
    print("解析结果2:", json.dumps(analysis2, ensure_ascii=False, indent=4))

    test_text_3 = """完全没有按格式要求。就是一段话。"""
    print(f"\n测试文本3:\n{test_text_3}")
    if client: kwargs = {'mock_return_text': test_text_3}
    analysis3 = analyze_image_content("dummy_path_for_test3.jpg")
    print("解析结果3:", json.dumps(analysis3, ensure_ascii=False, indent=4))
    
    test_text_4 = """{"description":"JSON对象在文本中间","keywords":["嵌入式","测试"]} 其他无关文字。"""
    print(f"\n测试文本4:\n{test_text_4}")
    if client: kwargs = {'mock_return_text': test_text_4}
    analysis4 = analyze_image_content("dummy_path_for_test4.jpg")
    print("解析结果4:", json.dumps(analysis4, ensure_ascii=False, indent=4))


    # 恢复原始方法 (如果mock了)
    if client and original_create_method:
        client.chat.completions.create = original_create_method

