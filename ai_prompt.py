import re
import requests
import json
import os
import time
import base64
from typing import Dict, Tuple, List
from PIL import Image
import io

_CACHE: Dict[Tuple[str, str, str], str] = {}
_CONFIG = None
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ai_config.json")

def _load_config():
    default = {
        "base_url": "http://127.0.0.1:1234",
        "token": "",
        "timeout": 60,
        "last_model": ""
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                user = json.load(f)
                default.update(user)
    except:
        pass
    return default

def _save_config(base_url=None, token=None, model=None):
    global _CONFIG
    try:
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        
        if base_url:
            cfg["base_url"] = base_url
        if token:
            cfg["token"] = token
        if model:
            cfg["last_model"] = model
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        _CONFIG = _load_config()
    except:
        pass

_CONFIG = _load_config()

def _clean_thinking_response(text: str) -> str:
    """清理响应中的思考过程，只保留最终结果"""
    # 定义需要清理的思考模式
    thinking_patterns = [
        r'^Thinking Process:.*?\n\n',
        r'^思考过程：.*?\n\n',
        r'^让我想想.*?\n',
        r'^首先.*?\n',
        r'^然后.*?\n',
        r'^接下来.*?\n',
        r'^综上所述.*?\n',
        r'^我认为.*?\n',
        r'^分析：.*?\n',
        r'^思考：.*?\n',
        r'^【思考】.*?\n',
    ]
    
    # 逐行清理
    lines = text.split('\n')
    cleaned_lines = []
    skip_next = False
    
    for line in lines:
        should_skip = False
        for pattern in thinking_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                should_skip = True
                break
        
        if re.match(r'^Thinking Process:', line, re.IGNORECASE):
            skip_next = True
            continue
        
        if skip_next:
            if line.strip() == '':
                skip_next = False
            continue
        
        if not should_skip:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    if re.search(r'Thinking Process|思考过程|让我想想|首先|然后', result, re.IGNORECASE):
        paragraphs = result.split('\n\n')
        if len(paragraphs) > 1:
            result = paragraphs[-1].strip()
    
    return result

def _ai_chat(url: str, token: str, model: str, msg: str, timeout: int = 60, image_base64: str = None) -> str:
    """通用的AI聊天接口，支持OpenAI兼容的API，支持多模态"""
    base = url.rstrip('/')
    
    if image_base64:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
        payload = {"model": model, "messages": messages, "temperature": 0.1}
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        try:
            r = requests.post(f"{base}/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    res = choice["message"]["content"].strip()
                    res = _clean_thinking_response(res)
                    return res
                elif "text" in choice:
                    res = choice["text"].strip()
                    res = _clean_thinking_response(res)
                    return res
                elif "content" in choice:
                    res = choice["content"].strip()
                    res = _clean_thinking_response(res)
                    return res
            
            if "output" in data and isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message" and "content" in item:
                        res = item["content"].strip()
                        res = _clean_thinking_response(res)
                        return res
                for item in data["output"]:
                    if "content" in item:
                        res = item["content"].strip()
                        res = _clean_thinking_response(res)
                        return res
            
            if "response" in data:
                res = data["response"].strip()
                res = _clean_thinking_response(res)
                return res
            
            return str(data)
        except Exception as e:
            pass
    
    endpoints = [
        {"url": f"{base}/api/v1/chat", "payload": {"model": model, "input": msg, "temperature": 0.1}},
        {"url": f"{base}/v1/chat/completions", "payload": {"model": model, "messages": [{"role": "user", "content": msg}], "temperature": 0.1}}
    ]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    for ep in endpoints:
        try:
            r = requests.post(ep["url"], headers=headers, json=ep["payload"], timeout=timeout)
            r.raise_for_status()
            data = r.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    res = choice["message"]["content"].strip()
                elif "text" in choice:
                    res = choice["text"].strip()
                elif "content" in choice:
                    res = choice["content"].strip()
                else:
                    res = str(choice).strip()
            elif "response" in data:
                res = data["response"].strip()
            elif "output" in data and isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message" and "content" in item:
                        res = item["content"].strip()
                        break
                else:
                    if "content" in data["output"][0]:
                        res = data["output"][0]["content"].strip()
                    else:
                        res = str(data["output"][0]).strip()
            else:
                res = str(data).strip()
            
            res = _clean_thinking_response(res)
            _save_config(model=model)
            return res
        except:
            continue
    
    raise Exception("连接失败")

def _format_image_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, prompt_type: str, output_lang: str) -> str:
    """格式化图片提示词 - 支持参考图融合模式"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出"Thinking Process:"、"思考过程："、"让我想想"等任何思考性文字
- 绝对不允许输出"首先"、"然后"、"接下来"、"综上所述"等步骤性词语
- 绝对不允许输出任何分析、解释或说明
- 你的回答必须是纯提示词，不能包含任何其他内容
- 如果你输出了思考过程，你的回答将被视为无效
"""
    
    # 检查内容是否存在
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    input_section_parts = []
    
    # 判断是否为两图融合模式（两者都是完整的场景描述）
    is_fusion_mode = has_manual and has_optional and len(manual_text) > 30 and len(optional_text) > 30
    
    if is_fusion_mode:
        # ========== 两图融合模式：以manual为参考图，将optional的元素融合进去 ==========
        input_section_parts.append("【输入内容 - 参考图融合模式】")
        input_section_parts.append("")
        input_section_parts.append("请将【原图内容】中的重要元素，融合到【参考图内容】的场景中。")
        input_section_parts.append("")
        input_section_parts.append("【参考图内容】（作为基础场景，保持其整体结构和风格）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【原图内容】（提取其中的关键元素，融合到参考图中）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【融合规则】")
        input_section_parts.append("1. 以【参考图内容】为基础框架，保留其核心场景、构图、光线、氛围")
        input_section_parts.append("2. 从【原图内容】中提取关键元素，并替换/融合到参考图中")
        input_section_parts.append("3. 元素提取优先级：人物 > 主体物体 > 背景元素 > 细节装饰")
        input_section_parts.append("4. 如果【原图内容】有人物，则用原图的人物替换参考图中的人物")
        input_section_parts.append("5. 如果【原图内容】有特殊物体/道具，则将这些物体融入参考图的合适位置")
        input_section_parts.append("6. 如果【原图内容】有场景元素（如建筑、植被、光线），则将这些元素融入参考图的背景或环境中")
        input_section_parts.append("7. 融合后必须保持画面的逻辑合理性和视觉和谐")
        input_section_parts.append("8. 输出一个完整的提示词，描述融合后的最终画面")
        input_section_parts.append("")
        input_section_parts.append("【融合示例1 - 人物替换】")
        input_section_parts.append("参考图: '一个女孩站在沙滩上，阳光明媚，海浪拍打岸边'")
        input_section_parts.append("原图: '一个戴帽子的男孩在森林里看书'")
        input_section_parts.append("融合结果: '一个戴帽子的男孩站在沙滩上，阳光明媚，海浪拍打岸边，男孩手里拿着一本书'")
        input_section_parts.append("")
        input_section_parts.append("【融合示例2 - 场景融合】")
        input_section_parts.append("参考图: '三个人在公园里散步，秋季，金黄色树叶'")
        input_section_parts.append("原图: '传统的中国山村，白墙黑瓦，晾晒的农作物'")
        input_section_parts.append("融合结果: '三个人在中国传统山村的石板路上散步，背景是白墙黑瓦的民居，晾晒架上挂着红辣椒和玉米，秋季的暖阳洒在山村中，金黄色的树叶点缀其间'")
        
    elif has_manual and has_optional:
        # ========== 原有模式：手动输入修改原图 ==========
        input_section_parts.append("【输入内容】")
        input_section_parts.append("")
        input_section_parts.append("以下是两部分需要综合的内容：")
        input_section_parts.append("")
        input_section_parts.append("1. 【用户修改指令】（最高优先级，必须严格遵守）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("2. 【原图内容】（需要被修改的基础图片）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【处理要求】")
        input_section_parts.append("请根据【用户修改指令】修改【原图内容】，生成最终的提示词。")
        input_section_parts.append("- 以【原图内容】为基础")
        input_section_parts.append("- 应用【用户修改指令】中的所有修改要求")
        input_section_parts.append("- 保持原图中没有被要求修改的部分不变")
        input_section_parts.append("- 当两者冲突时，优先遵循用户修改指令")
        input_section_parts.append("")
        input_section_parts.append("示例：")
        input_section_parts.append("用户修改指令: \"把头发改成红色\"")
        input_section_parts.append("原图内容: \"一个穿蓝色裙子的女孩\"")
        input_section_parts.append("最终提示词: \"一个穿蓝色裙子的女孩，红色头发\"")
        
    elif has_manual and not has_optional:
        # ========== 只有手工提示词 ==========
        input_section_parts.append("【用户输入】")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【处理要求】")
        input_section_parts.append("请根据用户输入生成提示词。")
        
    elif not has_manual and has_optional:
        # ========== 只有图片描述 ==========
        input_section_parts.append("【图片识别内容】")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【处理要求】")
        input_section_parts.append("请根据图片识别内容，生成高质量的提示词。")
    
    if not input_section_parts:
        return None
    
    input_section = "\n".join(input_section_parts)
    
    if detail_level == "标准":
        strategy = """【处理策略：严格遵循原文】
- 只做必要的语法修正和错别字修正
- 补全描述中明显缺失的关键元素
- 严禁自行添加原文没有的场景、环境、细节
- 保持原提示词的核心内容和风格不变
- 输出长度与原文相近"""
    elif detail_level == "详细":
        strategy = """【处理策略：适度补充】
- 在原文基础上进行合理的细节补充
- 可补充光影、质感、构图等与原文相关的内容
- 不添加原文完全未提及的新元素
- 描述可适当细化，但不改变原意
- 输出长度约为原文的1.5-2倍"""
    else:
        strategy = """【处理策略：全面丰富】
- 可充分发挥想象力，对原文进行全方位的丰富和扩展
- 添加精致的视觉细节：光影、质感、色彩、构图、氛围
- 补充完整的场景环境和情绪基调
- 追求电影级的画面效果
- 输出长度不限，追求极致细节"""
    
    # 根据模式调整指引
    if is_fusion_mode:
        mode_guide = """【参考图融合模式说明】
你的任务是将【原图】中的关键元素融合到【参考图】中：
- 以【参考图】为基础场景，保持其整体结构
- 从【原图】中提取人物、物体、场景元素
- 用原图的人物替换参考图中的人物（如果存在）
- 将原图的其他元素自然融入参考图的合适位置
- 输出一个完整、连贯的融合后场景描述"""
    elif mode == "文生图":
        if prompt_type == "正向":
            mode_guide = """【文生图正向提示词模式】
从零构建画面，侧重：
- 完整的场景设定和元素构建
- 主体形象的详细刻画
- 风格、构图、光影的明确描述"""
        else:
            mode_guide = """【文生图负向提示词模式】
排除不需要的元素，侧重：
- 列出需要避免的瑕疵和缺陷
- 排除不符合预期的元素
- 只针对用户输入中明确提到的负面内容
- 不添加用户未提及的负面元素"""
    else:
        if prompt_type == "正向":
            mode_guide = """【图生图正向提示词模式】
基于参考图进行重绘和优化，侧重：
- 综合【用户手动输入】和【原图识别内容】
- 用户手动输入是修改的方向，原图识别是基础内容
- 保留原图的基本元素，同时应用用户的修改要求
- 最终提示词应该融合两者的内容"""
        else:
            mode_guide = """【图生图负向提示词模式】
基于原图排除不需要的元素，侧重：
- 综合【用户手动输入】和【原图识别内容】
- 用户手动输入指定要排除的元素，原图识别是基础内容
- 保留原图的基本结构，去除用户不想要的元素"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
        output_format = "输出中文提示词"
    else:
        lang_instruction = "必须使用英文输出"
        output_format = "输出英文提示词"
    
    # 针对融合模式的额外强调
    extra_emphasis = ""
    if is_fusion_mode:
        extra_emphasis = """
【特别强调 - 参考图融合】
- 【参考图内容】是基础场景，必须保持其整体结构和风格
- 从【原图内容】中提取元素并融合到参考图中
- 如果原图有人物，必须用原图的人物替换参考图中的人物
- 如果原图有特殊物体/道具，将它们融入参考图的合适位置
- 如果原图有场景元素，将它们融入参考图的背景中
- 融合后的画面必须看起来自然、合理、和谐
"""
    
    return f"""{header}
{thinking_ban}

你是一个专业的Stable Diffusion提示词转换专家。
请将以下用户输入转换为高质量的{prompt_type}提示词，用于AI绘画。

【核心原则】
- 以【参考图内容】为基础框架，从【原图内容】中提取元素进行融合
- 原图中的人物优先级最高，用于替换参考图中的人物
- 原图中的物体和场景元素作为补充，融入参考图
- 融合后的画面应该看起来像一张完整、和谐的照片

【生成模式】
{mode_guide}
{extra_emphasis}
{strategy}

【最重要规则 - 必须绝对遵守】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"、"接下来"、"我认为"等任何思考性词语
3. 禁止输出"Thinking Process:"、"思考过程："等任何思考标签
4. 禁止输出任何标签，如"思考过程："、"分析："、"转换后的提示词："、"结果："等
5. 禁止输出任何解释性文字
6. 直接输出最终的提示词，不要有任何前缀或后缀
7. {lang_instruction}

【输出格式】
直接输出{output_format}，只输出提示词本身，不包含任何其他内容。

{input_section}

直接输出提示词："""

def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词 - 支持参考图融合模式"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出"Thinking Process:"、"思考过程："、"让我想想"等任何思考性文字
- 绝对不允许输出"首先"、"然后"、"接下来"、"综上所述"等步骤性词语
- 绝对不允许输出任何分析、解释或说明
- 你的回答必须是纯提示词，不能包含任何其他内容
- 如果你输出了思考过程，你的回答将被视为无效
"""
    
    # 检查内容是否存在
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    input_section_parts = []
    
    # 判断是否为两图融合模式（两者都是完整的场景描述）
    is_fusion_mode = has_manual and has_optional and len(manual_text) > 30 and len(optional_text) > 30
    
    if is_fusion_mode:
        # ========== 两图融合模式：以manual为参考图，将optional的元素融合进去 ==========
        input_section_parts.append("【输入内容 - 参考图融合模式】")
        input_section_parts.append("")
        input_section_parts.append("请将【原图内容】中的重要元素，融合到【参考图内容】的视频场景中。")
        input_section_parts.append("")
        input_section_parts.append("【参考图内容】（作为基础视频场景，保持其整体结构、动作和风格）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【原图内容】（提取其中的关键元素，融合到参考视频中）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【融合规则】")
        input_section_parts.append("1. 以【参考图内容】为基础框架，保留其核心场景、动作、光线、氛围和运动轨迹")
        input_section_parts.append("2. 从【原图内容】中提取关键元素，并替换/融合到参考视频中")
        input_section_parts.append("3. 元素提取优先级：人物动作 > 主体物体 > 背景元素 > 细节装饰")
        input_section_parts.append("4. 如果【原图内容】有人物，则用原图的人物替换参考视频中的人物（保持参考视频的动作和运动）")
        input_section_parts.append("5. 如果【原图内容】有特殊物体/道具，则将这些物体融入参考视频的合适位置")
        input_section_parts.append("6. 如果【原图内容】有场景元素（如建筑、植被、光线），则将这些元素融入参考视频的背景或环境中")
        input_section_parts.append("7. 融合后必须保持视频的逻辑合理性、视觉和谐以及运动的连贯性")
        input_section_parts.append("8. 输出一个完整的视频提示词，描述融合后的最终视频画面和动作")
        input_section_parts.append("")
        input_section_parts.append("【融合示例1 - 人物替换】")
        input_section_parts.append("参考视频: '一个女孩在沙滩上奔跑，阳光明媚，海浪拍打岸边'")
        input_section_parts.append("原图: '一个戴帽子的男孩在森林里看书'")
        input_section_parts.append("融合结果: '一个戴帽子的男孩在沙滩上奔跑，阳光明媚，海浪拍打岸边，男孩手里拿着一本书，书页随风翻动'")
        input_section_parts.append("")
        input_section_parts.append("【融合示例2 - 场景融合】")
        input_section_parts.append("参考视频: '三个人在公园里散步，秋季，金黄色树叶飘落'")
        input_section_parts.append("原图: '传统的中国山村，白墙黑瓦，晾晒的农作物'")
        input_section_parts.append("融合结果: '三个人在中国传统山村的石板路上散步，背景是白墙黑瓦的民居，晾晒架上挂着红辣椒和玉米，秋季的暖阳洒在山村中，金黄色的树叶随风飘落在石板路上'")
        input_section_parts.append("")
        input_section_parts.append("【融合示例3 - 动作保持】")
        input_section_parts.append("参考视频: '一只猫在草地上跳跃，追逐蝴蝶'")
        input_section_parts.append("原图: '一只金色的狗'")
        input_section_parts.append("融合结果: '一只金色的狗在草地上跳跃，追逐蝴蝶，动作敏捷，毛发在阳光下闪闪发光'")
        
    elif has_manual and has_optional:
        # ========== 原有模式：手动输入修改原图 ==========
        input_section_parts.append("【输入内容】")
        input_section_parts.append("")
        input_section_parts.append("以下是两部分需要综合的内容：")
        input_section_parts.append("")
        input_section_parts.append("1. 【用户修改指令】（最高优先级，必须严格遵守）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("2. 【原图内容】（需要被修改的基础图片）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【处理要求】")
        input_section_parts.append("请根据【用户修改指令】修改【原图内容】，生成最终的视频提示词。")
        input_section_parts.append("- 以【原图内容】为基础")
        input_section_parts.append("- 应用【用户修改指令】中的所有修改要求")
        input_section_parts.append("- 保持原图中没有被要求修改的部分不变")
        input_section_parts.append("- 当两者冲突时，优先遵循用户修改指令")
        input_section_parts.append("")
        input_section_parts.append("示例：")
        input_section_parts.append("用户修改指令: \"把奔跑改成跳跃\"")
        input_section_parts.append("原图内容: \"一个女孩在草地上奔跑\"")
        input_section_parts.append("最终提示词: \"一个女孩在草地上跳跃\"")
        
    elif has_manual and not has_optional:
        # ========== 只有手工提示词 ==========
        input_section_parts.append("【用户输入】")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【处理要求】")
        input_section_parts.append("请根据用户输入生成视频提示词。")
        
    elif not has_manual and has_optional:
        # ========== 只有图片描述 ==========
        input_section_parts.append("【图片识别内容】")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【处理要求】")
        input_section_parts.append("请根据图片识别内容，生成高质量的视频提示词。")
    
    if not input_section_parts:
        return None
    
    input_section = "\n".join(input_section_parts)
    
    # 根据详细程度定义不同的处理策略
    if detail_level == "标准":
        strategy = """【处理策略：严格遵循原文】
- 只做必要的语法修正和错别字修正
- 补全描述中明显缺失的关键元素（如主体缺失、动作缺失）
- 严禁自行添加原文没有的场景、环境、细节
- 保持原提示词的核心内容和风格不变
- 输出长度与原文相近"""
    elif detail_level == "详细":
        strategy = """【处理策略：适度补充】
- 在原文基础上进行合理的细节补充
- 可补充环境氛围、光线、情绪等与原文相关的内容
- 不添加原文完全未提及的新场景元素
- 动作描述可适当细化，但不改变原意
- 输出长度约为原文的1.5-2倍"""
    else:  # 极详细
        strategy = """【处理策略：全面丰富（电影级）】
- 可充分发挥想象力，对原文进行全方位的丰富和扩展
- 添加电影级的视觉细节：光影、质感、色彩、构图
- 补充完整的场景环境和氛围营造
- 细化动作过程的起承转合
- 可添加镜头语言描述（推拉摇移、景别变化）
- 营造沉浸式的情绪氛围
- 输出长度不限，追求极致细节"""
    
    # 根据融合模式区分
    if is_fusion_mode:
        mode_guide = """【参考图融合模式说明】
你的任务是将【原图】中的关键元素融合到【参考视频】中：
- 以【参考视频】为基础场景，保持其整体结构、动作和运动轨迹
- 从【原图】中提取人物、物体、场景元素
- 用原图的人物替换参考视频中的人物（保持参考视频的动作）
- 将原图的其他元素自然融入参考视频的合适位置
- 输出一个完整、连贯的融合后视频场景描述"""
    elif mode == "文生视频":
        mode_guide = """【文生视频模式说明】
从零构建视觉场景，侧重：
- 完整的场景设定和环境构建
- 主体形象的详细刻画
- 动作的完整过程描述
- 时间流逝和变化过程"""
    else:  # 图生视频
        mode_guide = """【图生视频模式说明】
基于参考图生成视频，侧重：
- 综合【用户手动输入】和【原图识别内容】
- 用户手动输入是修改的方向，原图识别是基础内容
- 保留原图的基本元素，同时应用用户的修改要求
- 最终提示词应该融合两者的内容"""
    
    # 根据输出语言设置
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    # 针对融合模式的额外强调
    extra_emphasis = ""
    if is_fusion_mode:
        extra_emphasis = """
【特别强调 - 参考视频融合】
- 【参考图内容】是基础视频场景，必须保持其整体结构、动作和运动风格
- 从【原图内容】中提取元素并融合到参考视频中
- 如果原图有人物，必须用原图的人物替换参考视频中的人物（保持参考视频的动作）
- 如果原图有特殊物体/道具，将它们融入参考视频的合适位置
- 如果原图有场景元素，将它们融入参考视频的背景中
- 融合后的视频必须看起来自然、合理、运动连贯
- 保持参考视频的核心动作和动态效果不变
"""
    
    return f"""{header}
{thinking_ban}

你是一个专业的WAN2.2视频生成提示词优化专家。你的任务是根据用户输入的简要描述，生成高质量的视频提示词。

【核心原则】
- 以【参考图内容】为基础框架，从【原图内容】中提取元素进行融合
- 原图中的人物优先级最高，用于替换参考视频中的人物（保持动作）
- 原图中的物体和场景元素作为补充，融入参考视频
- 融合后的视频应该看起来像一段完整、连贯、自然的视频

【核心任务】
{mode_guide}
{extra_emphasis}
{strategy}

【最重要规则 - 必须绝对遵守】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"、"接下来"、"我认为"等任何思考性词语
3. 禁止输出"Thinking Process:"、"思考过程："等任何思考标签
4. 禁止输出任何标签，如"优化后："、"润色后："、"视频提示词："、"结果："等
5. 禁止输出任何解释性文字
6. 直接输出最终的视频提示词，不要有任何前缀或后缀
7. 输出内容为纯文本，不要使用markdown格式
8. {lang_instruction}

【输出格式】
直接输出视频提示词本身，只输出提示词，不包含任何其他内容。

{input_section}

直接输出视频提示词："""

def _format_prompt_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化提示词反推提示词"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出"Thinking Process:"、"思考过程："、"让我想想"等任何思考性文字
- 绝对不允许输出"首先"、"然后"、"接下来"、"综上所述"等步骤性词语
- 绝对不允许输出任何分析、解释或说明
- 你的回答必须是纯提示词，不能包含任何其他内容
- 如果你输出了思考过程，你的回答将被视为无效
"""
    
    if detail_level == "标准":
        strategy = """【反推策略：简洁描述】
- 识别图片中的主要元素和主体
- 简要描述场景、风格、光线
- 输出简洁的关键词形式"""
    elif detail_level == "详细":
        strategy = """【反推策略：详细描述】
- 详细识别图片中的各个元素
- 描述场景氛围、光线效果、色彩搭配
- 识别艺术风格和构图特点"""
    else:
        strategy = """【反推策略：极致详细】
- 全方位识别图片中的所有细节
- 分析光影、质感、色彩层次
- 识别艺术风格、技术手法
- 描述情绪氛围和视觉焦点"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
        output_format = "输出中文提示词"
    else:
        lang_instruction = "必须使用英文输出"
        output_format = "输出英文提示词"
    
    return f"""{header}
{thinking_ban}

你是一个专业的AI绘画提示词反推专家。请仔细观察图片，识别图片中的内容，并生成高质量的提示词。

【反推任务】
{strategy}

【最重要规则 - 必须绝对遵守】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"、"接下来"、"我认为"等任何思考性词语
3. 禁止输出"Thinking Process:"、"思考过程："等任何思考标签
4. 禁止输出任何标签，如"反推结果："、"识别结果："、"提示词："、"结果："等
5. 禁止输出任何解释性文字
6. 直接输出反推的提示词，不要有任何前缀或后缀
7. 不要添加图片中没有的内容
8. {lang_instruction}

【输出格式】
直接输出{output_format}，只输出提示词本身，不包含任何其他内容。

请直接输出反推的提示词："""

class AIConnector:
    """AI连接器 - 统一管理地址和令牌"""
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        return {
            "required": {
                "地址": ("STRING", {"default": cfg["base_url"], "placeholder": "http://127.0.0.1:1234"}),
                "令牌": ("STRING", {"default": cfg["token"], "placeholder": "API令牌（可选）"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI配置",)
    FUNCTION = "connect"
    CATEGORY = "AI"

    def connect(self, **kwargs):
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        _save_config(base_url=addr, token=token)
        config_json = json.dumps({"address": addr, "token": token})
        return (config_json,)

class AIPromptInterrogator:
    """AI提示词反推节点 - 使用多模态LLM识别图片内容并生成提示词"""
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "图片": ("IMAGE",),
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入多模态模型名称"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["英文", "中文"], {"default": "英文"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("反推提示词",)
    FUNCTION = "interrogate"
    CATEGORY = "AI"

    def interrogate(self, **kwargs):
        image_tensor = kwargs["图片"]
        config_json = kwargs["AI配置"]
        model = kwargs["模型"].strip()
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
        except:
            return ("AI配置格式错误，请重新连接AI连接器",)
        
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入多模态模型名称",)
        
        try:
            if len(image_tensor.shape) == 4:
                img_tensor = image_tensor[0]
            else:
                img_tensor = image_tensor
            
            img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype('uint8'))
            
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            req = _format_prompt_interrogation(img_base64, detail_level, output_lang)
            
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"], img_base64)
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(反推的提示词:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|Thinking Process:)[\s:]*', '', res, flags=re.I)
            
            if '\n' in res:
                lines = [line.strip() for line in res.split('\n') if line.strip()]
                if lines:
                    res = lines[0]
            
            return (res,)
        except Exception as e:
            return (f"提示词反推错误: {str(e)}",)

class AIImagePromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "图片提示词": ("STRING", {"multiline": True, "placeholder": "输入图片提示词..."}),
                "图片反推描述（可选）": ("STRING", {"multiline": True, "placeholder": "可选，输入图片反推描述或手动填写"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
                "生成模式": (["文生图", "图生图"], {"default": "文生图"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "提示类型": (["正向", "负向"], {"default": "正向"}),
                "输出语言": (["英文", "中文"], {"default": "英文"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("图片提示词",)
    FUNCTION = "convert_image"
    CATEGORY = "AI"

    def convert_image(self, **kwargs):
        config_json = kwargs["AI配置"]
        manual_prompt = kwargs["图片提示词"].strip()
        optional_prompt = kwargs.get("图片反推描述（可选）", "").strip()
        model = kwargs["模型"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        prompt_type = kwargs["提示类型"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
        except:
            return ("AI配置格式错误，请重新连接AI连接器",)
        
        if not manual_prompt and not optional_prompt:
            return ("请输入图片提示词或填写图片反推描述（可选）",)
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入模型名称",)
        
        req = _format_image_prompt(manual_prompt, optional_prompt, mode, detail_level, prompt_type, output_lang)
        key = (manual_prompt, optional_prompt, model, mode, detail_level, prompt_type, output_lang)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(转换后的提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|Thinking Process:)[\s:]*', '', res, flags=re.I)
            
            if '\n' in res:
                lines = [line.strip() for line in res.split('\n') if line.strip()]
                if lines:
                    res = lines[0]
            
            _CACHE[key] = res
            return (res,)
        except Exception as e:
            return (str(e),)

class AIVideoPromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "视频提示词": ("STRING", {"multiline": True, "placeholder": "输入视频提示词..."}),
                "图片反推描述（可选）": ("STRING", {"multiline": True, "placeholder": "可选，输入图片反推描述或手动填写"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
                "生成模式": (["文生视频", "图生视频"], {"default": "文生视频"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["中文", "英文"], {"default": "中文"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频提示词",)
    FUNCTION = "convert_video"
    CATEGORY = "AI"

    def convert_video(self, **kwargs):
        config_json = kwargs["AI配置"]
        manual_prompt = kwargs["视频提示词"].strip()
        optional_prompt = kwargs.get("图片反推描述（可选）", "").strip()
        model = kwargs["模型"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
        except:
            return ("AI配置格式错误，请重新连接AI连接器",)
        
        if not manual_prompt and not optional_prompt:
            return ("请输入视频提示词或填写图片反推描述（可选）",)
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入模型名称",)
        
        video_req = _format_video_prompt(manual_prompt, optional_prompt, mode, detail_level, output_lang)
        
        key = (manual_prompt, optional_prompt, model, mode, detail_level, output_lang)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _ai_chat(addr, token, model, video_req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|Thinking Process:)[\s:]*', '', res, flags=re.I)
            
            if '\n' in res:
                lines = [line.strip() for line in res.split('\n') if line.strip() and not line.startswith(('```', '`'))]
                if lines:
                    res = lines[0]
            
            _CACHE[key] = res
            return (res,)
        except Exception as e:
            return (f"错误: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "AIConnector": AIConnector,
    "AIPromptInterrogator": AIPromptInterrogator,
    "AIImagePromptConverter": AIImagePromptConverter,
    "AIVideoPromptConverter": AIVideoPromptConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIConnector": "🤖 AI 连接器",
    "AIPromptInterrogator": "🔍 AI 提示词反推",
    "AIImagePromptConverter": "🎨 AI 图片提示词",
    "AIVideoPromptConverter": "🎬 AI 视频提示词",
}