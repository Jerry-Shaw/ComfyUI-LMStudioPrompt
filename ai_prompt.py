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
    """清理响应中的思考过程，只保留最终结果 - 保守版本"""
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    skip_current_line = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # 只删除明确的思考标签行
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\]|让我想想|我先|我需要|好的，|好，|明白了|理解了|收到|行，|嗯，|首先|第一步|然后|接着|接下来|之后|最后|综上所述|总结|我认为|我觉得|分析：|思考：|推理：|解读：|Let me think|First|Then|Next|Finally|I think|Analysis:|Reasoning:)\s*$', line_stripped, re.IGNORECASE):
            skip_current_line = True
            continue
        
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\])\s*$', line_stripped, re.IGNORECASE):
            continue
        
        if skip_current_line and not line_stripped:
            skip_current_line = False
            continue
        
        skip_current_line = False
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    if not result:
        return text.strip()
    
    return result

def _ai_chat(url: str, token: str, model: str, msg: str, timeout: int = 60, image_base64: str = None) -> str:
    """通用的AI聊天接口，支持OpenAI兼容的API，支持多模态"""
    base = url.rstrip('/')
    
    final_msg = msg.strip() + "\n\n/no_think"
    
    if image_base64:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_msg},
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
        {"url": f"{base}/api/v1/chat", "payload": {"model": model, "input": final_msg, "temperature": 0.1}},
        {"url": f"{base}/v1/chat/completions", "payload": {"model": model, "messages": [{"role": "user", "content": final_msg}], "temperature": 0.1}}
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
    """格式化图片提示词 - 以原图为基础，根据手工提示词扩写填充"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出任何思考过程、分析或解释
- 直接输出最终提示词，不要有任何前缀、后缀或标签
- 不要输出"好的"、"我理解了"、"根据要求"等开场白
- 只输出提示词本身
"""
    
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    input_section_parts = []
    
    if has_optional and has_manual:
        # ========== 核心模式：以原图为基础，根据手工提示词扩写 ==========
        input_section_parts.append("【任务说明 - 原图扩写模式】")
        input_section_parts.append("")
        input_section_parts.append("【核心原则 - 必须严格遵守】")
        input_section_parts.append("1. 【原图描述】是基础，必须保留其核心视觉元素（主体、场景、风格）")
        input_section_parts.append("2. 【手工提示词】是扩写方向，理解其语义意图，在原图基础上进行扩展")
        input_section_parts.append("3. 不要替换原图的主体或核心场景，只能做【添加】、【扩展】或【修改属性】")
        input_section_parts.append("4. 扩展的内容必须与原图的风格、世界观保持一致")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】（作为基础，必须保留核心内容）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】（理解意图，在原图基础上扩写）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【扩写规则】")
        input_section_parts.append("1. 原图描述的画面是【核心】和【起点】，在此基础上进行扩写")
        input_section_parts.append("2. 扩写类型：")
        input_section_parts.append("   - 空间延伸：展现更广阔的环境（背景延伸、周边环境、远景细节）")
        input_section_parts.append("   - 细节丰富：补充原图中隐含但未描述的细节（纹理、光影、质感）")
        input_section_parts.append("   - 元素添加：在原图场景中添加新元素（人物、物体、动物、植物）")
        input_section_parts.append("   - 氛围调整：调整光线、色调、天气来营造特定氛围")
        input_section_parts.append("   - 动作/状态：修改或添加角色的动作、表情、姿态")
        input_section_parts.append("   - 时间推进：表现时间变化（日出日落、季节更替）")
        input_section_parts.append("3. 扩写的内容必须与原图风格、光影、色调保持一致")
        input_section_parts.append("4. 不要删除或替换原图的核心主体和主要场景")
        input_section_parts.append("5. 如果手工提示词要求完全改变场景，优先保留原图，只做最小必要修改")
        input_section_parts.append("")
        input_section_parts.append("【图片构图指导】")
        input_section_parts.append("根据扩写需要，可以调整或指定：")
        input_section_parts.append("- 景别：特写、近景、中景、全景、远景")
        input_section_parts.append("- 角度：平视、俯视、仰视、鸟瞰、低角度")
        input_section_parts.append("- 构图：中心构图、三分法、引导线、框架构图、对称构图")
        input_section_parts.append("- 光线：顺光、侧光、逆光、轮廓光、柔光、硬光")
        input_section_parts.append("- 色调：暖色调、冷色调、对比色、单色调")
        input_section_parts.append("")
        input_section_parts.append("【扩写示例1 - 空间延伸】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上，阳光明媚'")
        input_section_parts.append("手工提示词: '展现海滩的广阔'")
        input_section_parts.append("正确输出: '一个穿红裙的女孩站在广阔的沙滩上，阳光明媚，金色的沙滩向远方延伸，左边是礁石群，右边是椰林，海浪一层层拍打岸边，天空飘着几朵白云，远处海天相接'")
        input_section_parts.append("（✅ 保留了原图主体，扩展了海滩环境）")
        input_section_parts.append("")
        input_section_parts.append("【扩写示例2 - 元素添加】")
        input_section_parts.append("原图描述: '一片宁静的森林，阳光透过树叶洒下斑驳光影'")
        input_section_parts.append("手工提示词: '奇幻冒险风格，添加一个冒险者'")
        input_section_parts.append("正确输出: '一片神秘的森林中，阳光透过茂密的树叶洒下斑驳光影，一个背着行囊的冒险者站在林间小路上，环顾四周。远处古树参天，藤蔓缠绕，树根盘错，偶尔有萤火虫般的光芒在林间飘动，空气中弥漫着魔法的气息'")
        input_section_parts.append("（✅ 保留了原图森林场景，添加了冒险者和奇幻元素）")
        input_section_parts.append("")
        input_section_parts.append("【扩写示例3 - 属性修改】")
        input_section_parts.append("原图描述: '一个女孩在草地上奔跑，穿着白色连衣裙'")
        input_section_parts.append("手工提示词: '把裙子改成红色，让她更快乐'")
        input_section_parts.append("正确输出: '一个女孩在草地上快乐地奔跑，穿着鲜艳的红色连衣裙，脸上洋溢着灿烂的笑容，裙摆随风飘动，阳光洒在她身上'")
        input_section_parts.append("（✅ 修改了裙子颜色和表情，保留了原图动作和场景）")
        input_section_parts.append("")
        input_section_parts.append("【错误示例 - 禁止这样做】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上'")
        input_section_parts.append("手工提示词: '让她跑起来'")
        input_section_parts.append("错误输出: '一个运动员在田径场上奔跑'")
        input_section_parts.append("（❌ 完全改变了场景，禁止！）")
        input_section_parts.append("")
        input_section_parts.append("【你的任务】")
        input_section_parts.append("请理解手工提示词的意图，在原图描述的基础上进行合理的扩写，输出一个完整的、高质量的图片提示词。")
        
    elif has_optional and not has_manual:
        # ========== 只有原图描述：优化模式 ==========
        input_section_parts.append("【任务说明 - 优化模式】")
        input_section_parts.append("")
        input_section_parts.append("请基于以下原图描述，优化成一个更高质量的图片提示词，不要改变核心内容。")
        input_section_parts.append("可以适当丰富细节、补充光影和质感描述。")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】：")
        input_section_parts.append(optional_text.strip())
        
    elif not has_optional and has_manual:
        # ========== 只有手工提示词：从零生成模式 ==========
        input_section_parts.append("【任务说明 - 从零生成模式】")
        input_section_parts.append("")
        input_section_parts.append("请根据以下手工提示词，直接生成一个高质量的图片提示词。")
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】：")
        input_section_parts.append(manual_text.strip())
    
    if not input_section_parts:
        return None
    
    input_section = "\n".join(input_section_parts)
    
    # 详细程度策略
    if detail_level == "标准":
        strategy = """【详细程度：标准】
- 简洁明了，突出重点
- 只描述核心元素"""
    elif detail_level == "详细":
        strategy = """【详细程度：详细】
- 适当补充环境、光线、质感
- 描述更加丰富"""
    else:
        strategy = """【详细程度：极详细】
- 电影级详细描述
- 包含光影、色彩、构图、氛围
- 尽可能丰富"""
    
    # ========== 修复：正确使用 prompt_type ==========
    if prompt_type == "正向":
        type_instruction = """【提示词类型：正向提示词】
生成想要呈现的内容，包括：
- 主体描述
- 场景环境
- 风格设定
- 光影构图
- 氛围情绪"""
    else:
        type_instruction = """【提示词类型：负向提示词】
生成需要避免的内容，包括：
- 低质量元素（模糊、噪点、锯齿）
- 畸形特征（畸形的手、多余的手指、扭曲的脸）
- 不想要的内容（水印、文字、遮挡物）
- 负面属性（丑陋、恐怖、血腥）"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    # 根据生成模式添加额外指导
    mode_instruction = ""
    if mode == "文生图":
        mode_instruction = "【生成模式：文生图】从零构建画面，侧重完整的场景设定和元素构建。"
    else:
        mode_instruction = "【生成模式：图生图】基于参考图进行重绘，保持与原图的基本结构一致性。"
    
    return f"""{header}
{thinking_ban}

你是一个专业的AI绘画提示词专家。

{strategy}

{type_instruction}

{mode_instruction}

{input_section}

【输出要求】
1. 直接输出提示词，不要有任何其他内容
2. 不要输出思考过程、分析、解释
3. 【最高优先级】如果提供了原图描述，必须保留其核心视觉元素
4. 根据手工提示词的意图进行扩写，而不是替换
5. {lang_instruction}

直接输出提示词："""


def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词 - 以原图为基础，根据手工提示词扩写填充，支持镜头语言和转场"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出任何思考过程、分析或解释
- 直接输出最终提示词，不要有任何前缀、后缀或标签
- 不要输出"好的"、"我理解了"、"根据要求"等开场白
- 只输出提示词本身
"""
    
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    input_section_parts = []
    
    if has_optional and has_manual:
        # ========== 核心模式：以原图为基础，根据手工提示词扩写视频 ==========
        input_section_parts.append("【任务说明 - 原图扩写模式（视频）】")
        input_section_parts.append("")
        input_section_parts.append("【核心原则 - 必须严格遵守】")
        input_section_parts.append("1. 【原图描述】是视频首帧的基础，必须保留其核心视觉元素")
        input_section_parts.append("2. 【手工提示词】是扩写方向，理解其语义意图，在原图基础上进行扩展")
        input_section_parts.append("3. 不要替换原图的主体或核心场景，只能做【添加】、【扩展】或【修改属性】")
        input_section_parts.append("4. 扩展的内容必须与原图的风格、世界观保持一致")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】（视频首帧以此为基础，必须保留核心内容）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】（理解意图，在原图基础上扩写视频）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【视频扩写规则】")
        input_section_parts.append("1. 原图描述的画面是【起点】，在此基础上演化出视频内容")
        input_section_parts.append("2. 扩写类型：")
        input_section_parts.append("   - 空间延伸：镜头拉远/摇移，展现更广阔的环境")
        input_section_parts.append("   - 时间推进：日出日落、天气变化、季节更替")
        input_section_parts.append("   - 情节发展：人物移动、新角色出现、事件发生")
        input_section_parts.append("   - 视角切换：从A角色切换到B角色的视角")
        input_section_parts.append("   - 细节丰富：补充原图中隐含但未描述的细节")
        input_section_parts.append("   - 动作/状态：添加或修改角色的动作、表情")
        input_section_parts.append("3. 扩写的内容必须与原图风格保持一致")
        input_section_parts.append("4. 不要完全替换原图的核心主体和主要场景")
        input_section_parts.append("")
        input_section_parts.append("【镜头语言指导】")
        input_section_parts.append("根据扩写需要，可以设计以下镜头：")
        input_section_parts.append("")
        input_section_parts.append("【景别变化】")
        input_section_parts.append("- 特写（Close-up）：聚焦面部表情或细节，强化情感")
        input_section_parts.append("- 近景（Medium Close-up）：胸部以上，展现表情和部分环境")
        input_section_parts.append("- 中景（Medium Shot）：腰部以上，展现动作和互动")
        input_section_parts.append("- 全景（Full Shot）：全身入画，展现完整动作")
        input_section_parts.append("- 远景（Long Shot）：人物在环境中较小，强调环境")
        input_section_parts.append("")
        input_section_parts.append("【镜头运动】")
        input_section_parts.append("- 推镜头（Push In）：聚焦细节，强调重点")
        input_section_parts.append("- 拉镜头（Pull Out）：展现环境，交代空间关系")
        input_section_parts.append("- 摇镜头（Pan）：水平扫视，展现场景全貌")
        input_section_parts.append("- 跟镜头（Follow）：追随主体运动")
        input_section_parts.append("- 环绕镜头（Orbit）：围绕主体旋转，全方位展示")
        input_section_parts.append("")
        input_section_parts.append("【视角选择】")
        input_section_parts.append("- 主观视角（POV）：从角色眼中看世界")
        input_section_parts.append("- 客观视角：旁观者视角，叙事性")
        input_section_parts.append("- 越肩视角：对话场景，展现互动")
        input_section_parts.append("- 低角度：强调力量和威严")
        input_section_parts.append("- 高角度：强调脆弱和渺小")
        input_section_parts.append("")
        input_section_parts.append("【转场效果指导】")
        input_section_parts.append("根据剧情节奏、时间跨度、空间变化，可以设计以下转场：")
        input_section_parts.append("- 硬切（Hard Cut）：直接切换，连贯动作")
        input_section_parts.append("- 淡入淡出（Fade）：时间跨越、开场结尾")
        input_section_parts.append("- 溶解（Dissolve）：时间流逝、回忆梦幻")
        input_section_parts.append("- 左右抽出（Wipe）：空间转换")
        input_section_parts.append("- 闪光转场（Flash）：紧张节奏")
        input_section_parts.append("")
        input_section_parts.append("【扩写示例1 - 空间延伸】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上，阳光明媚'")
        input_section_parts.append("手工提示词: '展现海滩的广阔，让她跑起来'")
        input_section_parts.append("正确输出: '【首帧】中景，一个穿红裙的女孩站在沙滩上面朝大海。【镜头运动】镜头开始向后拉远，从近景变为全景。【动作】女孩开始向前奔跑，镜头切换到跟拍，跟随她的侧脸移动。【场景扩展】随着她的奔跑，镜头慢慢拉远，展现更广阔的海滩：金色的沙滩延伸到远方，左边是礁石群，右边是椰林，海浪一层层拍打。【结尾】远景俯瞰整个海湾的壮丽景色。'")
        input_section_parts.append("（✅ 首帧忠于原图，扩展了海滩全景）")
        input_section_parts.append("")
        input_section_parts.append("【扩写示例2 - 元素添加】")
        input_section_parts.append("原图描述: '一片宁静的森林，阳光透过树叶洒下斑驳光影'")
        input_section_parts.append("手工提示词: '奇幻冒险风格，添加一个冒险者，让他探索森林'")
        input_section_parts.append("正确输出: '【首帧】森林地面的特写，阳光透过树叶洒下斑驳光影，一只脚踏过青苔石头。【镜头运动】镜头拉起，展现一个背着行囊的冒险者中景，他环顾四周，眼神坚定。【动作】他沿着林间小路前行，镜头切换到跟拍，跟随他的脚步移动。【场景扩展】随着他的前进，镜头慢慢拉远，展现更广阔的森林：古树参天，藤蔓缠绕，偶尔有萤火虫般的光芒飘动。【转场】硬切，冒险者拨开树枝。【视角切换】越肩视角，前方出现一座古老的石桥，桥的另一端通向迷雾深处。'")
        input_section_parts.append("（✅ 首帧忠于原图，添加了冒险者和奇幻元素）")
        input_section_parts.append("")
        input_section_parts.append("【错误示例 - 禁止】")
        input_section_parts.append("原图描述: '一个女孩在公园里散步'")
        input_section_parts.append("手工提示词: '让她跑起来'")
        input_section_parts.append("错误输出: '一个运动员在田径场上奔跑'")
        input_section_parts.append("（❌ 首帧完全改变了场景，禁止！）")
        input_section_parts.append("")
        input_section_parts.append("【你的任务】")
        input_section_parts.append("请理解手工提示词的意图，在原图描述的基础上进行合理的视频扩写，输出一个完整的、高质量的视频提示词。")
        
    elif has_optional and not has_manual:
        # ========== 只有原图描述：优化模式 ==========
        input_section_parts.append("【任务说明 - 优化模式（视频）】")
        input_section_parts.append("")
        input_section_parts.append("请基于以下原图描述，优化成一个更高质量的视频提示词。")
        input_section_parts.append("视频首帧必须与原图描述一致，可以添加合理的镜头运动和环境扩展。")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】：")
        input_section_parts.append(optional_text.strip())
        
    elif not has_optional and has_manual:
        # ========== 只有手工提示词：从零生成模式 ==========
        input_section_parts.append("【任务说明 - 从零生成模式（视频）】")
        input_section_parts.append("")
        input_section_parts.append("请根据以下手工提示词，直接生成一个高质量的视频提示词。")
        input_section_parts.append("可以自由设计场景、镜头运动和转场效果。")
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【镜头和转场设计建议】")
        input_section_parts.append("根据内容需要，可以设计：")
        input_section_parts.append("- 景别变化：特写、近景、中景、全景、远景")
        input_section_parts.append("- 镜头运动：推、拉、摇、移、跟、升降、环绕")
        input_section_parts.append("- 视角选择：主观、客观、越肩、鸟瞰")
        input_section_parts.append("- 转场效果：硬切、淡入淡出、溶解、左右抽出、闪光等")
    
    if not input_section_parts:
        return None
    
    input_section = "\n".join(input_section_parts)
    
    if detail_level == "标准":
        strategy = """【详细程度：标准】
- 简洁明了，突出重点动作
- 简单描述镜头变化和转场"""
    elif detail_level == "详细":
        strategy = """【详细程度：详细】
- 包含动作过程、镜头运动、环境氛围、转场效果
- 设计2-3个镜头变化和1-2个转场"""
    else:
        strategy = """【详细程度：极详细】
- 电影级详细描述
- 包含完整的镜头语言设计（景别、运动、视角）
- 包含完整的转场效果设计
- 包含动作细节、光影变化、情绪氛围
- 设计完整的镜头序列和转场序列"""
    
    # 根据生成模式添加指导
    if mode == "文生视频":
        mode_instruction = "【生成模式：文生视频】从零构建视频场景，侧重完整的场景设定和动作过程。"
    else:
        mode_instruction = "【生成模式：图生视频】基于参考图生成视频，首帧必须与原图一致。"
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    return f"""{header}
{thinking_ban}

你是一个专业的WAN2.2视频提示词专家。

{strategy}

{mode_instruction}

{input_section}

【输出要求】
1. 直接输出视频提示词，不要有任何其他内容
2. 不要输出思考过程、分析、解释
3. 【最高优先级】如果提供了原图描述，视频首帧必须保留其核心视觉元素
4. 根据手工提示词的意图进行扩写，而不是替换
5. 根据剧情需要设计合适的镜头运动、景别变化和转场效果
6. {lang_instruction}

直接输出视频提示词："""


def _format_prompt_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化提示词反推提示词"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出任何思考过程、分析或解释
- 直接输出反推的提示词，不要有任何前缀、后缀或标签
- 只输出提示词本身
"""
    
    if detail_level == "标准":
        strategy = """简洁描述主要元素和场景"""
    elif detail_level == "详细":
        strategy = """详细描述元素、光线、色彩、风格"""
    else:
        strategy = """极致详细地描述所有视觉元素，包括光影、质感、构图、氛围"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    return f"""{header}
{thinking_ban}

你是一个专业的AI绘画提示词反推专家。请仔细观察图片，识别图片中的内容，并生成高质量的提示词。

【反推要求】
{strategy}

【输出要求】
1. 直接输出反推的提示词，不要有任何其他内容
2. 不要输出思考过程、分析、解释
3. 不要添加图片中没有的内容
4. {lang_instruction}

直接输出提示词："""


# ========== 节点类 ==========

class AIConnector:
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
            
            res = re.sub(r'^(反推的提示词:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            if not res or len(res) < 5:
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
                "原图描述": ("STRING", {"multiline": True, "placeholder": "输入原图反推描述（作为基础，必须保留核心内容）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入你的需求/意图（在原图基础上扩写）"}),
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
        optional_prompt = kwargs["原图描述"].strip()  # 原图描述作为基础
        manual_prompt = kwargs["手工提示词"].strip()  # 手工提示词作为扩写方向
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
        
        if not optional_prompt and not manual_prompt:
            return ("请填写原图描述或手工提示词",)
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
            
            res = re.sub(r'^(转换后的提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            if not res or len(res) < 5:
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
                "原图描述": ("STRING", {"multiline": True, "placeholder": "输入原图反推描述（视频首帧以此为基础）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入你的需求/意图（在原图基础上扩写）"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
                "生成模式": (["文生视频", "图生视频"], {"default": "图生视频"}),
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
        optional_prompt = kwargs["原图描述"].strip()  # 原图描述作为基础
        manual_prompt = kwargs["手工提示词"].strip()  # 手工提示词作为扩写方向
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
        
        if not optional_prompt and not manual_prompt:
            return ("请填写原图描述或手工提示词",)
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
            
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            if not res or len(res) < 10:
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