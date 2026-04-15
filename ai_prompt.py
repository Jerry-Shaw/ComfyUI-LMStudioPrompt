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
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\]|让我想想|我先|我需要|好的，|好，|明白了|理解了|收到|行，|嗯，|首先|第一步|然后|接着|接下来|之后|最后|综上所述|总结|我认为|我觉得|分析：|思考：|推理：|解读：|Let me think|First|Then|Next|Finally|I think|Analysis:|Reasoning:)', line_stripped, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    if not result:
        return text.strip()
    
    return result

def _parse_marked_output(text: str) -> dict:
    """解析带标记的输出格式，返回 {positive, negative, description}"""
    result = {
        "positive": "",
        "negative": "",
        "description": ""
    }
    
    if not text:
        return result
    
    positive_match = re.search(r'\[POSITIVE\](.*?)(?=\[NEGATIVE\]|$)', text, re.DOTALL)
    negative_match = re.search(r'\[NEGATIVE\](.*?)(?=\[DESCRIPTION\]|$)', text, re.DOTALL)
    description_match = re.search(r'\[DESCRIPTION\](.*?)$', text, re.DOTALL)
    
    if positive_match:
        result["positive"] = positive_match.group(1).strip()
    if negative_match:
        result["negative"] = negative_match.group(1).strip()
    if description_match:
        result["description"] = description_match.group(1).strip()
    
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


def _format_image_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化图片提示词 - 同时输出正向和负向提示词"""
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    if output_lang == "中文":
        lang_inst = "使用中文输出"
    else:
        lang_inst = "use English output"
    
    lines = []
    lines.append("【指令】直接输出正向和负向提示词，不要有任何思考过程、分析或解释。")
    lines.append("严禁使用任何Markdown格式（不要使用**粗体**、*斜体*、`代码块`等）")
    lines.append("")
    lines.append(f"你是一个专业的AI绘画提示词专家。{lang_inst}。")
    lines.append("")
    
    # 提示词公式模板
    if mode == "文生图":
        lines.append("【提示词公式】主体 + 场景 + 风格 + 光线 + 构图 + 质量")
        lines.append("- 主体：核心对象，详细描述外观、姿态、表情")
        lines.append("- 场景：环境背景，地点、氛围、细节")
        lines.append("- 风格：艺术风格，写实/动漫/油画/赛博朋克等")
        lines.append("- 光线：光照类型，自然光/柔光/逆光/电影光等")
        lines.append("- 构图：镜头语言，景别/角度/视角")
        lines.append("- 质量：画质参数，8K/高细节/锐利焦点等")
    else:
        lines.append("【提示词公式】图生图：保留原图核心 + 描述修改方向")
        lines.append("- 原图基础：保留主体和场景的核心特征")
        lines.append("- 修改方向：明确要改变的属性（颜色/动作/风格等）")
        lines.append("- 融合要求：新旧元素自然融合，保持风格一致")
    lines.append("")
    
    # 知识库 - 根据语言选择版本
    if output_lang == "中文":
        lines.append("【关键词知识库】")
        lines.append("质量: 杰作, 最佳质量, 高质量, 高细节, 极致细节, 8K, 4K")
        lines.append("风格: 照片级写实, 电影感, 数字艺术, 油画, 水彩, 动漫风格, 赛博朋克")
        lines.append("光线: 电影级照明, 柔光, 自然光, 黄金时刻, 工作室灯光, 轮廓光")
        lines.append("构图: 特写, 广角, 低角度, 高角度, 景深, 焦外虚化, 中心构图")
    else:
        lines.append("【关键词知识库】")
        lines.append("quality: masterpiece, best quality, high quality, highly detailed, ultra-detailed, 8K, 4K")
        lines.append("style: photorealistic, cinematic, digital art, oil painting, watercolor, anime style, cyberpunk")
        lines.append("lighting: cinematic lighting, soft lighting, natural lighting, golden hour, studio lighting, rim lighting")
        lines.append("composition: close-up, wide shot, low angle, high angle, depth of field, bokeh, centered composition")
    lines.append("")
    
    # 详细程度
    if detail_level == "标准":
        lines.append("【详细程度】简洁明了，每个类别1-2个关键词，总词数30以内。")
    elif detail_level == "详细":
        lines.append("【详细程度】适当丰富，每个类别2-4个关键词，总词数30-60。")
    else:
        lines.append("【详细程度】极致详细，每个类别4-6个关键词，总词数60-100。")
    lines.append("")
    
    # 输出格式
    lines.append("【输出格式 - 必须严格遵守】")
    lines.append("[POSITIVE]正向提示词（多个关键词用英文逗号分隔）")
    lines.append("[NEGATIVE]负向提示词（多个关键词用英文逗号分隔）")
    lines.append("")
    lines.append("【格式示例】")
    if output_lang == "中文":
        lines.append("[POSITIVE]杰作, 最佳质量, 一个女孩, 红色连衣裙, 沙滩, 日落, 逆光, 柔光, 中景, 8K")
        lines.append("[NEGATIVE]模糊, 低质量, 畸形的手, 多余的手指, 水印, 文字")
    else:
        lines.append("[POSITIVE]masterpiece, best quality, a girl, red dress, beach, sunset, backlighting, soft lighting, medium shot, 8K")
        lines.append("[NEGATIVE]blurry, low quality, bad anatomy, extra fingers, watermark, text")
    lines.append("")
    
    # 用户输入
    if has_optional and has_manual:
        lines.append("【任务】综合以下内容，创作高质量的正向和负向提示词。")
        lines.append("")
        lines.append("【原图描述】（参考内容，了解画面基础）：")
        lines.append(optional_text.strip())
        lines.append("")
        lines.append("【手工提示词】（创作方向，请全面理解）：")
        lines.append(manual_text.strip())
        lines.append("")
        lines.append("【创作要求】")
        lines.append("- 如果两者都有，需融合两者元素，手工提示词优先级更高")
        lines.append("- 按提示词公式组织正向提示词结构")
        lines.append("- 负向提示词要基于对原图内容和手工提示词的理解，排除画面中不应出现的元素")
        lines.append("- 负向提示词包括：画质问题、结构问题、与原图/手工提示词冲突的元素、不想要的内容")
    elif has_optional and not has_manual:
        lines.append("【任务】基于以下原图描述，创作高质量的正向和负向提示词。")
        lines.append("")
        lines.append("【原图描述】：")
        lines.append(optional_text.strip())
        lines.append("")
        lines.append("【创作要求】")
        lines.append("- 保留原图核心内容")
        lines.append("- 补充画质、光线、构图关键词到正向提示词")
        lines.append("- 负向提示词要基于原图内容，排除可能出现的画质问题和结构问题")
    elif not has_optional and has_manual:
        lines.append("【任务】根据以下手工提示词，创作高质量的正向和负向提示词。")
        lines.append("")
        lines.append("【手工提示词】：")
        lines.append(manual_text.strip())
        lines.append("")
        lines.append("【创作要求】")
        lines.append("- 按提示词公式完整构建正向提示词")
        lines.append("- 包含画质、风格、光线、构图关键词")
        lines.append("- 负向提示词要基于手工提示词的内容，排除不想要或冲突的元素")
    
    if not lines:
        return None
    
    lines.append("")
    lines.append(f"直接输出（{lang_inst}）：")
    
    return "\n".join(lines)


def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词 - 基于文章方法论，强化叙事逻辑"""
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    if output_lang == "中文":
        lang_inst = "使用中文输出"
        mode_text = "文生视频" if mode == "文生视频" else "图生视频"
        
        # 基于文章的美学控制知识库
        knowledge = """【视觉元素知识库 - 按需选用】

光源类型: 日光、人工光、月光、实用光、火光、荧光、阴天光、混合光、晴天光
光线类型: 柔光、硬光、高对比度、侧光、底光、低对比度、边缘光、剪影、背光、逆光
时间段: 白天、夜晚、日出、日落、黎明、黄昏、黄金时刻
景别: 特写、近景、中景、中近景、全景、远景、广角
构图: 中心构图、平衡构图、左侧构图、右侧构图、对称构图、短边构图
镜头焦段: 广角、中焦距、长焦、鱼眼
镜头运动: 固定、推进、拉远、上摇、下摇、左移、右移、手持、跟随、环绕
角色情绪: 愤怒、恐惧、高兴、悲伤、惊讶、沉思、平静、焦虑
视觉风格: 写实、电影感、纪录片、像素、3D、二次元、印象派、油画
特效: 慢动作、动态模糊、镜头光晕、移轴、延时"""
    else:
        lang_inst = "use English output"
        mode_text = "text-to-video" if mode == "文生视频" else "image-to-video"
        
        knowledge = """【Visual Elements Knowledge Base】

Light Source: daylight, artificial light, moonlight, practical light, firelight, fluorescent, overcast, mixed light, sunlight
Lighting: soft lighting, hard lighting, high contrast, side lighting, underlighting, low contrast, rim lighting, silhouette, backlighting
Time: daytime, night, sunrise, sunset, dawn, dusk, golden hour
Shot Size: close-up, medium close-up, medium shot, medium wide shot, wide shot, extreme wide shot
Composition: center composition, balanced composition, left-heavy, right-heavy, symmetrical, short-side
Lens: wide-angle, medium lens, telephoto, fisheye
Camera Movement: static, push-in, pull-back, pan, tilt, tracking, handheld, follow, orbit
Emotion: angry, fearful, happy, sad, surprised, pensive, calm, anxious
Style: photorealistic, cinematic, documentary, pixel, 3D, anime, impressionist, oil painting
Effects: slow motion, motion blur, lens flare, tilt-shift, time-lapse"""
    
    lines = []
    lines.append("【指令】直接输出正向和负向提示词，不要有任何思考过程、分析或解释。")
    lines.append("严禁使用任何Markdown格式。")
    lines.append("")
    lines.append(f"你是一个专业的视频提示词专家。{lang_inst}，{mode_text}模式。")
    lines.append("")
    
    # 核心方法论（基于文章）
    lines.append("【核心方法论】")
    lines.append("提示词 = 关键词组合 + 详细场景描述")
    lines.append("")
    lines.append("关键词组合：从知识库中选择10-20个关键词，用英文逗号分隔，服务于统一的艺术目标")
    lines.append("")
    lines.append("详细场景描述结构：")
    lines.append("【首帧】描述画面起始状态")
    lines.append("【镜头运动】描述镜头如何运动")
    lines.append("【动作】按时间顺序描述动作序列（用'先...然后...接着...最后...'连接）")
    lines.append("【场景扩展】描述环境变化和细节")
    lines.append("【转场】描述如何过渡到下一段（如需要）")
    lines.append("")
    
    # 叙事逻辑强化
    lines.append("【叙事逻辑规则 - 必须遵守】")
    lines.append("")
    lines.append("1. 时间顺序：动作必须按发生先后描述")
    lines.append("   - 使用：'先'、'然后'、'接着'、'紧接着'、'之后'、'最后'")
    lines.append("")
    lines.append("2. 因果关系：动作之间的因果必须明确")
    lines.append("   - 使用：'导致'、'使得'、'于是'、'从而'")
    lines.append("")
    lines.append("3. 动作密度：每5秒视频最多包含2-4个主要动作节点")
    lines.append("   - 每个动作节点用1-2句话独立描述")
    lines.append("   - 禁止将多个动作堆叠在一句话里")
    lines.append("")
    lines.append("4. 禁止的错误写法示例：")
    lines.append("   - ❌ '角色奔跑、跳跃、转身、落地挥手'（动作堆叠，无顺序）")
    lines.append("   - ✅ '角色先快速奔跑，然后纵身一跳，接着在空中转身，最后落地向大家挥手'")
    lines.append("")
    
    # 输出格式
    lines.append("【输出格式】")
    lines.append("[POSITIVE]关键词（英文逗号分隔，10-20个）")
    lines.append("[NEGATIVE]负向关键词（英文逗号分隔，8-15个）")
    lines.append("[DESCRIPTION]详细场景描述（按上述结构组织）")
    lines.append("")
    lines.append("【格式示例】")
    if output_lang == "中文":
        lines.append("[POSITIVE]中景, 黄金时刻, 逆光, 暖色调, 跟拍, 柔光, 侧光, 动态模糊, 慢动作, 电影感, 宁静")
        lines.append("[NEGATIVE]模糊, 低质量, 畸形的手, 多余的手指, 水印, 文字, 抖动, 穿模, 动作卡顿")
        lines.append("[DESCRIPTION]【首帧】中景，一个穿白裙的女孩站在沙滩上面朝大海。【镜头运动】镜头向后拉远，从近景变为全景。【动作】女孩先开始奔跑，然后镜头切换到跟拍，跟随她的侧脸移动。【场景扩展】随着她的奔跑，镜头慢慢拉远，展现更广阔的海滩：金色沙滩延伸到远方，左边是礁石群，右边是椰林，海浪一层层拍打。【转场】淡入淡出，暗示时间流逝。")
    else:
        lines.append("[POSITIVE]medium shot, golden hour, backlighting, warm colors, tracking shot, soft lighting, side lighting, motion blur, slow motion, cinematic, serene")
        lines.append("[NEGATIVE]blurry, low quality, bad anatomy, extra fingers, watermark, text, shake, clipping, motion stutter")
        lines.append("[DESCRIPTION][First Frame] Medium shot, a girl in a white dress stands on the beach facing the sea. [Camera Movement] The camera pulls back, transitioning from medium to wide shot. [Action] First the girl starts running, then the camera switches to tracking shot, following her profile. [Scene Expansion] As she runs, the camera slowly pulls back to reveal the vast beach: golden sand stretching into the distance, reefs on the left, palm trees on the right, waves lapping the shore. [Transition] Fade to black, suggesting time passing.")
    lines.append("")
    
    lines.append(knowledge)
    lines.append("")
    
    # 用户输入
    if has_optional and has_manual:
        lines.append("【原图描述】")
        lines.append(optional_text.strip())
        lines.append("")
        lines.append("【手工提示词】")
        lines.append(manual_text.strip())
    elif has_optional and not has_manual:
        lines.append("【原图描述】")
        lines.append(optional_text.strip())
    elif not has_optional and has_manual:
        lines.append("【手工提示词】")
        lines.append(manual_text.strip())
    
    lines.append("")
    lines.append(f"直接输出（{lang_inst}）：")
    
    return "\n".join(lines)


def _format_content_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化内容反推提示词 - 输出内容描述和负向提示词"""
    if output_lang == "中文":
        lang_inst = "使用中文输出"
        if detail_level == "标准":
            strategy = "简洁描述图片中的主要元素和场景"
        elif detail_level == "详细":
            strategy = "详细描述图片中的元素、光线、色彩、风格和氛围"
        else:
            strategy = "极致详细地描述图片中的所有视觉元素，包括光影、质感、构图、色彩层次和情绪氛围"
    else:
        lang_inst = "use English output"
        if detail_level == "标准":
            strategy = "briefly describe the main elements and scene in the image"
        elif detail_level == "详细":
            strategy = "describe in detail the elements, lighting, color, style and atmosphere in the image"
        else:
            strategy = "describe all visual elements in extreme detail, including lighting, texture, composition, color gradation and emotional atmosphere"
    
    return f"""【指令】直接输出内容描述和负向提示词，不要有任何思考过程、分析或解释。
严禁使用任何Markdown格式（不要使用**粗体**、*斜体*、`代码块`等）

你是一个专业的AI视觉内容分析专家。请仔细观察图片，识别并描述图片中的所有内容。

【反推要求】{strategy}

【输出格式 - 必须严格遵守】
[POSITIVE]内容关键词（用英文逗号分隔，8-15个，必须使用{output_lang}）
[NEGATIVE]负向提示词（用英文逗号分隔，8-15个，必须使用{output_lang}，基于图片中不存在的常见问题或可能出现的瑕疵）
[DESCRIPTION]自然语言描述（用完整的句子描述，不要使用提示词格式，保持流畅的自然语言风格）

【格式示例】
{("[POSITIVE]风景, 山脉, 日落, 金色阳光, 云彩, 宁静" if output_lang == "中文" else "[POSITIVE]landscape, mountains, sunset, golden light, clouds, serene")}
{("[NEGATIVE]模糊, 低质量, 噪点, 畸变, 过曝, 构图杂乱, 多余物体" if output_lang == "中文" else "[NEGATIVE]blurry, low quality, noise, distortion, overexposed, messy composition, extra objects")}
{("[DESCRIPTION]这里是一片壮丽的山脉风景。金色的夕阳洒在连绵的山峰上，天空中飘着几朵白云，整个画面给人一种宁静祥和的感觉。" if output_lang == "中文" else "[DESCRIPTION]This is a magnificent mountain landscape. The golden sunset casts light on the rolling peaks, with a few white clouds in the sky, creating a peaceful and serene atmosphere.")}

直接输出："""


# ========== 节点类 ==========

class AIConnector:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "地址": ("STRING", {"default": cfg["base_url"], "placeholder": "http://127.0.0.1:1234"}),
                "令牌": ("STRING", {"default": cfg["token"], "placeholder": "API令牌（可选）"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI配置",)
    FUNCTION = "connect"
    CATEGORY = "AI"

    def connect(self, **kwargs):
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        model = kwargs["模型"].strip()
        _save_config(base_url=addr, token=token, model=model)
        config_json = json.dumps({
            "address": addr, 
            "token": token,
            "model": model
        })
        return (config_json,)


class AIContentInterrogator:
    """AI内容反推节点 - 输出内容描述和负向提示词"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["英文", "中文"], {"default": "中文"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("内容描述", "负向提示词")
    FUNCTION = "interrogate"
    CATEGORY = "AI"

    def interrogate(self, **kwargs):
        image_tensor = kwargs["图片"]
        config_json = kwargs["AI配置"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
            model = config.get("model", "")
        except:
            return ("", "")
        
        if not addr or not token or not model:
            return ("", "")
        
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
            
            req = _format_content_interrogation(img_base64, detail_level, output_lang)
            
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"], img_base64)
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(反推的提示词:|思考[：:]|分析[：:]|让我想想|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            parsed = _parse_marked_output(res)
            
            content_desc = ""
            if parsed["positive"] and parsed["description"]:
                content_desc = f"{parsed['positive']}\n\n{parsed['description']}"
            elif parsed["positive"]:
                content_desc = parsed["positive"]
            elif parsed["description"]:
                content_desc = parsed["description"]
            else:
                content_desc = res
            
            return (content_desc, parsed["negative"])
        except Exception as e:
            return ("", "")


class AIImagePromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "内容描述": ("STRING", {"multiline": True, "placeholder": "输入内容描述（图片反推结果）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入你的需求/意图（创作方向）"}),
                "生成模式": (["文生图", "图生图"], {"default": "文生图"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["英文", "中文"], {"default": "英文"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("正向提示词", "负向提示词")
    FUNCTION = "convert_image"
    CATEGORY = "AI"

    def convert_image(self, **kwargs):
        config_json = kwargs["AI配置"]
        optional_prompt = kwargs["内容描述"].strip()
        manual_prompt = kwargs["手工提示词"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
            model = config.get("model", "")
        except:
            return ("AI配置格式错误", "AI配置格式错误")
        
        if not optional_prompt and not manual_prompt:
            return ("请填写内容描述或手工提示词", "请填写内容描述或手工提示词")
        if not addr or not token or not model:
            return ("请先配置AI连接器", "请先配置AI连接器")
        
        req = _format_image_prompt(manual_prompt, optional_prompt, mode, detail_level, output_lang)
        key = (manual_prompt, optional_prompt, model, mode, detail_level, output_lang)
        
        if key in _CACHE:
            cached = _CACHE[key]
            parsed = _parse_marked_output(cached)
            return (parsed["positive"], parsed["negative"])
        
        try:
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(转换后的提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            parsed = _parse_marked_output(res)
            
            _CACHE[key] = f"[POSITIVE]{parsed['positive']}\n[NEGATIVE]{parsed['negative']}"
            return (parsed["positive"], parsed["negative"])
        except Exception as e:
            return (str(e), str(e))


class AIVideoPromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "内容描述": ("STRING", {"multiline": True, "placeholder": "输入内容描述（当前首帧图片描述）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入创作意图（可包含上一段内容、首尾帧描述等）"}),
                "生成模式": (["文生视频", "图生视频"], {"default": "图生视频"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["中文", "英文"], {"default": "中文"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("正向提示词", "负向提示词")
    FUNCTION = "convert_video"
    CATEGORY = "AI"

    def convert_video(self, **kwargs):
        config_json = kwargs["AI配置"]
        optional_prompt = kwargs["内容描述"].strip()
        manual_prompt = kwargs["手工提示词"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
            model = config.get("model", "")
        except:
            return ("AI配置格式错误", "AI配置格式错误")
        
        if not optional_prompt and not manual_prompt:
            return ("请填写内容描述或手工提示词", "请填写内容描述或手工提示词")
        if not addr or not token or not model:
            return ("请先配置AI连接器", "请先配置AI连接器")
        
        video_req = _format_video_prompt(manual_prompt, optional_prompt, mode, detail_level, output_lang)
        
        key = (manual_prompt, optional_prompt, model, mode, detail_level, output_lang)
        
        if key in _CACHE:
            cached = _CACHE[key]
            parsed = _parse_marked_output(cached)
            return (parsed["positive"], parsed["negative"])
        
        try:
            res = _ai_chat(addr, token, model, video_req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            parsed = _parse_marked_output(res)
            
            positive = ""
            if parsed["positive"] and parsed["description"]:
                positive = f"{parsed['positive']}\n\n{parsed['description']}"
            elif parsed["positive"]:
                positive = parsed["positive"]
            elif parsed["description"]:
                positive = parsed["description"]
            
            negative = parsed["negative"]
            
            _CACHE[key] = f"[POSITIVE]{positive}\n[NEGATIVE]{negative}"
            return (positive, negative)
        except Exception as e:
            return (f"错误: {str(e)}", f"错误: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "AIConnector": AIConnector,
    "AIContentInterrogator": AIContentInterrogator,
    "AIImagePromptConverter": AIImagePromptConverter,
    "AIVideoPromptConverter": AIVideoPromptConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIConnector": "🤖 AI 连接器",
    "AIContentInterrogator": "📝 AI 内容反推",
    "AIImagePromptConverter": "🎨 AI 图片提示词",
    "AIVideoPromptConverter": "🎬 AI 视频提示词",
}