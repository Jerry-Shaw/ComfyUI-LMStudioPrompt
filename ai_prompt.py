import re
import requests
import json
import os
import time
from typing import Dict, Tuple, List

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
                # 直接更新默认配置
                default.update(user)
    except:
        pass
    return default

def _save_config(base_url=None, token=None, model=None):
    global _CONFIG
    try:
        # 读取现有配置
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        
        # 更新配置
        if base_url:
            cfg["base_url"] = base_url
        if token:
            cfg["token"] = token
        if model:
            cfg["last_model"] = model
        
        # 保存配置
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        # 重新加载配置
        _CONFIG = _load_config()
    except:
        pass

_CONFIG = _load_config()

def _ai_chat(url: str, token: str, model: str, msg: str, timeout: int = 60) -> str:
    """通用的AI聊天接口，支持OpenAI兼容的API"""
    base = url.rstrip('/')
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
            if "response" in data:
                res = data["response"].strip()
            elif "choices" in data:
                if "message" in data["choices"][0]:
                    res = data["choices"][0]["message"]["content"].strip()
                elif "text" in data["choices"][0]:
                    res = data["choices"][0]["text"].strip()
                else:
                    res = str(data["choices"][0]).strip()
            else:
                res = str(data).strip()
            _save_config(model=model)
            return res
        except:
            continue
    
    raise Exception("连接失败")

def _format_prompt(text: str, ptype: str) -> str:
    """格式化图片提示词，严格围绕原始输入"""
    hint = "正向" if ptype == "positive" else "负向"
    return f"""你是一个专业的Stable Diffusion提示词转换专家。
请将以下用户输入转换为高质量的{hint}提示词，用于AI绘画。

【核心原则】
严格围绕用户输入的原始内容进行转换，不要自行添加新元素

【重要规则】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"等思考性词语
3. 禁止输出"思考过程："、"分析："等标签
4. 直接输出转换后的提示词，不要有任何前缀或后缀
5. 不要添加用户输入中没有描述的元素
6. 不要过度美化或扩展内容

【转换规则】
1. 保持所有权重标记不变，如 (word:1.2)、[word]、{{word}}
2. 保持所有特殊标记不变，如 BREAK、AND
3. 使用英文逗号分隔不同的提示词元素
4. 输出简洁、专业的英文提示词
5. 如果用户输入不完整，只补全必要的语法结构，不添加额外描述

用户输入：{text}

转换后的英文提示词："""

def _format_video_prompt(text: str, mode: str, detail_level: str) -> str:
    """格式化视频提示词，针对WAN2.2视频模型优化，保持中文"""
    
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
    
    # 根据生成模式区分
    if mode == "文生视频":
        mode_guide = """【文生视频模式说明】
从零构建视觉场景，侧重：
- 完整的场景设定和环境构建
- 主体形象的详细刻画
- 动作的完整过程描述
- 时间流逝和变化过程"""
    else:  # 图生视频
        mode_guide = """【图生视频模式说明】
基于参考图生成视频，侧重：
- 动作的流畅性和连续性
- 主体在画面中的动态变化
- 场景的微妙变化而非剧烈转变
- 保持与参考图的一致性"""
    
    return f"""你是一个专业的WAN2.2视频生成提示词优化专家。你的任务是根据用户输入的简要描述，生成高质量的中文视频提示词。

【核心任务】
{mode_guide}

{strategy}

【重要规则】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"等思考性词语
3. 禁止输出"优化后："、"润色后："、"提示词："等标签
4. 直接输出处理后的提示词，不要有任何前缀或后缀
5. **必须保持中文输出，不要翻译成英文**
6. 输出内容为纯文本，不要使用markdown格式

【WAN2.2视频模型特点】
- 原生支持中文提示词
- 对场景变化和动作连续性敏感
- 理解详细的视觉描述
- 支持镜头运动描述

用户输入的原始描述：{text}

请直接输出处理后的中文视频提示词："""

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
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("地址", "令牌")
    FUNCTION = "connect"
    CATEGORY = "AI"

    def connect(self, **kwargs):
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        _save_config(base_url=addr, token=token)
        return (addr, token)

class AIImagePromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "placeholder": "输入图片提示词..."}),
                "地址": ("STRING", {"forceInput": True}),
                "令牌": ("STRING", {"forceInput": True}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称，例如: gpt-4, qwen, llama3, deepseek"}),
                "类型": (["正向", "负向"], {"default": "正向"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("图片提示词",)
    FUNCTION = "convert"
    CATEGORY = "AI"

    def convert(self, **kwargs):
        prompt = kwargs["提示词"].strip()
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        model = kwargs["模型"].strip()
        ptype = "positive" if kwargs["类型"] == "正向" else "negative"
        
        if not prompt:
            return ("请输入图片提示词",)
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入模型名称",)
        
        req = _format_prompt(prompt, ptype)
        key = (prompt, model, ptype)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            # 清理思考过程
            res = re.sub(r'^(转换后的英文提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述)[\s:]*', '', res, flags=re.I)
            
            # 取第一段非空行
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
                "视频描述": ("STRING", {"multiline": True, "placeholder": "输入简要的视频描述（支持中英文）..."}),
                "地址": ("STRING", {"forceInput": True}),
                "令牌": ("STRING", {"forceInput": True}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称，例如: gpt-4, qwen, llama3, deepseek"}),
                "生成模式": (["文生视频", "图生视频"], {"default": "文生视频"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频提示词",)
    FUNCTION = "convert_video"
    CATEGORY = "AI"

    def convert_video(self, **kwargs):
        prompt = kwargs["视频描述"].strip()
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        model = kwargs["模型"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        
        if not prompt:
            return ("请输入视频描述",)
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入模型名称",)
        
        # 构建针对WAN2.2的提示词（保持中文）
        video_req = _format_video_prompt(prompt, mode, detail_level)
        
        key = (prompt, model, mode, detail_level)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _ai_chat(addr, token, model, video_req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            # 清理可能的思考过程
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述)[\s:]*', '', res, flags=re.I)
            
            # 取第一段非空行作为结果，跳过可能的代码块标记
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
    "AIImagePromptConverter": AIImagePromptConverter,
    "AIVideoPromptConverter": AIVideoPromptConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIConnector": "🤖 AI 连接器",
    "AIImagePromptConverter": "🎨 AI 图片提示词",
    "AIVideoPromptConverter": "🎬 AI 视频提示词",
}