import re
import requests
import json
import os
import time
from typing import Dict, Tuple, List

_CACHE: Dict[Tuple[str, str, str], str] = {}
_CONFIG = None
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def _load_config():
    default = {
        "lmstudio": {
            "base_url": "http://127.0.0.1:1234",
            "token": "lm-studio-api-token",
            "timeout": 60,
            "last_model": ""
        }
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                user = json.load(f)
                for k in user:
                    if k in default and isinstance(user[k], dict):
                        default[k].update(user[k])
                    else:
                        default[k] = user[k]
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
        if "lmstudio" not in cfg:
            cfg["lmstudio"] = {}
        if base_url:
            cfg["lmstudio"]["base_url"] = base_url
        if token:
            cfg["lmstudio"]["token"] = token
        if model:
            cfg["lmstudio"]["last_model"] = model
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        _CONFIG = _load_config()
    except:
        pass

_CONFIG = _load_config()

def _lmstudio_chat(url: str, token: str, model: str, msg: str, timeout: int = 60) -> str:
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
    hint = "正向" if ptype == "positive" else "负向"
    return f"""你是一个专业的Stable Diffusion提示词转换专家。
请将以下用户输入转换为高质量的{hint}提示词，用于AI绘画。

【重要规则】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"等思考性词语
3. 禁止输出"思考过程："、"分析："等标签
4. 直接输出转换后的提示词，不要有任何前缀或后缀

【转换规则】
1. 保持所有权重标记不变，如 (word:1.2)、[word]、{{word}}
2. 保持所有特殊标记不变，如 BREAK、AND
3. 使用英文逗号分隔不同的提示词元素
4. 输出简洁、专业的英文提示词

用户输入：{text}

转换后的英文提示词："""

class LMStudioConnector:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG["lmstudio"]
        return {
            "required": {
                "地址": ("STRING", {"default": cfg["base_url"], "placeholder": "http://127.0.0.1:1234"}),
                "令牌": ("STRING", {"default": cfg["token"], "placeholder": "LM-Studio令牌"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("地址", "令牌")
    FUNCTION = "connect"
    CATEGORY = "LM-Studio"

    def connect(self, **kwargs):
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        _save_config(base_url=addr, token=token)
        return (addr, token)

class LMStudioPromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG["lmstudio"]
        last = cfg["last_model"] or ""
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "placeholder": "输入提示词..."}),
                "地址": ("STRING", {"forceInput": True}),
                "令牌": ("STRING", {"forceInput": True}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称，例如: qwen/qwen3.5-9b"}),
                "类型": (["正向", "负向"], {"default": "正向"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("结果",)
    FUNCTION = "convert"
    CATEGORY = "LM-Studio"

    def convert(self, **kwargs):
        prompt = kwargs["提示词"].strip()
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        model = kwargs["模型"].strip()
        ptype = "positive" if kwargs["类型"] == "正向" else "negative"
        
        if not prompt:
            return ("请输入提示词",)
        if not addr or not token:
            return ("请先连接LM-Studio",)
        if not model:
            return ("请输入模型名称",)
        
        req = _format_prompt(prompt, ptype)
        key = (prompt, model, ptype)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _lmstudio_chat(addr, token, model, req, _CONFIG["lmstudio"]["timeout"])
            res = res.strip().strip('"\'')
            res = re.sub(r'^(转换后的英文提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述)[\s:]*', '', res, flags=re.I)

            if '\n' in res:
                lines = [line.strip() for line in res.split('\n') if line.strip()]
                if lines:
                    res = lines[0]
            _CACHE[key] = res
            return (res,)
        except Exception as e:
            return (str(e),)

NODE_CLASS_MAPPINGS = {
    "LMStudioConnector": LMStudioConnector,
    "LMStudioPromptConverter": LMStudioPromptConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioConnector": "LM-Studio 连接器",
    "LMStudioPromptConverter": "LM-Studio 转换器",
}