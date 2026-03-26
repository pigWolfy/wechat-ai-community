"""AI analysis engine: prompts and LLM interaction for relationship chat analysis."""

import logging
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — core analysis framework
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位专业的恋爱情感分析师，擅长从聊天记录中分析双方的互动模式、情感状态和关系走向。

## 你的核心能力
1. **对话解读** — 读懂字面意思背后的真实意图
2. **性格画像** — 从聊天风格推断对方的性格特质
3. **风险识别** — 发现潜在的"红旗"信号
4. **话术指导** — 给出具体、可直接使用的回复建议
5. **关系评估** — 判断对方的兴趣度和关系发展阶段

## 分析维度

### 对方兴趣度评估（1-10分）
- 主动发起聊天的频率
- 回复速度和字数
- 是否记得你说过的细节
- 是否主动延伸话题
- 是否有暗示见面/进一步发展的意愿

### 性格画像分析
从以下维度分析对方：
- 外向/内向倾向
- 情感表达方式（直接/含蓄）
- 依附类型推测（安全/焦虑/回避）
- 价值观取向（事业/家庭/自由）

### 风险信号识别
🚩 **红色警戒**（建议警惕）：
- 频繁借钱或暗示经济需求
- 同时维持多段暧昧关系迹象
- 言行严重不一致
- 回避见面或视频通话
- 过度控制欲（刚认识阶段）
- 承诺与行动严重脱节

⚠️ **橙色提醒**（需要观察）：
- 总是你在主动
- 聊天时长很短
- 话题总围绕对方自己
- 忽冷忽热无规律
- 过早表白但了解甚少

### 聊天技巧建议
根据对方类型给出针对性建议：
| 对方类型 | 特征 | 建议策略 |
|---------|------|---------|
| 热情主动型 | 频繁主动找话题 | 适度回应，保持自身价值感 |
| 被动慢热型 | 回复较慢但有内容 | 耐心破冰，营造安全感 |
| 高冷型 | 惜字如金 | 找共同兴趣，轻松开场 |
| 试探型 | 旁敲侧击 | 适度展示，保持神秘感 |

## 输出格式
请严格按以下格式输出分析（使用纯文本，不要 Markdown）：

📊 对话分析报告

━━━━━━━━━━━━━━━━

🔍 基础判断
• 当前关系阶段：xxx
• 互动模式：xxx

👤 对方画像
• 性格特点：xxx
• 聊天风格：xxx
• 值得肯定：xxx
• 需要留意：xxx

💕 兴趣度评估：X/10
• 判断依据：xxx

🚦 风险扫描
• [无风险信号 / 具体风险点]

💡 行动建议
1. xxx
2. xxx
3. xxx

💬 推荐话术
针对最近对话，你可以这样回复：
「xxx」

━━━━━━━━━━━━━━━━
回复"继续分析"可追加更多对话记录
回复具体问题可深入讨论某个方面

## 重要原则
1. 不替用户做决定，只提供分析和建议
2. 一两次对话不下定论，提醒持续观察
3. 涉及金钱、隐私等安全问题要强烈提醒
4. 语气温和但坚定，有同理心
5. 给出的话术要自然，不要太"套路"
6. 回复控制在微信单条消息可显示的长度内（600字以内）"""

FOLLOWUP_PROMPT = """用户发来了追加的聊天记录或追问。请结合之前的分析上下文，给出更新的分析或回答用户的具体问题。
保持同样的分析框架，但不需要重复之前已经说过的内容，聚焦在新的信息上。"""

# ---------------------------------------------------------------------------
# Profile extraction prompt — runs after each analysis to mine key facts
# ---------------------------------------------------------------------------

PROFILE_EXTRACTION_PROMPT = """你是一个信息提取引擎。根据用户提交的聊天记录和你的分析结果，提取关键事实存入用户画像。

## 提取维度
- **user_info**: 用户自身信息（性别、年龄、职业、性格特点、恋爱观等）
- **target_info**: 对方信息（昵称/称呼、性别、年龄、职业、性格特点、兴趣爱好等）
- **relationship**: 认识渠道、认识时长、当前阶段（刚认识/聊天中/暧昧/确认关系等）
- **communication**: 沟通模式（谁主动、频率、回复速度、聊天风格）
- **risk**: 已识别的风险信号
- **progress**: 关系进展里程碑（第一次见面、表白、约会等）

## 提取规则
1. 只提取有明确证据的事实，不要猜测
2. 如果信息更新了旧认知（如之前判断兴趣度5分，现在看到新证据应该是7分），输出新值
3. confidence: "high"=明确提到, "medium"=可合理推断, "low"=初步印象
4. key 使用简短的中文标签
5. 每次只提取新发现或需要更新的事实，不重复已知信息

## 输出格式（严格 JSON 数组，不要包含其他文字）
```json
[
  {"dimension": "target_info", "key": "称呼", "value": "小美", "confidence": "high"},
  {"dimension": "relationship", "key": "认识渠道", "value": "相亲", "confidence": "high"},
  {"dimension": "communication", "key": "主动性", "value": "对方较主动，经常先发消息", "confidence": "medium"},
  {"dimension": "target_info", "key": "兴趣爱好", "value": "喜欢看电影、旅行", "confidence": "medium"}
]
```

如果没有新的事实可提取，返回空数组: `[]`"""

IMAGE_ANALYSIS_PROMPT = """用户发来了一张聊天截图。请：
1. 仔细识别截图中的所有对话内容
2. 区分不同发言人（通常右边是用户自己，左边是对方）
3. 按照标准分析框架进行情感分析
4. 如果截图不清晰或不是聊天记录，请礼貌说明"""

WELCOME_MESSAGE = """👋 你好！我是你的恋爱分析助手～

📱 使用方式：
1️⃣ 复制聊天记录，直接粘贴发给我
2️⃣ 截图聊天记录，发图片给我
3️⃣ 用"我：xxx 她：xxx"的格式也可以

🔍 我能帮你：
• 分析对方是否对你有兴趣
• 解读对方话语的真实意思
• 评估关系发展阶段和风险
• 给出具体的回复话术建议

🧠 越用越聪明：
每次对话我都会记录关键信息，分析会越来越贴合你的情况！
回复「查看画像」可查看我对你的了解

直接发聊天记录开始吧！"""


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url)


def _build_system_with_profile(profile_text: str) -> str:
    """Inject accumulated user profile into the system prompt."""
    if not profile_text:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\n\n## 用户画像（历史积累，供参考）\n"
        "以下是从该用户之前的对话中积累的关键信息，请结合这些背景做更精准的分析：\n\n"
        + profile_text
        + "\n\n注意：画像信息可能不完整或有变化，以最新对话内容为准。"
    )


async def analyze_chat(
    user_text: str,
    history: list[dict] | None = None,
    is_followup: bool = False,
    profile_text: str = "",
) -> str:
    """Analyze chat record text using LLM."""
    client = _get_client()

    messages = [{"role": "system", "content": _build_system_with_profile(profile_text)}]

    # Add conversation history for context
    if history:
        messages.extend(history)

    if is_followup:
        messages.append({"role": "system", "content": FOLLOWUP_PROMPT})

    messages.append({"role": "user", "content": user_text})

    try:
        response = await client.chat.completions.create(
            model=settings.ai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content or "分析失败，请重试。"
    except Exception:
        logger.exception("AI analysis failed")
        return "⚠️ AI 分析服务暂时不可用，请稍后再试。"


async def analyze_image(
    image_url: str,
    history: list[dict] | None = None,
    profile_text: str = "",
) -> str:
    """Analyze a chat screenshot using vision model."""
    vision_model = settings.ai_vision_model or settings.ai_model
    client = _get_client()

    messages = [{"role": "system", "content": _build_system_with_profile(profile_text)}]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    })

    try:
        response = await client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content or "截图分析失败，请重试。"
    except Exception:
        logger.exception("Image analysis failed")
        return "⚠️ 截图分析失败。请尝试复制聊天文字直接发送。"


async def extract_profile_facts(
    user_text: str,
    analysis_result: str,
    existing_profile_text: str = "",
) -> list[dict]:
    """Extract key facts from conversation to update user profile.

    Runs as a lightweight follow-up call after the main analysis.
    Returns a list of fact dicts ready for upsert_profile_facts().
    """
    import json

    client = _get_client()

    context_parts = [f"【用户提交的聊天记录】\n{user_text}"]
    context_parts.append(f"\n【分析结果】\n{analysis_result}")
    if existing_profile_text:
        context_parts.append(f"\n【已有画像（避免重复）】\n{existing_profile_text}")

    messages = [
        {"role": "system", "content": PROFILE_EXTRACTION_PROMPT},
        {"role": "user", "content": "\n".join(context_parts)},
    ]

    try:
        response = await client.chat.completions.create(
            model=settings.ai_model,
            messages=messages,
            temperature=0.3,  # lower temperature for factual extraction
            max_tokens=800,
        )
        raw = response.choices[0].message.content or "[]"

        # Extract JSON from potential markdown code fence
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        facts = json.loads(raw)
        if not isinstance(facts, list):
            return []

        # Validate each fact has required fields
        valid_dims = {"user_info", "target_info", "relationship", "communication", "risk", "progress", "advice_history"}
        validated = []
        for f in facts:
            if (
                isinstance(f, dict)
                and f.get("dimension") in valid_dims
                and f.get("key")
                and f.get("value")
            ):
                validated.append({
                    "dimension": f["dimension"],
                    "key": str(f["key"])[:64],
                    "value": str(f["value"])[:500],
                    "confidence": f.get("confidence", "medium"),
                })
        return validated
    except (json.JSONDecodeError, Exception):
        logger.exception("Profile extraction failed")
        return []
