"""WeChat AI Relationship Coach — FastAPI entry point.

Handles WeChat server verification, incoming messages, and orchestrates
async AI analysis with WeChat's retry mechanism for the 5-second timeout.
"""

import asyncio
import logging
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Query, Response
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.wechat import (
    verify_signature,
    parse_xml_message,
    build_text_reply,
    WeChatMessage,
)
from app.parser import parse_chat_record, format_for_ai
from app.analyzer import (
    analyze_chat,
    analyze_image,
    extract_profile_facts,
    WELCOME_MESSAGE,
)
from app.database import (
    init_db,
    save_message,
    get_recent_history,
    set_pending_task,
    complete_pending_task,
    get_pending_task,
    get_user_profile,
    upsert_profile_facts,
    format_profile_for_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WeChat AI Relationship Coach")

# In-memory cache for ongoing tasks: msg_id -> asyncio.Task
_running_tasks: dict[str, asyncio.Task] = {}
# In-memory cache for quick result lookup (avoids DB roundtrip on retry)
_result_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    Path("data").mkdir(exist_ok=True)
    await init_db()
    logger.info("Database initialized")


# ---------------------------------------------------------------------------
# WeChat server verification (GET)
# ---------------------------------------------------------------------------

@app.get("/wechat")
async def wechat_verify(
    signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query(""),
    echostr: str = Query(""),
):
    if verify_signature(settings.wechat_token, signature, timestamp, nonce):
        return PlainTextResponse(echostr)
    return PlainTextResponse("Invalid signature", status_code=403)


# ---------------------------------------------------------------------------
# WeChat message handler (POST)
# ---------------------------------------------------------------------------

@app.post("/wechat")
async def wechat_message(request: Request):
    body = await request.body()
    msg = parse_xml_message(body)

    # Handle events (subscribe / unsubscribe)
    if msg.msg_type == "event":
        return _handle_event(msg)

    # Handle text messages
    if msg.msg_type == "text":
        return await _handle_text(msg)

    # Handle image messages (screenshots)
    if msg.msg_type == "image":
        return await _handle_image(msg)

    # Handle voice messages (use recognition text)
    if msg.msg_type == "voice" and msg.recognition:
        msg.content = msg.recognition
        return await _handle_text(msg)

    # Unsupported message types
    return _reply(msg, "目前支持文字和图片消息哦～\n请复制聊天记录发文字，或截图发给我。")


# ---------------------------------------------------------------------------
# Message type handlers
# ---------------------------------------------------------------------------

def _handle_event(msg: WeChatMessage) -> Response:
    if msg.event.lower() == "subscribe":
        return _reply(msg, WELCOME_MESSAGE)
    return PlainTextResponse("success")


async def _handle_text(msg: WeChatMessage) -> Response:
    text = msg.content.strip()

    # Quick commands
    if text in ("帮助", "help", "?", "？"):
        return _reply(msg, WELCOME_MESSAGE)

    # Check if user is asking for a pending result
    if text in ("查看结果", "结果", "分析结果"):
        return await _get_latest_result(msg)

    # Check if user wants to see their profile
    if text in ("查看画像", "我的画像", "画像", "档案"):
        return await _show_profile(msg)

    # --- Retry mechanism ---
    # WeChat retries with the same MsgId up to 3 times (at ~5s intervals)
    # 1st call: start async processing, return "processing" message
    # 2nd/3rd call: if done, return result; if not, return wait message
    msg_id = msg.msg_id

    # Check if result is already cached
    if msg_id in _result_cache:
        result = _result_cache.pop(msg_id)
        return _reply(msg, result)

    # Check if task is already running for this msg_id
    if msg_id in _running_tasks:
        task = _running_tasks[msg_id]
        if task.done():
            # Task finished — return result
            result = task.result()
            _running_tasks.pop(msg_id, None)
            return _reply(msg, result)
        else:
            # Still processing — tell user to wait
            return _reply(msg, "🔄 正在深度分析中，请稍候...\n如超时可回复「查看结果」获取报告。")

    # First time seeing this message — start async analysis
    parsed = parse_chat_record(text)
    ai_input = format_for_ai(parsed)

    # Get conversation history and user profile for context
    history = await get_recent_history(msg.from_user, limit=6)
    is_followup = len(history) > 0
    profile_facts = await get_user_profile(msg.from_user)
    profile_text = format_profile_for_prompt(profile_facts)

    # Save user message
    await save_message(msg.from_user, "user", text)

    # Start async task
    task = asyncio.create_task(_process_analysis(
        msg_id, msg.from_user, ai_input, history, is_followup, profile_text
    ))
    _running_tasks[msg_id] = task

    # Wait a bit to see if it completes quickly
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=4.0)
        result = task.result()
        _running_tasks.pop(msg_id, None)
        return _reply(msg, result)
    except asyncio.TimeoutError:
        # Not done yet — return processing message; WeChat will retry
        return _reply(msg, "🔍 收到！正在为你分析聊天记录...\n分析约需10-15秒，请耐心等待～")


async def _handle_image(msg: WeChatMessage) -> Response:
    """Handle image messages (chat screenshots)."""
    if not msg.pic_url:
        return _reply(msg, "图片接收失败，请重新发送。")

    msg_id = msg.msg_id

    # Retry check
    if msg_id in _result_cache:
        result = _result_cache.pop(msg_id)
        return _reply(msg, result)

    if msg_id in _running_tasks:
        task = _running_tasks[msg_id]
        if task.done():
            result = task.result()
            _running_tasks.pop(msg_id, None)
            return _reply(msg, result)
        return _reply(msg, "🔄 正在识别截图内容...\n如超时可回复「查看结果」获取报告。")

    history = await get_recent_history(msg.from_user, limit=6)
    profile_facts = await get_user_profile(msg.from_user)
    profile_text = format_profile_for_prompt(profile_facts)
    await save_message(msg.from_user, "user", "[聊天截图]")

    # Download image and convert to base64 for vision API
    image_url = await _download_wechat_image(msg.pic_url)

    task = asyncio.create_task(_process_image_analysis(
        msg_id, msg.from_user, image_url, history, profile_text
    ))
    _running_tasks[msg_id] = task

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=4.0)
        result = task.result()
        _running_tasks.pop(msg_id, None)
        return _reply(msg, result)
    except asyncio.TimeoutError:
        return _reply(msg, "🔍 收到截图！正在识别和分析中...\n分析约需15-20秒，请耐心等待～")


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

async def _process_analysis(
    msg_id: str, open_id: str, text: str, history: list[dict],
    is_followup: bool, profile_text: str = "",
) -> str:
    """Run AI analysis in background."""
    try:
        result = await analyze_chat(text, history, is_followup, profile_text)
        # Save to DB and cache
        await save_message(open_id, "assistant", result)
        _result_cache[msg_id] = result

        # Extract profile facts in background (don't block response)
        asyncio.create_task(_update_profile(open_id, text, result, profile_text))

        # If customer service API is enabled, push result directly
        if settings.enable_customer_service_api:
            await _send_customer_service_message(open_id, result)

        return result
    except Exception:
        logger.exception("Analysis task failed")
        return "⚠️ 分析出错了，请重新发送聊天记录试试。"


async def _process_image_analysis(
    msg_id: str, open_id: str, image_url: str, history: list[dict],
    profile_text: str = "",
) -> str:
    """Run image analysis in background."""
    try:
        result = await analyze_image(image_url, history, profile_text)
        await save_message(open_id, "assistant", result)
        _result_cache[msg_id] = result

        # Extract profile facts from image analysis too
        asyncio.create_task(_update_profile(open_id, "[聊天截图分析]", result, profile_text))

        if settings.enable_customer_service_api:
            await _send_customer_service_message(open_id, result)

        return result
    except Exception:
        logger.exception("Image analysis task failed")
        return "⚠️ 截图分析出错了，请尝试复制聊天文字发送。"


# ---------------------------------------------------------------------------
# Profile update (runs in background after each analysis)
# ---------------------------------------------------------------------------

async def _update_profile(
    open_id: str, user_text: str, analysis_result: str, existing_profile_text: str
):
    """Extract key facts from conversation and update user profile."""
    try:
        facts = await extract_profile_facts(user_text, analysis_result, existing_profile_text)
        if facts:
            await upsert_profile_facts(open_id, facts)
            logger.info("Updated profile for %s: %d facts", open_id, len(facts))
    except Exception:
        logger.exception("Profile update failed for %s", open_id)


# ---------------------------------------------------------------------------
# Helper: show user profile
# ---------------------------------------------------------------------------

async def _show_profile(msg: WeChatMessage) -> Response:
    """Show accumulated profile facts to user."""
    profile_facts = await get_user_profile(msg.from_user)
    if not profile_facts:
        return _reply(msg, "📋 你的画像还是空白的～\n发送聊天记录，我会自动积累对你和对方的了解，分析会越来越精准！")

    profile_text = format_profile_for_prompt(profile_facts)
    header = f"📋 你的专属画像（共 {len(profile_facts)} 条记录）\n━━━━━━━━━━━━━━━━\n"
    footer = "\n━━━━━━━━━━━━━━━━\n💡 每次对话都会自动更新画像，越聊越懂你"
    return _reply(msg, header + profile_text + footer)


# ---------------------------------------------------------------------------
# Helper: get latest result for a user
# ---------------------------------------------------------------------------

async def _get_latest_result(msg: WeChatMessage) -> Response:
    """Return the most recent analysis result for this user."""
    # Check in-memory cache first
    for mid, result in list(_result_cache.items()):
        # Simple check — in production, you'd want to filter by open_id
        _result_cache.pop(mid, None)
        return _reply(msg, result)

    # Check running tasks
    for mid, task in list(_running_tasks.items()):
        if task.done():
            result = task.result()
            _running_tasks.pop(mid, None)
            return _reply(msg, result)

    # Check if any tasks are still running
    if _running_tasks:
        return _reply(msg, "⏳ 分析还在进行中，请再等几秒后回复「查看结果」。")

    return _reply(msg, "暂时没有待查看的分析结果。\n请发送聊天记录开始分析～")


# ---------------------------------------------------------------------------
# WeChat API helpers
# ---------------------------------------------------------------------------

_access_token_cache: dict[str, tuple[str, float]] = {}


async def _get_access_token() -> str:
    """Get WeChat access token (cached)."""
    cache = _access_token_cache.get("token")
    if cache and cache[1] > time.time():
        return cache[0]

    url = "https://api.weixin.qq.com/cgi-bin/token"
    params = {
        "grant_type": "client_credential",
        "appid": settings.wechat_app_id,
        "secret": settings.wechat_app_secret,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    token = data.get("access_token", "")
    expires_in = data.get("expires_in", 7200)
    _access_token_cache["token"] = (token, time.time() + expires_in - 300)
    return token


async def _send_customer_service_message(open_id: str, content: str):
    """Send message via customer service API (requires certified account)."""
    token = await _get_access_token()
    url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}"
    payload = {
        "touser": open_id,
        "msgtype": "text",
        "text": {"content": content},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        result = resp.json()
        if result.get("errcode", 0) != 0:
            logger.warning("Customer service message failed: %s", result)


async def _download_wechat_image(pic_url: str) -> str:
    """Download image from WeChat and convert to data URL for vision API.

    Returns a data URL (data:image/jpeg;base64,...) that can be passed
    to multimodal LLMs.
    """
    import base64

    async with httpx.AsyncClient() as client:
        resp = await client.get(pic_url, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg")
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{content_type};base64,{b64}"


# ---------------------------------------------------------------------------
# Reply helper
# ---------------------------------------------------------------------------

def _reply(msg: WeChatMessage, content: str) -> Response:
    """Build an XML text reply."""
    xml = build_text_reply(msg.to_user, msg.from_user, content)
    return Response(content=xml, media_type="application/xml")
