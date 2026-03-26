"""WeChat message handling: parsing, verification, reply formatting."""

import hashlib
import time
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET


@dataclass
class WeChatMessage:
    """Parsed incoming WeChat message."""
    to_user: str = ""
    from_user: str = ""
    create_time: int = 0
    msg_type: str = ""
    msg_id: str = ""
    content: str = ""          # text messages
    pic_url: str = ""          # image messages
    media_id: str = ""         # image/voice/video media id
    recognition: str = ""      # voice recognition result
    event: str = ""            # event type (subscribe, unsubscribe, etc.)
    event_key: str = ""        # event key


def verify_signature(token: str, signature: str, timestamp: str, nonce: str) -> bool:
    """Verify WeChat server callback signature."""
    items = sorted([token, timestamp, nonce])
    hash_str = hashlib.sha1("".join(items).encode("utf-8")).hexdigest()
    return hash_str == signature


def parse_xml_message(xml_bytes: bytes) -> WeChatMessage:
    """Parse incoming XML message from WeChat."""
    root = ET.fromstring(xml_bytes)
    msg = WeChatMessage()
    msg.to_user = _get_text(root, "ToUserName")
    msg.from_user = _get_text(root, "FromUserName")
    msg.create_time = int(_get_text(root, "CreateTime") or "0")
    msg.msg_type = _get_text(root, "MsgType")
    msg.msg_id = _get_text(root, "MsgId")
    msg.content = _get_text(root, "Content")
    msg.pic_url = _get_text(root, "PicUrl")
    msg.media_id = _get_text(root, "MediaId")
    msg.recognition = _get_text(root, "Recognition")
    msg.event = _get_text(root, "Event")
    msg.event_key = _get_text(root, "EventKey")
    return msg


def build_text_reply(from_user: str, to_user: str, content: str) -> str:
    """Build XML text reply message."""
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "</xml>"
    )


def _get_text(root: ET.Element, tag: str) -> str:
    el = root.find(tag)
    return (el.text or "") if el is not None else ""
