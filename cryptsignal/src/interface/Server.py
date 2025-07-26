import os
import json
import logging
import asyncio
from pathlib import Path
from aiohttp import web, WSMsgType
from typing import Dict, Optional

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = PROJECT_ROOT / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIOHTTPServer")

# Global state
ACTIVE_SESSIONS: Dict[int, "ChatSession"] = {}

# -------------------------------
# Chat Session Class
# -------------------------------
class ChatSession:
    def __init__(self, ws: web.WebSocketResponse, user_id: int):
        self.ws = ws
        self.user_id = user_id
        self.nlp_handler = TradingAssistantNLPHandler(user_id=user_id)

    async def initialize(self):
        await self.nlp_handler.initialize()
        welcome = await self.nlp_handler.get_dynamic_welcome()
        await self.send_message(welcome, "assistant")

    async def send_message(self, text: str, sender: str):
        await self.ws.send_json({
            "type": "new_message",
            "sender": sender,
            "text": text
        })

    async def handle_query(self, query: str):
        response, should_exit = await self.nlp_handler.handle_query(query)
        await self.send_message(response, "assistant")
        if should_exit:
            await self.ws.close()

# -------------------------------
# API Handlers
# -------------------------------
async def handle_generate(request: web.Request) -> web.Response:
    """HTTP endpoint: POST /api/generate"""
    try:
        data = await request.json()
        query = data.get("query")
        user_id = int(data.get("userId", 0))

        if not query or not user_id:
            return web.json_response({"error": "Query and userId are required"}, status=400)

        handler = TradingAssistantNLPHandler(user_id=user_id)
        await handler.initialize()
        response_text, _ = await handler.handle_query(query)

        return web.json_response({"response": response_text})

    except Exception as e:
        logger.exception("Error in /api/generate")
        return web.json_response({"error": "Internal server error"}, status=500)

# -------------------------------
# WebSocket Handler
# -------------------------------
async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    user_id: Optional[int] = None
    session: Optional[ChatSession] = None

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "init":
                    user_id = int(data.get("userId", 0))
                    if not user_id:
                        await ws.send_json({"error": "userId is required"})
                        await ws.close()
                        break

                    session = ChatSession(ws, user_id)
                    ACTIVE_SESSIONS[user_id] = session
                    await session.initialize()

                elif data.get("type") == "send_message" and session:
                    query = data.get("text", "").strip()
                    if query:
                        await session.handle_query(query)

            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket connection closed with error {ws.exception()}")

    finally:
        if user_id and user_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[user_id]
            logger.info(f"Session for user {user_id} cleaned up")

    return ws

# -------------------------------
# App Setup
# -------------------------------
async def init_app() -> web.Application:
    app = web.Application()

    # Routes
    app.router.add_post("/api/generate", handle_generate)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/media/", path=MEDIA_DIR, name="media")

    return app

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    web.run_app(init_app(), host="0.0.0.0", port=port)