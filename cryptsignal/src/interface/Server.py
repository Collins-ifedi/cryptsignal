# server.py
import os
import sys
import json
import logging
import asyncio
from http import HTTPStatus
from pathlib import Path
from typing import Dict

# --- Starlette Imports ---
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Your original application logic is imported here
from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# --- Config ---
MEDIA_DIR = PROJECT_ROOT / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedServer")

# --- Global Session Management (Unchanged) ---
ACTIVE_SESSIONS: Dict[int, "ChatSession"] = {}

# -------------------------------------------------------
# AI Query Processing (Unchanged)
# -------------------------------------------------------
async def process_api_query(query: str, user_id: int) -> str:
    logger.info(f"[AI] User {user_id} -> {query}")
    try:
        handler_instance = TradingAssistantNLPHandler(user_id=user_id)
        await handler_instance.initialize()
        response_text, _ = await handler_instance.handle_query(query)
        return response_text
    except Exception:
        logger.exception(f"Error in NLP handler for user {user_id}")
        return "An internal AI error occurred."

# -------------------------------------------------------
# WebSocket ChatSession Class (Unchanged)
# -------------------------------------------------------
class ChatSession:
    def __init__(self, websocket: WebSocket, user_id: int):
        self.websocket = websocket
        self.user_id = user_id
        self.nlp_handler = TradingAssistantNLPHandler(user_id=user_id)

    async def initialize(self):
        await self.nlp_handler.initialize()
        welcome = await self.nlp_handler.get_dynamic_welcome()
        await self.send(welcome, "assistant")

    async def send(self, text: str, sender: str):
        await self.websocket.send_text(json.dumps({"type": "new_message", "sender": sender, "text": text}))

    async def handle_user_query(self, text: str):
        response, should_exit = await self.nlp_handler.handle_query(text)
        await self.send(response, "assistant")
        if should_exit:
            await self.websocket.close()

# -------------------------------------------------------
# HTTP API Endpoint (Refactored for Starlette)
# -------------------------------------------------------
async def api_generate_endpoint(request):
    """Handles POST requests to generate an AI response."""
    try:
        body = await request.json()
        query = body.get("query")
        user_id = body.get("userId")

        if not query or user_id is None:
            return JSONResponse(
                {"error": "query and userId required"},
                status_code=HTTPStatus.BAD_REQUEST,
            )

        response_text = await process_api_query(query, int(user_id))

        return JSONResponse({"response": response_text})

    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=HTTPStatus.BAD_REQUEST)
    except Exception as e:
        logger.exception(f"HTTP API error: {e}")
        return JSONResponse({"error": "Internal Server Error"}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

# -------------------------------------------------------
# WebSocket Endpoint (Refactored for Starlette)
# -------------------------------------------------------
async def websocket_handler_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection and message flow."""
    await websocket.accept()
    logger.info(f"WebSocket client connected: {websocket.client}")
    session = None
    user_id = None

    try:
        # The original logic for initialization and message handling is preserved
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        user_id = init_data.get("userId")
        if not user_id or init_data.get("type") != "init":
            await websocket.close(code=1008, reason="Invalid init message")
            return

        session = ChatSession(websocket, user_id)
        ACTIVE_SESSIONS[user_id] = session
        await session.initialize()

        async for message in websocket.iter_text():
            data = json.loads(message)
            if data.get("type") == "send_message":
                await session.handle_user_query(data.get("text", ""))

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        if user_id and user_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[user_id]
            logger.info(f"Session removed for user {user_id}")
        logger.info(f"WebSocket closed for client {websocket.client}")
        # Starlette handles closing the connection, but you can add cleanup here

# -------------------------------------------------------
# Application Setup
# -------------------------------------------------------
# Define the routes for the application
routes = [
    Route("/api/generate", endpoint=api_generate_endpoint, methods=["POST"]),
    WebSocketRoute("/ws", endpoint=websocket_handler_endpoint),
    # This Mount serves all your frontend files (index.html, main.js, etc.)
    # It MUST be the last route.
    Mount("/", app=StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static"),
]

# Create the main Starlette application instance
app = Starlette(debug=False, routes=routes)

# To run this server, use the command:
# uvicorn Server:app --host 0.0.0.0 --port 8000