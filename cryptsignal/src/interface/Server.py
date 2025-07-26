# server.py
import os
import sys
import json
import logging
import asyncio
from http import HTTPStatus
from websockets.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed
from websockets.http import Headers
from websockets.server import WebSocketServerProtocol
from websockets.typing import Data
from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Dict, Optional
import threading

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# --- Config ---
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
MEDIA_DIR = PROJECT_ROOT / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedServer")

ACTIVE_SESSIONS: Dict[int, "ChatSession"] = {}
main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# -------------------------------------------------------
# HTTP Handler
# -------------------------------------------------------
class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            if not main_event_loop:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Service not ready"}).encode("utf-8"))
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)

                query = data.get("query")
                user_id = data.get("userId")
                if not query or user_id is None:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "query and userId required"}).encode("utf-8"))
                    return

                # Run AI query in asyncio loop
                future = asyncio.run_coroutine_threadsafe(
                    process_api_query(query, int(user_id)), main_event_loop
                )
                response_text = future.result(timeout=60)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"response": response_text}).encode("utf-8"))

            except Exception as e:
                logger.exception(f"HTTP error: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


# -------------------------------------------------------
# AI Query Processing
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
# WebSocket ChatSession
# -------------------------------------------------------
class ChatSession:
    def __init__(self, websocket: WebSocketServerProtocol, user_id: int):
        self.websocket = websocket
        self.user_id = user_id
        self.nlp_handler = TradingAssistantNLPHandler(user_id=user_id)

    async def initialize(self):
        await self.nlp_handler.initialize()
        welcome = await self.nlp_handler.get_dynamic_welcome()
        await self.send(welcome, "assistant")

    async def send(self, text: str, sender: str):
        await self.websocket.send(json.dumps({"type": "new_message", "sender": sender, "text": text}))

    async def handle_user_query(self, text: str):
        response, should_exit = await self.nlp_handler.handle_query(text)
        await self.send(response, "assistant")
        if should_exit:
            await self.websocket.close()


async def websocket_handler(websocket: WebSocketServerProtocol):
    logger.info(f"WebSocket client connected: {websocket.remote_address}")
    session = None
    user_id = None

    try:
        init_data = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
        user_id = init_data.get("userId")
        if not user_id or init_data.get("type") != "init":
            await websocket.close(code=1008, reason="Invalid init message")
            return

        session = ChatSession(websocket, user_id)
        ACTIVE_SESSIONS[user_id] = session
        await session.initialize()

        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "send_message":
                await session.handle_user_query(data.get("text", ""))

    except ConnectionClosed:
        logger.info(f"WebSocket closed for user {user_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        if user_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[user_id]
            logger.info(f"Session removed for user {user_id}")


# -------------------------------------------------------
# Unified Main
# -------------------------------------------------------
async def start_servers():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    # Start HTTP in a thread
    def start_http():
        server = HTTPServer((HOST, PORT), CustomHandler)
        logger.info(f"HTTP server listening on http://{HOST}:{PORT}")
        server.serve_forever()

    http_thread = threading.Thread(target=start_http, daemon=True)
    http_thread.start()

    # WebSocket server on same port using upgrade requests
    ws_server = await ws_serve(websocket_handler, HOST, PORT, process_request=None)
    logger.info(f"WebSocket server on ws://{HOST}:{PORT}")
    await ws_server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(start_servers())
    except KeyboardInterrupt:
        logger.info("Server stopped.")