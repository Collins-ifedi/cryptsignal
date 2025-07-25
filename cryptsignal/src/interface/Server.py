# server.py
import os
import asyncio
import websockets
import json
import logging
from http.server import SimpleHTTPRequestHandler
import socketserver
import threading
import sys
from pathlib import Path
from typing import Dict, Optional

# --- Universal Path Setup ---
# This ensures the project's root directory is on the Python path,
# making imports robust regardless of the execution directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now that the path is set, we can reliably import our handler
from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# --- Configuration ---
HOST = "0.0.0.0"  # Listen on all available network interfaces
WS_PORT = int(os.environ.get("PORT", 8000))   # Port for WebSocket connections
HTTP_PORT = 8001  # Port for serving generated images
MEDIA_DIR = PROJECT_ROOT / "generated_media"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChatServer")

# --- Global State ---
# This dictionary will hold active, user-specific WebSocket sessions.
ACTIVE_SESSIONS: Dict[int, "ChatSession"] = {}
main_event_loop: asyncio.AbstractEventLoop | None = None


async def process_api_query(query: str, user_id: int) -> str:
    """
    Creates a user-specific NLP handler, initializes it, processes the query,
    and returns the response. This ensures user isolation for every API request.
    """
    logger.info(f"Creating handler and processing API query for User ID: {user_id}")
    try:
        handler_instance = TradingAssistantNLPHandler(user_id=user_id)
        await handler_instance.initialize()
        response_text, _ = await handler_instance.handle_query(query)
        return response_text
    except Exception as e:
        logger.error(f"Error in handler for user {user_id}: {e}", exc_info=True)
        # Return a JSON string with error details for the server to handle
        return json.dumps({'error': 'An internal error occurred in the NLP handler.'})

# --- HTTP Server for Images and API ---
def run_http_server():
    """Serves files and handles API requests."""
    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            # Serve files from the project root to make generated_media accessible
            super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            """Handles POST requests for AI query processing."""
            if self.path == '/api/generate':
                if not main_event_loop:
                    self.send_response(503, "Service Unavailable")
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'NLP service is not ready.'}).encode('utf-8'))
                    return

                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    body = json.loads(post_data)
                    query = body.get("query")
                    user_id = body.get("userId")

                    if not query or user_id is None:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Query and userId are required.'}).encode('utf-8'))
                        return

                    # Run the user-specific async handler from this sync thread
                    future = asyncio.run_coroutine_threadsafe(
                        process_api_query(query, int(user_id)),
                        main_event_loop
                    )
                    response_text = future.result(timeout=60) # 60-second timeout

                    # Check if the handler returned a JSON error string
                    try:
                        data = json.loads(response_text)
                        if 'error' in data:
                            self.send_response(500)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(data).encode('utf-8'))
                            return
                    except json.JSONDecodeError:
                        # This is the expected path for a successful, non-error response
                        pass

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response_payload = {'response': response_text}
                    self.wfile.write(json.dumps(response_payload).encode('utf-8'))

                except asyncio.TimeoutError:
                    logger.error("Request timed out while processing in NLP handler.")
                    self.send_response(408, "Request Timeout")
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Request timed out.'}).encode('utf-8'))
                except Exception as e:
                    logger.error(f"Error processing POST request: {e}", exc_info=True)
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Internal server error'}).encode('utf-8'))
            else:
                # Fallback to serving files for GET requests or other paths
                super().do_GET()


    with socketserver.TCPServer(("", HTTP_PORT), CORSRequestHandler) as httpd:
        logger.info(f"Serving images and API from '{PROJECT_ROOT}' on http://{HOST}:{HTTP_PORT}")
        httpd.serve_forever()

# --- WebSocket Server Logic ---
class ChatSession:
    """Manages a single user's WebSocket connection and conversation state."""
    def __init__(self, websocket, user_id: int):
        self.websocket = websocket
        self.user_id = user_id
        # Each session gets its own isolated handler instance, keyed by user_id.
        self.nlp_handler = TradingAssistantNLPHandler(user_id=self.user_id)
        # Conversation state like `full_response` is now encapsulated within the `nlp_handler`.

    async def initialize(self):
        """Initializes the NLP handler and sends the dynamic welcome message."""
        await self.nlp_handler.initialize()
        welcome_message = await self.nlp_handler.get_dynamic_welcome()
        await self.send_final_message(welcome_message, "assistant")

    async def send_final_message(self, text: str, sender: str):
        """Sends a complete, JSON-formatted message to the client."""
        if not self.websocket.closed:
            await self.websocket.send(json.dumps({
                "type": "new_message",
                "sender": sender,
                "text": text,
                "timestamp": asyncio.get_event_loop().time()
            }))

    async def handle_user_query(self, query_text: str):
        """Processes a user's query via the isolated NLP handler and sends the response."""
        response, should_exit = await self.nlp_handler.handle_query(query_text)
        await self.send_final_message(response, "assistant")
        if should_exit:
            await self.websocket.close(reason="User requested exit.")

async def handler(websocket, path):
    """
    Main WebSocket connection handler.
    Manages session creation, message routing, and cleanup for each user.
    """
    logger.info(f"Client connected from {websocket.remote_address}")
    session: Optional[ChatSession] = None
    user_id: Optional[int] = None

    try:
        # 1. Wait for an initialization message with the user's ID
        init_message = await asyncio.wait_for(websocket.recv(), timeout=10)
        init_data = json.loads(init_message)

        user_id = init_data.get("userId")
        if not user_id or init_data.get("type") != "init":
            await websocket.close(code=1008, reason="Invalid initialization: 'userId' and 'type: init' are required.")
            logger.warning(f"Connection from {websocket.remote_address} closed due to invalid init message.")
            return

        user_id = int(user_id)
        logger.info(f"Initializing session for User ID: {user_id}")

        # 2. Create and register a new, user-specific session
        # If a session for this user already exists, it will be replaced by this new one.
        session = ChatSession(websocket, user_id=user_id)
        ACTIVE_SESSIONS[user_id] = session

        await session.initialize()

        # 3. Listen for and handle incoming messages for this session
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "send_message":
                query = data.get("text", "").strip()
                if query:
                    logger.info(f"Received from User ID {user_id}: {query}")
                    await session.handle_user_query(query)

    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
        if user_id:
            logger.info(f"Client for User ID {user_id} disconnected. Reason: {e}")
        else:
            logger.info(f"Client {websocket.remote_address} disconnected before initialization.")
    except Exception as e:
        logger.error(f"An unexpected error occurred with client {websocket.remote_address} (User ID: {user_id}): {e}", exc_info=True)
        if session:
            # Inform the client about the server-side error
            error_message = f"An internal server error occurred: {type(e).__name__}"
            await session.send_final_message(error_message, "system")
    finally:
        # 4. Clean up the session on disconnect to free resources
        if user_id and user_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[user_id]
            logger.info(f"Cleaned up session for User ID: {user_id}. Active sessions: {len(ACTIVE_SESSIONS)}")


async def main():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    logger.info("Starting servers. NLP Handlers will be created on-demand per user session.")

    # Start the HTTP server in a separate thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # Start the WebSocket server
    async with websockets.serve(handler, HOST, WS_PORT):
        logger.info(f"WebSocket server started on ws://{HOST}:{WS_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())