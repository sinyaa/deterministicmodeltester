from __future__ import annotations
import websocket
import json
import threading
import time
from typing import Optional
from .base import BaseConnector, GenerateResult


class OpenAIRealtimeWebSocketConnector(BaseConnector):
    """OpenAI Realtime API connector using WebSocket connections."""
    
    def __init__(self, model_name: str = "gpt-4o-realtime-preview", endpoint: str = "wss://api.openai.com/v1/realtime", 
                 api_key: str = "", timeout_s: int = 30, max_retries: int = 3, retry_delay: float = 10.0):
        super().__init__(model_name, endpoint, api_key, timeout_s, max_retries, retry_delay)
        self.response_received = False
        self.response_text = ""
        self.response_error = None
        self.websocket = None
        self.session_id = None
        self.first_byte_time = None  # Track when first response data arrives
        
    def _should_retry_empty_response(self, result: GenerateResult, attempt: int) -> bool:
        """Retry on empty responses for OpenAI Realtime API calls."""
        return not result.text or result.text.strip() == ""
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            # print(f"📨 Received message type: {data.get('type', 'unknown')} - {data}")  # Debug output
            
            # Handle different message types from the realtime API
            if data.get("type") == "session.created":
                # Session created, store session ID
                self.session_id = data.get("session", {}).get("id")
                # print(f"✅ Session created with ID: {self.session_id}")
            elif data.get("type") == "session.update":
                # print(f"📝 Session update received: {data}")
                pass
            elif data.get("type") == "conversation.item.create":
                # print(f"💬 Conversation item created: {data}")
                pass
            elif data.get("type") == "response.create":
                # print(f"🚀 Response create received: {data}")
                pass
            elif data.get("type") == "response.output_text.delta":
                # Newer text delta event - capture first byte timing
                if self.first_byte_time is None:
                    self.first_byte_time = time.time()
                delta_text = data.get("delta", "")
                self.response_text += delta_text
                # print(f"📝 Text delta: '{delta_text}' | Accumulated: '{self.response_text}'")
            elif data.get("type") == "response.content_block.delta":
                # Older-style content block deltas - capture first byte timing
                if self.first_byte_time is None:
                    self.first_byte_time = time.time()
                delta = data.get("delta", {})
                if isinstance(delta, dict) and "text" in delta:
                    self.response_text += delta["text"]
                    # print(f"📝 Content block delta: '{delta['text']}' | Accumulated: '{self.response_text}'")
            elif data.get("type") == "response.audio_transcript.done":
                # Handle audio responses by extracting transcript - capture first byte timing
                if self.first_byte_time is None:
                    self.first_byte_time = time.time()
                transcript = data.get("transcript", "")
                if transcript:
                    self.response_text = transcript
                    # print(f"🎵 Audio transcript: '{transcript}'")
            elif data.get("type") == "response.audio.delta":
                # Handle audio delta events (ignore for now)
                pass
            elif data.get("type") == "response.audio.done":
                # Handle audio done events (ignore for now)
                pass
            elif data.get("type") in ("response.completed", "response.done"):
                # print(f"✅ Response completed! Final text: '{self.response_text}'")
                self.response_received = True
            elif data.get("type") == "error":
                # Handle errors
                error_msg = data.get("error", {}).get("message", "Unknown error")
                # print(f"❌ Error: {error_msg}")
                self.response_error = error_msg
                self.response_received = True
            else:
                # print(f"❓ Unknown message type: {data.get('type', 'unknown')}")
                pass
                
        except json.JSONDecodeError as e:
            self.response_error = f"Failed to parse WebSocket message: {str(e)}"
            self.response_received = True
            # print(f"Failed to parse WebSocket message: {str(e)}")
        except Exception as e:
            self.response_error = f"Error processing WebSocket message: {str(e)}"
            self.response_received = True
            # print(f"Error processing WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        error_msg = f"WebSocket error: {str(error)}"
        # print(f"❌ WebSocket error: {error_msg}")
        self.response_error = error_msg
        self.response_received = True
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        # print(f"🔌 WebSocket closed - Status: {close_status_code}, Message: {close_msg}")
        if not self.response_received and not self.response_error:
            error_msg = "WebSocket connection closed unexpectedly"
            # print(f"❌ {error_msg}")
            self.response_error = error_msg
            self.response_received = True
    
    def _on_open(self, ws):
        """Handle WebSocket open."""
        # print("🔌 WebSocket connection opened")
        # Send session update message to initialize the session
        # Note: We'll send the actual session update after we get the session ID
        pass
    
    def _generate_single(self, prompt: str) -> GenerateResult:
        """Single generation attempt using the OpenAI Realtime WebSocket API."""
        # print(f"🚀 Starting generation for prompt: '{prompt[:50]}...'")
        try:
            # Reset state
            self.response_received = False
            self.response_text = ""
            self.response_error = None
            self.session_id = None  # ← ensure fresh for each run
            self.first_byte_time = None  # ← reset first byte timing
            # print("🔄 State reset complete")

            ws_url = f"{self.endpoint}?model={self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            # print(f"🌐 Connecting to: {ws_url}")
            # print(f"🔑 Using API key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else '***'}")

            self.websocket = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            # Keepalive helps some networks
            ws_thread = threading.Thread(
                target=lambda: self.websocket.run_forever(ping_interval=20, ping_timeout=10)
            )
            ws_thread.daemon = True
            ws_thread.start()
            # print("🧵 WebSocket thread started")

            # Wait for session.created
            session_timeout = 5
            start_time = time.time()
            # print(f"⏳ Waiting for session creation (timeout: {session_timeout}s)...")
            while not self.session_id and (time.time() - start_time) < session_timeout:
                time.sleep(0.05)

            if not self.session_id:
                # print("❌ Session creation timeout!")
                return GenerateResult(text="", model=self.model_name,
                                      error="Failed to create session within timeout")
            # print(f"✅ Session ready: {self.session_id}")

            # 1) Send the user item
            user_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                }
            }
            # print(f"📤 Sending user message: {user_message}")
            self.websocket.send(json.dumps(user_message))

            # 2) Explicitly trigger inference for text-only flows
            response_create = {
                "type": "response.create"
                # Optionally: "response": {"temperature": 0}
            }
            # print(f"📤 Sending response.create: {response_create}")
            self.websocket.send(json.dumps(response_create))

            # Wait for response
            # print(f"⏳ Waiting for response (timeout: {self.timeout_s}s)...")
            start_time = time.time()
            while not self.response_received and (time.time() - start_time) < self.timeout_s:
                time.sleep(0.05)

            # print(f"🔌 Closing WebSocket connection...")
            # Close and join
            if self.websocket:
                self.websocket.close()
            ws_thread.join(timeout=3)
            # print(f"🧵 WebSocket thread joined")

            # Calculate first-byte latency
            first_byte_latency_ms = None
            if self.first_byte_time is not None:
                first_byte_latency_ms = (self.first_byte_time - start_time) * 1000

            if not self.response_received and not self.response_error:
                # print(f"❌ Request timeout after {self.timeout_s} seconds")
                return GenerateResult(text="", model=self.model_name,
                                      error=f"Request timeout after {self.timeout_s} seconds",
                                      first_byte_latency_ms=first_byte_latency_ms)

            if self.response_error:
                # print(f"❌ Response error: {self.response_error}")
                return GenerateResult(text="", model=self.model_name, error=self.response_error,
                                      first_byte_latency_ms=first_byte_latency_ms)

            # print(f"✅ Success! Final response: '{self.response_text}'")
            return GenerateResult(text=self.response_text, model=self.model_name,
                                  first_byte_latency_ms=first_byte_latency_ms)

        except Exception as e:
            # print(f"❌ Exception in _generate_single: {str(e)}")
            return GenerateResult(text="", model=self.model_name,
                                  error=f"WebSocket connection error: {str(e)}",
                                  first_byte_latency_ms=None)
        finally:
            # print("🧹 Cleaning up session_id")
            self.session_id = None  # ← avoid bleed into next call
