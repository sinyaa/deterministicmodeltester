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
        
    def _should_retry_empty_response(self, result: GenerateResult, attempt: int) -> bool:
        """Retry on empty responses for OpenAI Realtime API calls."""
        return not result.text or result.text.strip() == ""
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            print(f"Received message: {data}")  # Debug output
            
            # Handle different message types from the realtime API
            if data.get("type") == "session.created":
                # Session created, store session ID
                self.session_id = data.get("session", {}).get("id")
                print(f"Session created with ID: {self.session_id}")
            elif data.get("type") == "response.delta":
                # Accumulate response text
                if "delta" in data and "content" in data["delta"]:
                    self.response_text += data["delta"]["content"]
            elif data.get("type") == "response.content_block.delta":
                # Handle content block deltas
                if "delta" in data and "text" in data["delta"]:
                    self.response_text += data["delta"]["text"]
            elif data.get("type") == "response.done":
                # Response is complete
                self.response_received = True
            elif data.get("type") == "error":
                # Handle errors
                self.response_error = data.get("error", {}).get("message", "Unknown error")
                self.response_received = True
                
        except json.JSONDecodeError as e:
            self.response_error = f"Failed to parse WebSocket message: {str(e)}"
            self.response_received = True
        except Exception as e:
            self.response_error = f"Error processing WebSocket message: {str(e)}"
            self.response_received = True
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        self.response_error = f"WebSocket error: {str(error)}"
        self.response_received = True
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        if not self.response_received and not self.response_error:
            self.response_error = "WebSocket connection closed unexpectedly"
            self.response_received = True
    
    def _on_open(self, ws):
        """Handle WebSocket open."""
        # Send session update message to initialize the session
        session_message = {
            "type": "session.update",
            "model": self.model_name,
            "instructions": "You are a helpful AI assistant. Respond to user messages clearly and concisely."
        }
        print(f"Sending session message: {session_message}")
        ws.send(json.dumps(session_message))
    
    def _generate_single(self, prompt: str) -> GenerateResult:
        """Single generation attempt using the OpenAI Realtime WebSocket API."""
        try:
            # Reset state
            self.response_received = False
            self.response_text = ""
            self.response_error = None
            
            # Create WebSocket connection
            ws_url = f"{self.endpoint}?model={self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Create WebSocket with custom headers
            self.websocket = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.websocket.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for session to be created
            time.sleep(1)
            
            # Send the user message
            user_message = {
                "type": "conversation.item.create",
                "session": self.session_id,
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            }
            
            if self.websocket:
                self.websocket.send(json.dumps(user_message))
            
            # Send response creation request
            response_message = {
                "type": "response.create",
                "session": self.session_id
            }
            
            print(f"Sending response message: {response_message}")
            if self.websocket:
                self.websocket.send(json.dumps(response_message))
            
            # Wait for response with timeout
            start_time = time.time()
            while not self.response_received and (time.time() - start_time) < self.timeout_s:
                time.sleep(0.1)
            
            # Close WebSocket connection
            if self.websocket:
                self.websocket.close()
            
            # Check for timeout
            if not self.response_received and not self.response_error:
                return GenerateResult(
                    text="",
                    model=self.model_name,
                    error=f"Request timeout after {self.timeout_s} seconds"
                )
            
            # Return result
            if self.response_error:
                return GenerateResult(
                    text="",
                    model=self.model_name,
                    error=self.response_error
                )
            else:
                return GenerateResult(
                    text=self.response_text,
                    model=self.model_name
                )
                
        except Exception as e:
            return GenerateResult(
                text="",
                model=self.model_name,
                error=f"WebSocket connection error: {str(e)}"
            )
