# live_voice_agent.py
import os
import asyncio
import threading
import io
import pygame
from google import genai
from google.genai import types
from src.rag.rag_tools import search_knowledge_base
from dotenv import load_dotenv

load_dotenv() # This loads the variables from the .env file

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LIVE_MODEL = "gemini-3.1-flash-live-preview"

class GeminiLiveAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
        self.session = None
        self.is_active = False
        self._loop = None
        
        # Initialize Pygame for audio output
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)

    async def _run_session(self):
        tools = [search_knowledge_base]
        config = {"tools": tools}

        try:
            async with self.client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                self.session = session
                self.is_active = True
                print("🚀 Gemini Live WebSocket Connected.")

                async for message in session.receive():
                    # CASE 1: Gemini is SPEAKING (TTS is handled by Gemini)
                    if message.data and message.data.audio:
                        audio_bytes = message.data.audio
                        # Play the audio chunk immediately
                        audio_stream = io.BytesIO(audio_bytes)
                        pygame.mixer.music.load(audio_stream)
                        pygame.mixer.music.play()

                    # CASE 2: Gemini is USING A TOOL (RAG)
                    elif message.tool_call:
                        for call in message.tool_call.function_calls:
                            query_arg = call.args.get("query")
                            result = search_knowledge_base(query_arg)
                            
                            await session.send(
                                input=types.LiveClientToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(name=call.name, response={"result": result})
                                    ]
                                ),
                                end_of_turn=True
                            )
                            print(f"[TOOL] Sent RAG results back to Gemini.")

                    # CASE 3: Gemini is Transcribing (STT is handled by Gemini)
                    elif message.text:
                        print(f"[GEMINI TRANSCRIPT]: {message.text}")

        except Exception as e:
            print(f"❌ Session Error: {e}")
        finally:
            self.is_active = False
            print("🛑 Gemini Session Closed.")

    def start_live_session(self):
        def _thread_target():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_session())
        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()
        return True

    def stop_live_session(self):
        self.is_active = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        pygame.mixer.music.stop()

    def send_instruction(self, instruction: str):
        if self._loop and self.is_active:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._send_initial_instruction(instruction))
            )

    async def _send_initial_instruction(self, instruction: str):
        if self.session and self.is_active:
            await self.session.send(input=instruction, end_of_turn=True)

_agent_instance = GeminiLiveAgent()

def init_live_agent(): return True
def start_live_session(): return _agent_instance.start_live_session()
def stop_live_session(): return _agent_instance.stop_live_session()
def is_session_active(): return _agent_instance.is_active
def send_instruction(instruction: str): return _agent_instance.send_instruction(instruction)