import asyncio
import io
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

_llm_executor = ThreadPoolExecutor(max_workers=2)

class LLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    @property
    @abstractmethod
    def provider(self) -> str: ...
    @abstractmethod
    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str: ...

class GeminiClient(LLMClient):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 없습니다.")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    @property
    def model_name(self): return self._model_name
    @property
    def provider(self): return "gemini"

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_llm_executor, self._analyze_sync, image_bytes, prompt)

    def _analyze_sync(self, image_bytes: bytes, prompt: str) -> str:
        import PIL.Image
        model = self._genai.GenerativeModel(self._model_name)
        image = PIL.Image.open(io.BytesIO(image_bytes))
        return model.generate_content([prompt, image]).text

class OllamaClient(LLMClient):
    def __init__(self):
        self._model_name = os.getenv("OLLAMA_MODEL", "llava")
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def model_name(self): return self._model_name
    @property
    def provider(self): return "ollama"

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_llm_executor, self._analyze_sync, image_bytes, prompt)

    def _analyze_sync(self, image_bytes: bytes, prompt: str) -> str:
        import ollama
        client = ollama.Client(host=self._base_url)
        response = client.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt, "images": [image_bytes]}]
        )
        return response["message"]["content"]

def get_llm_client(provider: str) -> LLMClient:
    match provider:
        case "gemini": return GeminiClient()
        case "ollama": return OllamaClient()
        case _: raise ValueError(f"알 수 없는 provider: '{provider}'")
