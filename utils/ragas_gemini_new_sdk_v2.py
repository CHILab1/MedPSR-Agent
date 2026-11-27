"""
ragas_gemini_new_sdk_v2.py — Gemini Wrapper for RAGAS (google-genai SDK)
═══════════════════════════════════════════════════════════════════════
Custom wrapper to use Gemini as LLM-as-a-judge with RAGAS,
bypassing LangChain limitations.

Features:
  - Exponential backoff retry for 429 errors (rate limit)
  - Automatic rotation across multiple API keys
  - Token usage tracking for cost analysis
  - Real async support
  - Multiple candidates generation (n > 1) via sequential calls
"""

import asyncio
import logging
import os
import time
from typing import Any, List, Optional

from google import genai
from google.genai import types
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings

logger = logging.getLogger("ragas_gemini")


# ══════════════════════════════════════════════════════════════════════════════
#  Structures compatible with RAGAS
# ══════════════════════════════════════════════════════════════════════════════

class Generation:
    """Single text generation (RAGAS compatible)."""
    def __init__(self, text: str):
        self.text = text
        self.message = None


class LLMResult:
    """Result of an LLM call (RAGAS compatible)."""
    def __init__(self, generations: List[List[Generation]]):
        self.generations = generations


# ══════════════════════════════════════════════════════════════════════════════
#  Token Usage Tracker
# ══════════════════════════════════════════════════════════════════════════════

class TokenTracker:
    """Tracks token usage for cost analysis."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.errors = 0
        self.retries = 0

    def record(self, input_tokens: int = 0, output_tokens: int = 0):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def record_error(self):
        self.errors += 1

    def record_retry(self):
        self.retries += 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "retries": self.retries,
        }

    def cost_estimate(self, model: str = "gemini-2.5-flash") -> dict:
        """Estimates costs based on model."""
        pricing = {
            "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
            "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        }
        # Normalize model name
        model_key = model.lower()
        for key in pricing:
            if key in model_key:
                p = pricing[key]
                input_cost = (self.total_input_tokens / 1_000_000) * p["input"]
                output_cost = (self.total_output_tokens / 1_000_000) * p["output"]
                return {
                    "model": key,
                    "input_cost_usd": round(input_cost, 4),
                    "output_cost_usd": round(output_cost, 4),
                    "total_cost_usd": round(input_cost + output_cost, 4),
                }
        return {"model": model, "note": "Pricing not available for this model"}

    def __repr__(self):
        return (
            f"TokenTracker(calls={self.total_calls}, "
            f"in={self.total_input_tokens:,}, out={self.total_output_tokens:,}, "
            f"total={self.total_tokens:,}, errors={self.errors}, retries={self.retries})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GeminiLLM — Wrapper for RAGAS
# ══════════════════════════════════════════════════════════════════════════════

class GeminiLLM(BaseRagasLLM):
    """
    LLM wrapper for Gemini using google-genai SDK.

    Args:
        model:        Gemini model name (e.g. "gemini-2.5-flash")
        api_key:      Single API key
        api_keys:     List of API keys for rotation (optional)
        max_retries:  Maximum retries for 429 errors
        base_delay:   Initial delay in seconds for exponential backoff
        tracker:      Shared TokenTracker (optional)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        max_retries: int = 5,
        base_delay: float = 5.0,
        tracker: Optional[TokenTracker] = None,
        vertexai: Optional[bool] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.tracker = tracker or TokenTracker()

        # Vertex AI configuration
        use_vertex_env = os.getenv("USE_VERTEXAI", "false").lower() in ("true", "1", "yes")
        self.use_vertex = vertexai if vertexai is not None else use_vertex_env

        if self.use_vertex:
            proj = project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            loc = location or os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION")
            logger.info(f"Initializing Vertex AI LLM client (project={proj}, location={loc})")
            self.client = genai.Client(
                vertexai=True,
                project=proj,
                location=loc
            )
            self._api_keys = []
            self._key_idx = 0
        else:
            # Setup API key(s)
            if api_keys:
                self._api_keys = [k for k in api_keys if k]
            elif api_key:
                self._api_keys = [api_key]
            else:
                key = os.getenv("GEMINI_API_KEY", "")
                self._api_keys = [key] if key else []

            if not self._api_keys:
                raise ValueError("No API key configured!")

            self._key_idx = 0
            self.client = genai.Client(api_key=self._current_key)

        from ragas.run_config import RunConfig
        self.run_config = RunConfig()

    @property
    def _current_key(self) -> str:
        if self.use_vertex:
            return ""
        return self._api_keys[self._key_idx % len(self._api_keys)]

    def _rotate_key(self):
        """Switches to the next API key."""
        if self.use_vertex:
            return
        if len(self._api_keys) > 1:
            self._key_idx = (self._key_idx + 1) % len(self._api_keys)
            self.client = genai.Client(api_key=self._current_key)
            logger.info(f"API key rotation → key #{self._key_idx + 1}")

    def _call_with_retry(self, prompt_str: str, temperature: float = 0.0,
                         stop: Optional[List[str]] = None) -> str:
        """Gemini call with retry and exponential backoff."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_str,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        stop_sequences=stop or [],
                    ),
                )
                text = response.text or ""

                # Track token usage if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    self.tracker.record(
                        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                    )
                else:
                    # Approximate estimate
                    self.tracker.record(
                        input_tokens=len(prompt_str) // 4,
                        output_tokens=len(text) // 4,
                    )

                return text

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                if "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str:
                    self.tracker.record_retry()
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limit (429) attempt {attempt + 1}/{self.max_retries + 1}. "
                        f"Waiting {delay:.0f}s..."
                    )
                    self._rotate_key()
                    time.sleep(delay)
                elif "500" in err_str or "503" in err_str or "internal" in err_str:
                    self.tracker.record_retry()
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"Server error attempt {attempt + 1}/{self.max_retries + 1}. "
                        f"Waiting {delay:.0f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.tracker.record_error()
                    logger.error(f"Non-recoverable error: {e}")
                    raise

        self.tracker.record_error()
        raise RuntimeError(
            f"Max retries ({self.max_retries}) exhausted. Last error: {last_error}"
        )

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        prompt_str = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        
        # Support n > 1 by making n independent calls
        generations = []
        for _ in range(n):
            text = self._call_with_retry(prompt_str, temperature, stop)
            generations.append(Generation(text=text))
            
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        # Run in a separate thread so as not to block the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate_text, prompt, n, temperature, stop, callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  GeminiEmbeddings — Wrapper for RAGAS
# ══════════════════════════════════════════════════════════════════════════════

class GeminiEmbeddings(BaseRagasEmbeddings):
    """
    Embeddings wrapper for Gemini using google-genai SDK.

    Args:
        model:    Embedding model name (e.g. "gemini-embedding-001")
        api_key:  API key
        tracker:  Shared TokenTracker (optional)
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        tracker: Optional[TokenTracker] = None,
        vertexai: Optional[bool] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 5.0,
    ):
        self.model = model
        self.tracker = tracker or TokenTracker()
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Vertex AI configuration
        use_vertex_env = os.getenv("USE_VERTEXAI", "false").lower() in ("true", "1", "yes")
        self.use_vertex = vertexai if vertexai is not None else use_vertex_env

        if self.use_vertex:
            proj = project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            loc = location or os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION")
            logger.info(f"Initializing Vertex AI Embeddings client (project={proj}, location={loc})")
            self.client = genai.Client(
                vertexai=True,
                project=proj,
                location=loc
            )
            self.api_key = ""
        else:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            self.client = genai.Client(api_key=self.api_key)

        from ragas.run_config import RunConfig
        self.run_config = RunConfig()

    def embed_query(self, text: str) -> List[float]:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                return response.embeddings[0].values
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str:
                    self.tracker.record_retry()
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"Embeddings rate limit (429) attempt {attempt + 1}/{self.max_retries + 1}. "
                        f"Waiting {delay:.0f}s..."
                    )
                    time.sleep(delay)
                elif "500" in err_str or "503" in err_str or "internal" in err_str:
                    self.tracker.record_retry()
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"Embeddings server error attempt {attempt + 1}/{self.max_retries + 1}. "
                        f"Waiting {delay:.0f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.tracker.record_error()
                    logger.error(f"Non-recoverable embedding error: {e}")
                    raise
        self.tracker.record_error()
        raise RuntimeError(
            f"Max retries ({self.max_retries}) exhausted for embeddings. Last error: {last_error}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch in chunks of 100 to avoid limits
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            last_error = None
            success = False
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str:
                        self.tracker.record_retry()
                        delay = self.base_delay * (2 ** attempt)
                        logger.warning(
                            f"Embeddings batch rate limit (429) attempt {attempt + 1}/{self.max_retries + 1}. "
                            f"Waiting {delay:.0f}s..."
                        )
                        time.sleep(delay)
                    elif "500" in err_str or "503" in err_str or "internal" in err_str:
                        self.tracker.record_retry()
                        delay = self.base_delay * (2 ** attempt)
                        logger.warning(
                            f"Embeddings batch server error attempt {attempt + 1}/{self.max_retries + 1}. "
                            f"Waiting {delay:.0f}s..."
                        )
                        time.sleep(delay)
                    else:
                        self.tracker.record_error()
                        logger.error(f"Non-recoverable batch embedding error: {e}")
                        raise
            if not success:
                self.tracker.record_error()
                raise RuntimeError(
                    f"Max retries ({self.max_retries}) exhausted for batch embeddings. Last error: {last_error}"
                )
        return all_embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

