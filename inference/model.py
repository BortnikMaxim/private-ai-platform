import threading
import time
from typing import Any

from mlx_vlm import generate, load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_ID = "mlx-community/gemma-3-4b-it-qat-4bit"


class GemmaService:
    def __init__(self) -> None:
        print(f"Loading model: {MODEL_ID}")

        self.model, self.processor = load(MODEL_ID)
        self.config = load_config(MODEL_ID)
        self.inference_lock = threading.Lock()

        print("Model loaded successfully")

    @staticmethod
    def _usage(result: Any) -> dict[str, Any]:
        """Token counts from the generation result, when the runtime reports them.

        mlx-vlm fills prompt_tokens/generation_tokens/total_tokens from the
        tokenizer it just ran, so these are measured rather than estimated. A
        runtime that does not provide them yields None and the caller reports
        no usage at all instead of guessing.
        """
        prompt_tokens = getattr(result, "prompt_tokens", None)
        completion_tokens = getattr(result, "generation_tokens", None)

        if prompt_tokens is None or completion_tokens is None:
            return {"usage": None, "finish_reason": getattr(result, "finish_reason", None)}

        total = getattr(result, "total_tokens", None)

        return {
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(
                    total if total is not None else prompt_tokens + completion_tokens
                ),
                "source": "local_tokenizer",
            },
            "finish_reason": getattr(result, "finish_reason", None),
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.2,
    ) -> tuple[str, float, dict[str, Any]]:
        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=0,
        )

        start = time.perf_counter()

        with self.inference_lock:
            result = generate(
                self.model,
                self.processor,
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )

        elapsed = time.perf_counter() - start

        return result.text, elapsed, self._usage(result)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.2,
    ) -> tuple[str, float, dict[str, Any]]:
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        start = time.perf_counter()

        with self.inference_lock:
            result = generate(
                self.model,
                self.processor,
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )

        elapsed = time.perf_counter() - start

        return result.text, elapsed, self._usage(result)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.2,
    ):
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        with self.inference_lock:
            for result in stream_generate(
                self.model,
                self.processor,
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                yield result.text


gemma_service = GemmaService()
