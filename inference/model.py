import time

from mlx_vlm import generate, load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


MODEL_ID = "mlx-community/gemma-3-4b-it-qat-4bit"


class GemmaService:
    def __init__(self) -> None:
        print(f"Loading model: {MODEL_ID}")

        self.model, self.processor = load(MODEL_ID)
        self.config = load_config(MODEL_ID)

        print("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.2,
    ) -> tuple[str, float]:

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=0,
        )

        start = time.perf_counter()

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )

        elapsed = time.perf_counter() - start

        return result.text, elapsed

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.2,
    ) -> tuple[str, float]:

        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        start = time.perf_counter()

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )

        elapsed = time.perf_counter() - start

        return result.text, elapsed

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

        for result in stream_generate(
            self.model,
            self.processor,
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield result.text

gemma_service = GemmaService()
