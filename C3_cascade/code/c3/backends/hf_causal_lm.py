# -*- coding: utf-8 -*-
"""Local HuggingFace CausalLM backend wrappers."""

from __future__ import annotations


class LocalCausalLMBackend:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int,
        torch_module=None,
        transformers_module=None,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.torch = torch_module
        self.transformers = transformers_module
        self.tokenizer = None
        self.model = None
        self.device = None

    def generate_text(self, prompt: str) -> str:
        self._load()
        inputs, input_ids = self._encode_prompt(prompt)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        output_ids = generated[0][input_ids.shape[1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _encode_prompt(self, prompt: str):
        if getattr(self.tokenizer, "chat_template", None) and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return self._prepare_model_inputs(encoded)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        return self._prepare_model_inputs(encoded)

    def _prepare_model_inputs(self, encoded):
        inputs = encoded.to(self.model.device) if hasattr(encoded, "to") else encoded
        if hasattr(inputs, "keys") and "input_ids" in inputs:
            return inputs, inputs["input_ids"]
        if hasattr(inputs, "input_ids"):
            return inputs, inputs.input_ids
        return {"input_ids": inputs}, inputs

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.torch = self.torch or __import__("torch")
        self.transformers = self.transformers or __import__("transformers")
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
        tokenizer_class = getattr(self.transformers, "AutoTokenizer")
        model_class = getattr(self.transformers, "AutoModelForCausalLM")
        self.tokenizer = tokenizer_class.from_pretrained(self.model_path)
        self.model = model_class.from_pretrained(self.model_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()


class LocalCorrectionBackend(LocalCausalLMBackend):
    def __init__(self, model_path: str, max_new_tokens: int = 128, torch_module=None, transformers_module=None):
        super().__init__(model_path, max_new_tokens, torch_module=torch_module, transformers_module=transformers_module)

    def correct(self, prompt: str) -> str:
        return self.generate_text(prompt)


class LocalTranslationBackend(LocalCausalLMBackend):
    def __init__(self, model_path: str, max_new_tokens: int = 256, torch_module=None, transformers_module=None):
        super().__init__(model_path, max_new_tokens, torch_module=torch_module, transformers_module=transformers_module)

    def translate(self, prompt: str) -> str:
        return self.generate_text(prompt)
