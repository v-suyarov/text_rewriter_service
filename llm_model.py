import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen3-8B", max_new_tokens=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        self.max_new_tokens = max_new_tokens
        self.THINK_TOKEN_ID = 151668
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
        }

    def generate(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, **self.generation_config)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(self.THINK_TOKEN_ID)
        except ValueError:
            index = 0

        return self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
