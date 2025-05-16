import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenTagPredictor:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        print("📦 TagPredictor загружен на:", self.model.device)
        self.max_new_tokens = max_new_tokens
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": self.max_new_tokens,
        }

    def predict_tags(self, text: str, available_tags: list[str]) -> list[str]:
        tag_list = ", ".join(available_tags)
        prompt = (
            f"Прочитай текст, который я оставлю ниже, и выбери от одного "
            f"до двух тегов, которые наилучшим образом его описывают. "
            f"Допустимы только следующие теги: {tag_list}. "
            f"Верни только список выбранных тегов, разделённый запятыми, "
            f"без лишних слов.\n\n"
            f"Текст:\n{text.strip()}"
        )

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, **self.generation_config)[0][len(inputs.input_ids[0]):]

        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        print(raw_output)
        tags = [tag.strip() for tag in raw_output.split(",") if tag.strip()]
        return [tag for tag in tags if tag in available_tags]
