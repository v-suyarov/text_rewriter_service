import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenRewriter:
    THINK_TOKEN_ID = 151668

    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_new_tokens=2000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        print(self.model.hf_device_map)
        self.max_new_tokens = max_new_tokens
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": self.max_new_tokens,
        }
    def rewrite(self, original_text: str, prompt: str = None) -> str:
        user_prompt = (
            f"Ты профессиональный редактор. Изучи следующий текст:\n\n"
            f"{original_text.strip()}\n\n"
            f"На основе изученного текста напиши оригинальный пост "
            f"для телеграмм канала, учитывая следующие требования: "
        )
        if prompt:
            user_prompt += (
                f"{prompt.strip()}\n")
        else:
            user_prompt += "Перепиши профессионально и лаконично.\n"
        prompt += ("В ответе укажи только готовый пост, "
                   "никаких уточнений, размышлений и прочего")
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 🔧 отключаем "мышление"
        )

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_config
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(self.THINK_TOKEN_ID)
        except ValueError:
            index = 0

        rewritten = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return rewritten or original_text
