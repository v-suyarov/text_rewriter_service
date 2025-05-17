from llm_model import QwenLLM


class QwenRewriter:
    def __init__(self, shared_llm: QwenLLM):
        self.llm = shared_llm

    def rewrite(self, original_text: str, prompt: str = "") -> str:
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
        user_prompt += ("В ответе укажи только готовый пост, "
                        "никаких уточнений, размышлений и прочего")
        return self.llm.generate(user_prompt) or original_text
