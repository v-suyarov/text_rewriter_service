from llm_model import QwenLLM


class QwenTagPredictor:
    def __init__(self, shared_llm: QwenLLM):
        self.llm = shared_llm

    def predict_tags(self, text: str, available_tags: list[str]) -> list[str]:
        tag_list = ", ".join(available_tags)
        user_prompt = (
            f"Прочитай текст ниже и выбери от одного до двух тегов, которые наилучшим образом его описывают. "
            f"Допустимы только следующие теги: {tag_list}. Верни только список тегов через запятую, без лишних слов.\n\n"
            f"Текст:\n{text.strip()}"
        )
        raw_output = self.llm.generate(user_prompt)
        tags = [t.strip() for t in raw_output.split(",") if t.strip()]
        return [t for t in tags if t in available_tags]
