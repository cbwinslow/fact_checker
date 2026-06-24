from langchain_openai import ChatOpenAI

from fact_checker.config.settings import get_settings


def build_chat_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_model,
        temperature=0,
    )
