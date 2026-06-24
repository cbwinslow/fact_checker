from fact_checker.domain.models import Claim
from fact_checker.utils.openrouter import build_chat_model


class ClaimExtractionAgent:
    def __init__(self) -> None:
        self.model = build_chat_model()

    def extract(self, transcript_chunk: str) -> list[Claim]:
        _ = transcript_chunk
        return []
