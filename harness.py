class FactCheckHarness:
    def run(self, source: str) -> dict:
        return {
            "source": source,
            "status": "queued",
        }
