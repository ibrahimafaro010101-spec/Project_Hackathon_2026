# modules/nlp_engine.py
import re

class NLOEngine:
    """
    NLQ Engine LIGHT – 100% offline – Hackathon safe
    """

    def __init__(self):
        self.patterns = {
            "renewal": r"(renouvel|probabilit)",
            "risk": r"(risque|résili|lapse)"
        }

    def parse_query(self, query: str):
        query = query.lower()

        if re.search(self.patterns["renewal"], query):
            return {"intent": "renewal_probability", "confidence": 0.8, "entities": {}}

        if re.search(self.patterns["risk"], query):
            return {"intent": "lapse_risk", "confidence": 0.7, "entities": {}}

        return {"intent": "unknown", "confidence": 0.3, "entities": {}}

    def generate_response(self, intent, value=None):
        if intent == "renewal_probability":
            return f"Probabilité estimée de renouvellement : {value:.1%}"
        if intent == "lapse_risk":
            return f"Risque estimé de résiliation : {value:.1%}"
        return "Je n’ai pas compris la question."
