import logging
import re
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIIGuard")

class PIIGuard:
    def __init__(self):
        # 1. Configure the NLP engine to use en_core_web_md
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # 2. Define custom Aadhaar Number recognizer
        aadhaar_pattern = Pattern(
            name="aadhaar_pattern",
            regex=r"\b\d{4}\s?\d{4}\s?\d{4}\b",
            score=0.85
        )
        aadhaar_recognizer = PatternRecognizer(
            supported_entity="AADHAAR_NUMBER",
            patterns=[aadhaar_pattern],
            context=["aadhaar", "id", "identification", "uidai"]
        )

        # 3. Define custom Salary recognizer (Currency patterns + numbers)
        salary_pattern = Pattern(
            name="salary_pattern",
            regex=r"(?i)(salary|ctc|pay|compensation|income|remuneration)\b.{0,20}\b\d{5,8}(\.\d{2})?\b",
            score=0.7
        )
        salary_recognizer = PatternRecognizer(
            supported_entity="SALARY",
            patterns=[salary_pattern],
            context=["salary", "ctc", "pay", "rupees", "inr"]
        )

        # 4. Define custom Performance Rating recognizer
        perf_pattern = Pattern(
            name="perf_pattern",
            regex=r"(?i)(performance|rating|review period|last review)\b.{0,20}\b[1-5](\.0)?\b",
            score=0.6
        )
        perf_recognizer = PatternRecognizer(
            supported_entity="PERFORMANCE_INFO",
            patterns=[perf_pattern],
            context=["rating", "score", "performance", "review"]
        )

        # 5. Initialize AnalyzerEngine with the custom nlp_engine and add custom recognizers
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
        self.analyzer.registry.add_recognizer(aadhaar_recognizer)
        self.analyzer.registry.add_recognizer(salary_recognizer)
        self.analyzer.registry.add_recognizer(perf_recognizer)
        
        self.anonymizer = AnonymizerEngine()
        
        # Target entities for our guardrails
        self.target_entities = [
            "PERSON", 
            "EMAIL_ADDRESS", 
            "PHONE_NUMBER", 
            "CREDIT_CARD", 
            "AADHAAR_NUMBER",
            "SALARY",
            "PERFORMANCE_INFO"
        ]

    def _get_operators(self, analyzer_results: List):
        """
        Creates custom operators for pseudonymization (e.g., <PERSON_1>, <EMAIL_1>).
        """
        operators = {}
        # Track counts for each entity type to create unique placeholders
        entity_counts = {}
        
        # Sort results by position to maintain order for numbering
        sorted_results = sorted(analyzer_results, key=lambda x: x.start)
        
        # We need to map each specific result to its numbered pseudonym
        # Presidio allows passing a map of results to specific operators or just entity-level operators.
        # However, for sequential numbering, it's easier to iterate and build the map.
        
        for res in sorted_results:
            if res.entity_type not in entity_counts:
                entity_counts[res.entity_type] = 0
            
            # This is tricky because Presidio's AnonymizerEngine.anonymize 
            # takes a dict of entity_type -> OperatorConfig.
            # To get unique IDs per instance, we can't easily use the built-in anonymizer 
            # unless we use a custom operator class.
            pass
            
        return operators

    def _anonymize_sequentially(self, text: str, analyzer_results: List) -> str:
        """
        Custom anonymization that handles sequential numbering like <PERSON_1>, <PERSON_2>.
        """
        # Sort results from last to first to avoid offset issues
        sorted_results = sorted(analyzer_results, key=lambda x: x.start, reverse=True)
        
        # Track counts (per entity type) in reverse order of discovery (meaning first in text)
        # We'll first count them all in order, then re-apply in reverse.
        entity_ids = {}
        processed_results = []
        
        # Get in-order counts
        in_order_results = sorted(analyzer_results, key=lambda x: x.start)
        counts = {}
        for res in in_order_results:
            if res.entity_type not in counts:
                counts[res.entity_type] = 0
            counts[res.entity_type] += 1
            # Store the ID for this specific result
            res.sequential_id = counts[res.entity_type]
            processed_results.append(res)
        
        # Apply in reverse order to text
        sanitized_text = text
        for res in sorted(processed_results, key=lambda x: x.start, reverse=True):
            placeholder = f"<{res.entity_type}_{res.sequential_id}>"
            sanitized_text = sanitized_text[:res.start] + placeholder + sanitized_text[res.end:]
            
        return sanitized_text

    def sanitize_input(self, user_query: str) -> str:
        """
        Detects and redacts PII in user input.
        """
        logger.info(f"Sanitizing input query...")
        
        results = self.analyzer.analyze(
            text=user_query, 
            entities=self.target_entities, 
            language='en'
        )
        
        for res in results:
            logger.info(f"Detected {res.entity_type} with confidence {res.score:.2f} at [{res.start}:{res.end}]")

        # Pseudonymization logic
        return self._anonymize_sequentially(user_query, results)

    def sanitize_output(self, llm_response: str) -> str:
        """
        Detects and anonymizes PII in LLM output.
        """
        logger.info(f"Sanitizing LLM output...")
        
        results = self.analyzer.analyze(
            text=llm_response, 
            entities=self.target_entities, 
            language='en'
        )
        
        for res in results:
            logger.info(f"Detected {res.entity_type} with confidence {res.score:.2f} at [{res.start}:{res.end}]")

        # Pseudonymization logic
        return self._anonymize_sequentially(llm_response, results)

# Singleton instance
guard = PIIGuard()

def sanitize_input(user_query: str) -> str:
    return guard.sanitize_input(user_query)

def sanitize_output(llm_response: str) -> str:
    return guard.sanitize_output(llm_response)

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "My name is Poorva and my email is poorva@example.com.",
        "Call me at +91-9876543210 or check my card 4111-2222-3333-4444.",
        "My Aadhaar number is 1234 5678 9012.",
        "Can you help Rahul (rahul.sharma@gmail.com) with his account?"
    ]
    
    print("--- PII Guardrail Testing ---")
    for query in test_queries:
        print(f"\nOriginal: {query}")
        sanitized = sanitize_input(query)
        print(f"Sanitized: {sanitized}")

    llm_sample = "Sure, I have updated the record for Poorva (poorva@example.com). Her ID is 1234 5678 9012."
    print(f"\nLLM Output: {llm_sample}")
    print(f"Sanitized LLM Output: {sanitize_output(llm_sample)}")
