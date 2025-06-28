"""
LLM Parser Module - AI-Powered Data Extraction
Handles LLM-based parsing using Ollama for complex healthcare document extraction
"""

import json
import re
import logging
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import data structures
from .data_structures import PatientData, OrderData, EpisodeDiagnosis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.output_parsers import OutputFixingParser
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    _LANGCHAIN_AVAILABLE = True
except ModuleNotFoundError:
    _LANGCHAIN_AVAILABLE = False

class LLMParser:
    """Advanced LLM-based parser using Ollama for complex extractions"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "phi", fast_mode: bool = True):
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Speed optimizations - increased timeouts to prevent failures
        self.fast_mode = fast_mode  # Configurable fast extraction mode
        # Longer timeouts for slow mode
        self.timeout = 45 if fast_mode else 90
        
        # Prompt length: use more context in slow mode
        self.prompt_len = 4000 if fast_mode else 12000
        # Max tokens
        self.num_predict = 512 if fast_mode else 1024
        
        # Test connection on initialization
        if not self._test_connection():
            self.logger.warning("⚠️ Cannot connect to Ollama - LLM parsing will be disabled")
            self.available = False
        else:
            mode_text = "FAST MODE" if fast_mode else "STANDARD MODE"
            self.logger.info(f"✅ Connected to Ollama at {ollama_url} using model {model_name} ({mode_text})")
            self.available = True
        
        # LangChain Chat wrapper
        if self.available and _LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOllama(
                    model_name=self.model_name,
                    base_url=self.ollama_url,
                    temperature=0.1,
                    max_tokens=self.num_predict,
                    request_timeout=self.timeout,
                )
                parser = JsonOutputParser()
                self._safe_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)
                self.langchain_ready = True
            except Exception as e:
                self.logger.warning(f"LangChain ChatOllama init failed: {e}")
                self.langchain_ready = False
        else:
            self.langchain_ready = False
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Generate extraction prompt using configurable context length."""

        snippet = text[: self.prompt_len]
        prompt = f"""You are a medical data-extraction expert. Extract the following fields from the document text below.

IMPORTANT:
• Respond with a single JSON object – no code fences, no extra commentary.
• If a field is missing, use an empty string."""

        prompt += f"\n\n--- DOCUMENT START (truncated) ---\n{snippet}\n--- DOCUMENT END ---\n"

        prompt += "\nReturn JSON with ONLY these keys:\n{\n    \"patient_fname\": \"\",\n    \"patient_lname\": \"\",\n    \"dob\": \"\",\n    \"patient_sex\": \"\",\n    \"medical_record_no\": \"\",\n    \"order_date\": \"\",\n    \"order_no\": \"\",\n    \"episode_start_date\": \"\",\n    \"episode_end_date\": \"\",\n    \"start_of_care\": \"\",\n    \"physician_name\": \"\",\n    \"primary_diagnosis\": \"\"\n}"
        return prompt
    
    def _query_llm(self, prompt: str, doc_id: str = "unknown") -> Optional[Dict]:
        """Query Ollama via LangChain; returns parsed dict or None."""

        if not self.available or not self.langchain_ready:
            return None
        
        max_retries = 2 if self.fast_mode else 3
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.logger.debug(f"🔄 LLM retry {attempt} for {doc_id}")

                raw_response = self.llm([
                    HumanMessage(content=prompt)
                ]).content

                try:
                    parsed = self._safe_parser.parse(raw_response)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as pe:
                    self.logger.debug(f"Parser failed on attempt {attempt}: {pe}")

            except Exception as err:
                self.logger.warning(f"LLM call failed on attempt {attempt}: {err}")
            # loop continues if more retries allowed
        return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response"""
        if not response:
            return None
        
        try:
            # Try to find JSON in the response
            json_patterns = [
                r'\{.*\}',  # Basic JSON pattern
                r'```json\s*(\{.*\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*\})\s*```'  # JSON in code blocks without json label
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean up the JSON string
                        json_str = match.strip()
                        if not json_str.startswith('{'):
                            continue
                        
                        # Parse JSON
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # If no patterns work, try parsing the entire response
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                pass
            
            self.logger.warning("Could not extract valid JSON from LLM response")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {e}")
            return None
    
    def parse_combined_with_llm(self, text: str, doc_id: str = "unknown") -> Tuple[PatientData, OrderData]:
        """Parse both patient and order data in a single fast LLM call"""
        self.logger.info(f"🚀 Fast LLM parsing for {doc_id}")
        
        if not self.available:
            return PatientData(), OrderData()
        
        try:
            # Use fast extraction prompt
            prompt = self._create_extraction_prompt(text)

            extracted_data = self._query_llm(prompt, doc_id)
            
            if not extracted_data:
                return PatientData(), OrderData()
            
            # Create patient data
            patient_data = PatientData()
            patient_data.patient_fname = extracted_data.get('patient_fname', '').strip()
            patient_data.patient_lname = extracted_data.get('patient_lname', '').strip()
            patient_data.dob = extracted_data.get('dob', '').strip()
            patient_data.patient_sex = extracted_data.get('patient_sex', '').strip()
            patient_data.medical_record_no = extracted_data.get('medical_record_no', '').strip()
            
            # Create order data
            order_data = OrderData()
            order_data.order_date = extracted_data.get('order_date', '').strip()
            order_data.order_no = extracted_data.get('order_no', '').strip()
            order_data.episode_start_date = extracted_data.get('episode_start_date', '').strip()
            order_data.episode_end_date = extracted_data.get('episode_end_date', '').strip()
            order_data.physician_name = extracted_data.get('physician_name', '').strip()
            order_data.primary_diagnosis = extracted_data.get('primary_diagnosis', '').strip()
            
            self.logger.info(f"✅ Fast LLM parsing completed for {doc_id}")
            return patient_data, order_data
            
        except Exception as e:
            self.logger.error(f"Error in fast LLM parsing for {doc_id}: {e}")
            return PatientData(), OrderData()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        if not self.available:
            return []
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    def set_model(self, model_name: str) -> bool:
        """Set the model to use for extraction"""
        available_models = self.get_available_models()
        if model_name in available_models:
            self.model_name = model_name
            self.logger.info(f"Model set to: {model_name}")
            return True
        else:
            self.logger.warning(f"Model {model_name} not available. Available models: {available_models}")
            return False
    
    def parse_patient_with_llm(self, text: str, doc_id: str = "unknown") -> PatientData:
        """Parse patient data using LLM (backward compatibility)"""
        patient_data, _ = self.parse_combined_with_llm(text, doc_id)
        return patient_data
    
    def parse_order_with_llm(self, text: str, doc_id: str = "unknown") -> OrderData:
        """Parse order data using LLM (backward compatibility)"""
        _, order_data = self.parse_combined_with_llm(text, doc_id)
        return order_data
    
    def parse_episode_dates(self, text: str, doc_id: str = "unknown") -> Tuple[str, str]:
        """Light-weight LLM call that extracts only episode start/end dates."""
        if not self.available:
            return "", ""
        try:
            prompt = (
                "You are given OCR extracted text from a home-health order. "
                "Return ONLY valid JSON containing the two keys shown below. "
                "If a date is missing or uncertain return an empty string.\n\n"
                "JSON schema example:\n"
                "{\n  \"episode_start_date\": \"MM/DD/YYYY\",\n  \"episode_end_date\": \"MM/DD/YYYY\"\n}\n\n"
                "Text:\n---\n" + text[:4000] + "\n---")  # cap to 4k chars
            response = self._query_llm(prompt, doc_id)
            extracted = self._extract_json_from_response(response) if response else None
            if extracted:
                return extracted.get('episode_start_date', '').strip(), extracted.get('episode_end_date', '').strip()
            return "", ""
        except Exception as e:
            self.logger.error(f"Episode date LLM extraction error for {doc_id}: {e}")
            return "", "" 