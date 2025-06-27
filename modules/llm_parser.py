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

class LLMParser:
    """Advanced LLM-based parser using Ollama for complex extractions"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "phi", fast_mode: bool = True):
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Speed optimizations - increased timeouts to prevent failures
        self.fast_mode = fast_mode  # Configurable fast extraction mode
        self.timeout = 30 if fast_mode else 60  # Increased from 15 to 30 seconds
        
        # Test connection on initialization
        if not self._test_connection():
            self.logger.warning("⚠️ Cannot connect to Ollama - LLM parsing will be disabled")
            self.available = False
        else:
            mode_text = "FAST MODE" if fast_mode else "STANDARD MODE"
            self.logger.info(f"✅ Connected to Ollama at {ollama_url} using model {model_name} ({mode_text})")
            self.available = True
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def _create_fast_extraction_prompt(self, text: str) -> str:
        """Create fast, focused prompt for critical fields only"""
        
        prompt = f"""Extract patient info from this medical document. Return ONLY JSON:

{text[:1500]}

Return JSON with ONLY these critical fields:
{{
    "patient_fname": "first name",
    "patient_lname": "last name", 
    "dob": "date of birth as MM/DD/YYYY",
    "patient_sex": "M or F",
    "medical_record_no": "MRN",
    "order_date": "order date as MM/DD/YYYY",
    "physician_name": "doctor name",
    "primary_diagnosis": "main diagnosis"
}}

Use "" for missing fields. Return ONLY the JSON:"""
        return prompt
    
    def _query_ollama(self, prompt: str, doc_id: str = "unknown") -> Optional[str]:
        """Send prompt to Ollama with speed optimizations and retry logic"""
        if not self.available:
            return None
        
        # Retry logic for better reliability
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 512 if self.fast_mode else 2048  # Shorter responses
                    }
                }
                
                if attempt == 0:
                    self.logger.debug(f"🚀 Fast querying Ollama for document {doc_id}")
                else:
                    self.logger.debug(f"🔄 Retry {attempt} for document {doc_id}")
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout  # Use configurable timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        continue  # Try again
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"⏰ Timeout on attempt {attempt + 1} for {doc_id}")
                if attempt < max_retries - 1:
                    continue  # Try again with next attempt
                self.logger.error(f"❌ All retry attempts failed for {doc_id} due to timeout")
                return None
            except Exception as e:
                self.logger.error(f"Error querying Ollama for {doc_id}: {e}")
                if attempt < max_retries - 1:
                    continue  # Try again
                return None
        
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
            prompt = self._create_fast_extraction_prompt(text)
            response = self._query_ollama(prompt, doc_id)
            
            if not response:
                return PatientData(), OrderData()
            
            # Extract JSON from response
            extracted_data = self._extract_json_from_response(response)
            
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