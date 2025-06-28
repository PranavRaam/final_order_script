"""
Enhanced Field Extraction Module
Specialized patterns for critical healthcare fields that are commonly missing
"""

import re
from typing import Optional, List, Dict, Any
from .data_structures import EpisodeDiagnosis
import logging

class EnhancedFieldExtractor:
    """Enhanced field extraction with specialized patterns for critical missing fields"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_enhanced_patterns()
    
    def _compile_enhanced_patterns(self):
        """Compile enhanced regex patterns for critical fields"""
        
        # Enhanced DOB patterns - more comprehensive
        self.dob_patterns = [
            # Standard formats with labels
            r'(?:DOB|Date\s*of\s*Birth|Birth\s*Date|Born)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:DOB|Date\s*of\s*Birth|Birth\s*Date|Born)\s*:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            
            # Context-based patterns
            r'Patient.*?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}).*?(?:Age|years?\s*old)',
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}).*?(?:Age|years?\s*old)',
            r'Age\s*\d+.*?(?:DOB|Born|Birth)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            
            # Standalone date patterns in patient context
            r'Patient\s*Information.*?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            r'Demographics.*?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            
            # Table-like formats
            r'Birth\s*Date[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            r'D\.?O\.?B\.?\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            
            # More flexible formats
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\s*\([^)]*(?:birth|born|DOB)[^)]*\)',
            r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b(?=.*(?:patient|birth|born|age))',
        ]
        
        # Enhanced SSN patterns
        self.ssn_patterns = [
            r'(?:SSN|Social\s*Security|Social\s*Security\s*Number)\s*:?\s*(\d{3}[-.\\s]?\d{2}[-.\\s]?\d{4})',
            r'(?:SS#|SSN#)\s*:?\s*(\d{3}[-.\\s]?\d{2}[-.\\s]?\d{4})',
            r'Social\s*:?\s*(\d{3}[-.\\s]?\d{2}[-.\\s]?\d{4})',
            r'\b(\d{3}[-.]?\d{2}[-.]?\d{4})\b(?=.*(?:social|ssn|security))',
            r'Security\s*Number\s*:?\s*(\d{3}[-.\\s]?\d{2}[-.\\s]?\d{4})',
        ]
        
        # Enhanced MRN patterns
        self.mrn_patterns = [
            r'(?:MRN|Medical\s*Record\s*Number|Medical\s*Record|Patient\s*ID)\s*:?\s*([A-Za-z0-9\-]{4,15})',
            r'(?:Record\s*Number|Record\s*#|Patient\s*Number)\s*:?\s*([A-Za-z0-9\-]{4,15})',
            r'(?:Chart\s*Number|Chart\s*#)\s*:?\s*([A-Za-z0-9\-]{4,15})',
            r'(?:ID|Identifier)\s*:?\s*([A-Za-z0-9\-]{6,15})',
            r'MR#\s*:?\s*([A-Za-z0-9\-]{4,15})',
            r'Patient.*?ID\s*:?\s*([A-Za-z0-9\-]{4,15})',
            # 9-10 digit MRN in parentheses after patient name, e.g., "Patient: Blair, Edward (100996456)"
            r"Patient:?\s+[A-Za-z,\s]+\s*\((\d{9,10})\)",
            # Name with parentheses digits and no prefix, e.g., "Oliveira, Kathleen (1275520097)"
            r"[A-Z][A-Za-z]+,\s*[A-Z][A-Za-z]+\s*\((\d{9,10})\)",
            # Stand-alone 9-10 digit number inside parentheses (minimal false positives)
            r"\((\d{9,10})\)",
            # "MR Number: 000000200" format
            r"MR\s+Number:?\s*([A-Za-z0-9\-]{6,15})",
            # Patient line with dash MR#, e.g., "Patient: Hamilton, Doris-MR#000000200"
            r"Patient:?[\sA-Za-z,]+-?\s*MR#?\s*([A-Za-z0-9\-]{6,15})",
        ]
        
        # Enhanced episode date patterns
        self.episode_start_patterns = [
            r'(?:Episode\s*Start|Start\s*of\s*Care|SOC|Care\s*Start)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:Start\s*Date|Begin\s*Date|Admission\s*Date)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:From|Starting)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'Care\s*Period\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\s*(?:to|through|-)',
            r'Service\s*Start\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            # "Certification Period 05/02/2025 -- 06/30/2025" first date capture
            r"Certification\s+Period:?\s*(\d{1,2}[/\-\.]{1}\d{1,2}[/\-\.]{1}\d{2,4})",
        ]
        
        self.episode_end_patterns = [
            r'(?:Episode\s*End|End\s*of\s*Care|Care\s*End)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:End\s*Date|Discharge\s*Date|Termination\s*Date)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:To|Until|Through)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'Care\s*Period\s*:?\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s*(?:to|through|-)\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'Service\s*End\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        ]
        
        # Enhanced NPI patterns
        self.npi_patterns = [
            r'(?:NPI|National\s*Provider\s*Identifier)\s*:?\s*(\d{10})',
            r'(?:Provider\s*Number|Provider\s*ID)\s*:?\s*(\d{10})',
            r'NPI#\s*:?\s*(\d{10})',
            r'Physician.*?NPI\s*:?\s*(\d{10})',
            r'Doctor.*?NPI\s*:?\s*(\d{10})',
            r'\b(\d{10})\b(?=.*(?:NPI|provider|physician))',
        ]
        
        # Enhanced diagnosis patterns
        self.diagnosis_patterns = [
            # Primary diagnosis
            r'(?:Primary\s*Diagnosis|Primary\s*Dx|Diagnosis\s*1)\s*:?\s*([^:\n]+?)(?:\n|$|Diagnosis\s*2)',
            r'(?:Principal\s*Diagnosis)\s*:?\s*([^:\n]+?)(?:\n|$)',
            r'(?:Main\s*Diagnosis)\s*:?\s*([^:\n]+?)(?:\n|$)',
            
            # Secondary diagnosis
            r'(?:Secondary\s*Diagnosis|Secondary\s*Dx|Diagnosis\s*2)\s*:?\s*([^:\n]+?)(?:\n|$|Diagnosis\s*3)',
            r'(?:Additional\s*Diagnosis)\s*:?\s*([^:\n]+?)(?:\n|$)',
            
            # General diagnosis patterns
            r'(?:Diagnosis|Dx)\s*:?\s*([^:\n]+?)(?:\n|$)',
            r'(?:ICD-?10?)\s*:?\s*([A-Z]\d{2}(?:\.\d{1,2})?)\s*([^:\n]+?)(?:\n|$)',
            r'(?:ICD-?9?)\s*:?\s*(\d{3}(?:\.\d{1,2})?)\s*([^:\n]+?)(?:\n|$)',
            
            # Condition patterns
            r'(?:Condition|Medical\s*Condition)\s*:?\s*([^:\n]+?)(?:\n|$)',
            r'(?:Problem|Health\s*Problem)\s*:?\s*([^:\n]+?)(?:\n|$)',
        ]
        
        # Enhanced insurance patterns
        self.insurance_patterns = [
            r'(?:Insurance|Primary\s*Insurance|Payer)\s*:?\s*([^:\n]+?)(?:\n|$)',
            r'(?:Coverage|Health\s*Plan)\s*:?\s*([^:\n]+?)(?:\n|$)',
            r'(?:Carrier|Insurance\s*Carrier)\s*:?\s*([^:\n]+?)(?:\n|$)',
            r'(?:Plan|Insurance\s*Plan)\s*:?\s*([^:\n]+?)(?:\n|$)',
        ]
        
        # Compile all patterns
        self._compiled_patterns = {}
        for pattern_name in ['dob_patterns', 'ssn_patterns', 'mrn_patterns', 
                           'episode_start_patterns', 'episode_end_patterns', 
                           'npi_patterns', 'diagnosis_patterns', 'insurance_patterns']:
            patterns = getattr(self, pattern_name)
            self._compiled_patterns[pattern_name] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL) 
                for pattern in patterns
            ]
    
    def extract_dob_enhanced(self, text: str) -> str:
        """Enhanced DOB extraction with multiple strategies"""
        # First try the compiled patterns
        for pattern in self._compiled_patterns['dob_patterns']:
            match = pattern.search(text)
            if match:
                date_str = match.group(1).strip()
                if self._is_reasonable_date(date_str):
                    return date_str
        
        # Secondary strategy: look for dates in patient context
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'patient|demographics|personal', line, re.IGNORECASE):
                # Look in this line and next few lines for dates
                search_text = ' '.join(lines[i:i+3])
                date_match = re.search(r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b', search_text)
                if date_match and self._is_reasonable_date(date_match.group(1)):
                    return date_match.group(1)
        
        return ""
    
    def extract_ssn_enhanced(self, text: str) -> str:
        """Enhanced SSN extraction"""
        for pattern in self._compiled_patterns['ssn_patterns']:
            match = pattern.search(text)
            if match:
                ssn = re.sub(r'[^\d]', '', match.group(1))
                if len(ssn) == 9 and not ssn.startswith('000'):
                    return match.group(1).strip()
        return ""
    
    def extract_mrn_enhanced(self, text: str) -> str:
        """Enhanced MRN extraction"""
        for pattern in self._compiled_patterns['mrn_patterns']:
            match = pattern.search(text)
            if match:
                mrn = match.group(1).strip()
                # ----- VALIDATION RULES -----
                # 1️⃣ Reject if all characters are the same (0000, AAAA …)
                if re.match(r'^(.)\1+$', mrn):
                    continue

                # 2️⃣ Digits-only MRN must be 8-12 digits (matches typical MRN lengths)
                if mrn.isdigit():
                    if 8 <= len(mrn) <= 12:
                        return mrn
                    else:
                        continue  # Too short/long – likely not an MRN

                # 3️⃣ Alphanumeric MRN must contain at least 1 letter & 1 digit and be 6-15 chars
                if 6 <= len(mrn) <= 15 and re.search(r'[A-Za-z]', mrn) and re.search(r'\d', mrn):
                    return mrn
        return ""
    
    def extract_episode_dates_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced episode date extraction"""
        dates = {}
        
        # Extract start date
        for pattern in self._compiled_patterns['episode_start_patterns']:
            match = pattern.search(text)
            if match:
                date_str = match.group(1).strip()
                if self._is_reasonable_date(date_str):
                    dates['start'] = date_str
                    break
        
        # Extract end date
        for pattern in self._compiled_patterns['episode_end_patterns']:
            match = pattern.search(text)
            if match:
                date_str = match.group(1).strip()
                if self._is_reasonable_date(date_str):
                    dates['end'] = date_str
                    break
        
        # Look for date ranges
        range_pattern = r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\s*(?:to|through|[-–—]{1,2}|until)\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
        range_match = re.search(range_pattern, text, re.IGNORECASE)
        if range_match:
            start_date = range_match.group(1).strip()
            end_date = range_match.group(2).strip()
            if self._is_reasonable_date(start_date) and self._is_reasonable_date(end_date):
                if 'start' not in dates:
                    dates['start'] = start_date
                if 'end' not in dates:
                    dates['end'] = end_date
        
        return dates
    
    def extract_npi_enhanced(self, text: str) -> str:
        """Enhanced NPI extraction"""
        for pattern in self._compiled_patterns['npi_patterns']:
            match = pattern.search(text)
            if match:
                npi = re.sub(r'[^\d]', '', match.group(1))
                if len(npi) == 10:
                    return npi
        return ""
    
    def extract_diagnoses_enhanced(self, text: str) -> List[EpisodeDiagnosis]:
        """Enhanced diagnosis extraction"""
        diagnoses = []
        
        for pattern in self._compiled_patterns['diagnosis_patterns']:
            matches = pattern.finditer(text)
            for match in matches:
                if len(match.groups()) >= 2:
                    # Pattern with ICD code
                    code = match.group(1).strip()
                    description = match.group(2).strip()
                else:
                    # Pattern with just description
                    code = ""
                    description = match.group(1).strip()
                
                # Clean up description
                description = self._clean_diagnosis_description(description)
                
                if description and len(description) > 3:
                    # Determine diagnosis type
                    diag_type = "primary" if "primary" in pattern.pattern.lower() else "secondary"
                    
                    diagnosis = EpisodeDiagnosis(
                        diagnosis_code=code,
                        diagnosis_description=description,
                        diagnosis_type=diag_type,
                        icd_version="ICD-10" if re.match(r'^[A-Z]\d{2}', code) else "ICD-9" if re.match(r'^\d{3}', code) else ""
                    )
                    
                    diagnoses.append(diagnosis)
                    
                    if len(diagnoses) >= 6:  # Limit to 6 diagnoses
                        break
            
            if len(diagnoses) >= 6:
                break
        
        return diagnoses
    
    def extract_insurance_enhanced(self, text: str) -> str:
        """Enhanced insurance extraction"""
        for pattern in self._compiled_patterns['insurance_patterns']:
            match = pattern.search(text)
            if match:
                insurance = match.group(1).strip()
                # Clean up insurance name
                insurance = re.sub(r'\s+', ' ', insurance)
                if len(insurance) > 2 and not re.match(r'^\d+$', insurance):
                    return insurance
        return ""
    
    def _is_reasonable_date(self, date_str: str) -> bool:
        """Check if date string represents a reasonable date"""
        if not date_str:
            return False
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = int(year_match.group(0))
            # Reasonable birth year range (1900-2024)
            return 1900 <= year <= 2024
        
        # Check 2-digit years
        two_digit_match = re.search(r'\b(\d{2})\b', date_str)
        if two_digit_match:
            year_2d = int(two_digit_match.group(0))
            # Assume reasonable range
            return 0 <= year_2d <= 99
        
        return False
    
    def _clean_diagnosis_description(self, description: str) -> str:
        """Clean diagnosis description"""
        if not description:
            return ""
        
        # Remove extra whitespace
        description = re.sub(r'\s+', ' ', description).strip()
        
        # Remove common prefixes/suffixes
        description = re.sub(r'^(diagnosis|dx|condition|problem)\s*:?\s*', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\s*(icd|code)\s*:?\s*[A-Z0-9\.]*$', '', description, flags=re.IGNORECASE)
        
        # Remove trailing punctuation except periods that are part of abbreviations
        description = re.sub(r'[,;:]+$', '', description)
        
        return description.strip()
    
    def extract_all_critical_fields(self, text: str) -> Dict[str, Any]:
        """Extract all critical fields that are commonly missing"""
        return {
            'dob': self.extract_dob_enhanced(text),
            'ssn': self.extract_ssn_enhanced(text),
            'mrn': self.extract_mrn_enhanced(text),
            'episode_dates': self.extract_episode_dates_enhanced(text),
            'npi': self.extract_npi_enhanced(text),
            'diagnoses': self.extract_diagnoses_enhanced(text),
            'insurance': self.extract_insurance_enhanced(text)
        } 