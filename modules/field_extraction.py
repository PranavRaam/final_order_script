"""
Advanced Field Extraction Module
Optimized regex patterns based on actual healthcare document formats
"""

import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from modules.data_structures import EpisodeDiagnosis

class FieldExtractor:
    """Advanced field extraction with patterns optimized for actual healthcare document formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns based on actual document analysis"""
        
        # -----------------------------------------------------------------
        # PATIENT NAME PATTERNS
        # -----------------------------------------------------------------
        # Added variants to catch ALL-UPPERCASE names that appear after
        # "CLIENT:" (e.g. "CLIENT:  GRIFFITHS, THOMAS") and other headers.
        # The new patterns are prepended so they are evaluated first.
        self.patient_name_patterns = [
            # FULL UPPERCASE after CLIENT:
            r"CLIENT:\s*([A-Z]+,\s*[A-Z]+(?:\s+[A-Z]\.?)*)(?:\s|$)",
            # FULL UPPERCASE after PATIENT: (fallback)
            r"Patient:?\s*([A-Z]+,\s*[A-Z]+(?:\s+[A-Z]\.?)*)(?:\s|$)",
            # "Rodrigues, Doris ( MA250415113301 )" - with spaces - MOST RELIABLE
            r"([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*\(\s*[A-Z0-9]+\s*\)",
            
            # "Rodrigues, Doris (MA250415113301)" - tight spacing - SECOND MOST RELIABLE  
            r"([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*\([A-Z0-9]+\)",
            
            # "CLIENT: RAJU SINGLA, MD BURHOE, ALDEN R" - extract only the patient part after MD/DO
            r"CLIENT:.*?(?:MD|DO)\s+([A-Z][A-Za-z\s,.-]+?)(?:\s+\d|\s*$|\n)",
            
            # Standard "CLIENT: PATIENT_NAME" without physician prefix
            r"CLIENT:\s*([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)",
            
            # "Patient: Smith, John A" - but avoid "Patient identity" etc.
            r"Patient:\s*([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)",
            
            # "Veeder, Frances E" (standalone name in Last, First format) - must be at line start
            r"^([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*$",
            
            # More specific pattern for patient name that avoids "identity", "information" etc.
            r"(?:patient|client)\s+name:?\s*([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)",
        ]
        
        # DATE OF BIRTH PATTERNS - Based on actual formats
        self.dob_patterns = [
            # "Patient's Date of Birth: 11/19/1939"
            r"Patient'?s?\s+Date\s+of\s+Birth:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            
            # "DOB: 12/15/1931"
            r"DOB:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            
            # "Date of Birth: May 15, 1980"
            r"Date\s+of\s+Birth:?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            
            # Generic date patterns near patient info
            r"(?:born|birth|dob)(?:\s+date)?:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ]
        
        # GENDER/SEX PATTERNS
        self.gender_patterns = [
            # "Patient's Gender: FEMALE" - most reliable
            r"Patient'?s?\s+Gender:?\s*(MALE|FEMALE|M|F)(?!\s+or\s+F)(?!\s+or\s+M)",
            
            # "Sex: F" - but not "Sex: M or F"
            r"Sex:?\s*([MF]|MALE|FEMALE)(?!\s+or\s+[MF])(?!\s+or\s+MALE|FEMALE)",
            
            # "Gender: MALE" - but not "Gender: M or F" 
            r"Gender:?\s*(MALE|FEMALE|M|F)(?!\s+or\s+[MF])(?!\s+or\s+MALE|FEMALE)",
            
            # Look for standalone gender values that are NOT part of form instructions
            r"(?<!or\s)(?<!Gender\s)(?<!Sex\s)\b(MALE|FEMALE)\b(?!\s+or\s+)",
            
            # Single letter gender but not in "M or F" context
            r"(?<!or\s)(?<!/)\b([MF])\b(?!\s+or\s+[MF])(?!/)",
            
            # Patient information context
            r"Patient.*?(?:Gender|Sex).*?:?\s*(MALE|FEMALE|M|F)(?!\s+or)",
        ]
        
        # MEDICAL RECORD NUMBER PATTERNS
        self.mrn_patterns = [
            # "(MA250430076706)" - MRN in parentheses format
            r"\(([A-Z]{2}\d{12})\)",
            
            # "Patient: Barlow, Maria (MA250430076706)" - MRN after patient name
            r"Patient:?\s+[A-Za-z,\s]+\s+\(([A-Z]{2}\d{12})\)",
            
            # "Medical Record No. RLN00426061501" - target the actual format
            r"Medical\s+Record\s+No\.?\s+([A-Za-z0-9]+)(?:\s+Provider|\s+\d|\s*$)",
            
            # "Medical Record No. Provider No. 2Q76FN4GC66 5/21/2025 5/21/2025 to 7/19/2025 RLN00426061501"
            r"Medical\s+Record\s+No\..*?([A-Za-z]{3}\d{11})(?:\s+\d|\s*$)",
            
            # "MR#: C0200089881701"
            r"MR#:?\s*([A-Za-z0-9]+)",
            
            # New: "MR# RLN00399791301" (no colon) or "MRF: RLN00399791301"
            r"MR[#F]?:?\s*([A-Za-z0-9]{8,})",
            
            # Generic MRN patterns
            r"(?:mrn|medical\s+record|patient\s+id)(?:\s+no)?\.?\s*:?\s*([A-Za-z0-9]{10,})",
            
            # Stand-alone code (placed LAST so labelled versions win first)
            r"\b((?:MA|RLN|C0|C02|HH|HS)[A-Z0-9]{8,13})\b",
            
            # 9-10 digit MRN (digits only) inside parentheses after patient name e.g. "Patient: Blair, Edward (100996456)"
            r"Patient:?\s+[A-Za-z,\s]+\s*\((\d{9,10})\)",
            # 9-10 digit MRN inside parentheses with no preceding label e.g. "Oliveira, Kathleen (1275520097)"
            r"[A-Z][A-Za-z]+,\s*[A-Z][A-Za-z]+\s*\((\d{9,10})\)",
            # Stand-alone 9-10 digit MRN inside parentheses anywhere in the doc
            r"\((\d{9,10})\)",
            # "MR Number: 000000200" or "MR Number 000000200"
            r"MR\s+Number:?\s*([A-Za-z0-9]{6,})",
            # Patient line with dash then MR#, e.g., "Patient: Hamilton, Doris-MR#000000200"
            r"Patient:?[\sA-Za-z,]+-?\s*MR#?\s*([A-Za-z0-9]{6,})",
        ]
        
        # MEDICARE NUMBER PATTERNS
        self.medicare_patterns = [
            # "Patient's Medicare No. 2Q76FN4GC66" - target the actual format
            r"Patient'?s?\s+Medicare\s+No\.?\s+([A-Za-z0-9]+)(?:\s+\d|\s+SOC|\s*$)",
            
            # "Medicare No. Provider No. 2Q76FN4GC66"
            r"Medicare\s+No\..*?([A-Za-z0-9]{10,})(?:\s+\d|\s*$)",
            
            # "Medicare: [number]"
            r"Medicare:?\s*([A-Za-z0-9]{10,})",
        ]
        
        # SSN PATTERNS
        self.ssn_patterns = [
            # "SSN: 138-20-9261"
            r"SSN:?\s*(\d{3}-\d{2}-\d{4})",
            
            # Generic SSN patterns
            r"\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b",
        ]
        
        # ORDER NUMBER PATTERNS
        self.order_number_patterns = [
            # "Order Number #1286414219" 
            r"Order\s+Number\s+#(\d{10})",
            
            # "Order #1289280264"
            r"Order\s+#(\d{10})",
            
            # "Order Number 12273338"
            r"Order\s+Number:?\s*(\d{6,})",
            
            # "Order Number: HOME HEALTH CERTIFICATION AND PLAN OF CARE 38031747"
            r"Order\s+Number:?\s*(?:[A-Z\s]+?)(\d{8,})",
            
            # More specific order patterns
            r"Order\s+(?:Number|No)\.?\s*:?\s*(\d{6,})",
        ]
        
        # -----------------------------------------------------------------
        # ORDER DATE PATTERNS
        # -----------------------------------------------------------------
        self.order_date_patterns = [
            # "Order Date: 5/21/2025 8:41 AM"
            r"Order\s+Date:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            
            # "Order Date: 5/29/2025 9:40 AM"
            r"Order\s+Date:?\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*[AP]M)",
            
            # New: includes seconds component e.g. "5/30/2025 6:44:12 AM"
            r"Order\s+Date:?\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)",
            
            # Generic order date patterns
            r"(?:ordered|order\s+date|date\s+ordered):?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ]
        
        # START OF CARE PATTERNS
        self.soc_patterns = [
            # "SOC Date 5/21/2025"
            r"SOC\s+Date:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            
            # "Start of Care 04/07/2025"
            r"Start\s+of\s+Care:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            
            # Generic start patterns
            r"(?:start|soc)(?:\s+of\s+care|\s+date)?:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ]
        
        # CERTIFICATION PERIOD PATTERNS
        self.cert_period_patterns = [
            # "Certification Period 5/21/2025 to 7/19/2025"
            r"Certification\s+Period:?\s*(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})",
            
            # "CERT: 4/24/2025 to 6/22/2025"
            r"CERT:?\s*(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})",
            
            # "04/07/2025 Through 06/05/2025"
            r"(\d{1,2}/\d{1,2}/\d{4})\s+Through\s+(\d{1,2}/\d{1,2}/\d{4})",
        ]
        
        # PHYSICIAN NAME PATTERNS
        self.physician_patterns = [
            # "CHRISTOPHER S. JONCAS, MD"
            r"([A-Z][A-Za-z]+(?:\s+[A-Z]\.?)*\s+[A-Z][A-Za-z]+,\s*MD)",
            
            # "Stephen R Butler, DO"
            r"([A-Z][A-Za-z]+(?:\s+[A-Z]\.?)*\s+[A-Z][A-Za-z]+,\s*DO)",
            
            # "PHYSICIAN: RAJU SINGLA, MD"
            r"PHYSICIAN:?\s*([A-Z][A-Za-z\s,.-]+,\s*MD)",
            
            # "Dr. [Name]"
            r"Dr\.?\s+([A-Z][A-Za-z\s,.-]+)",
            
            # Generic physician patterns
            r"(?:physician|doctor|dr|attending):?\s*([A-Z][A-Za-z\s,.-]+(?:,\s*MD|,\s*DO)?)",
        ]
        
        # PHONE NUMBER PATTERNS
        self.phone_patterns = [
            # "Phone: (508) 235-6312"
            r"Phone:?\s*\((\d{3})\)\s*(\d{3})-(\d{4})",
            
            # "(401) 624-1880"
            r"\((\d{3})\)\s*(\d{3})-(\d{4})",
            
            # Generic phone patterns
            r"(?:phone|tel|telephone):?\s*(\(?\d{3}\)?[-.\s]*\d{3}[-.\s]*\d{4})",
        ]
        
        # ADDRESS PATTERNS
        self.address_patterns = [
            # "1215 MAIN ROAD TIVERTON, RI 02878"
            r"(\d+\s+[A-Z][A-Za-z\s]+(?:ROAD|STREET|ST|AVENUE|AVE|DRIVE|DR|LANE|LN|BOULEVARD|BLVD)[A-Za-z\s,]*\d{5})",
            
            # "191 BEDFORD STREET FALL RIVER, MA 02720"
            r"(\d+\s+[A-Z][A-Za-z\s]+[A-Z]{2}\s+\d{5})",
            
            # Generic address patterns
            r"(\d+\s+[A-Za-z\s]+(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd)[A-Za-z\s,]*)",
        ]
        
        # DIAGNOSIS PATTERNS (ICD-10)
        self.diagnosis_patterns = [
            # "1 M48.56XD COLLAPSED VERT, NEC, LUMBAR REGION, SUBS FOR FX W ROUTN HEAL"
            r"\d+\s+([A-Z]\d{2}\.?\d*[A-Z]*)\s+([A-Z][A-Za-z\s,]+)",
            
            # Generic ICD patterns
            r"(?:diagnosis|dx|icd):?\s*([A-Z]\d{2}\.?\d*[A-Z]*)\s*([A-Za-z\s,]+)",
        ]
        
        # PROVIDER INFORMATION PATTERNS
        self.provider_patterns = [
            # "NURSE ON CALL - LINCOLN"
            r"Provider'?s?\s+Name[^:]*:\s*([A-Z][A-Za-z\s-]+)",
            
            # Agency/Provider names
            r"(ACCENTCARE[A-Za-z\s,.-]*|Community\s+Nurse[A-Za-z\s,.-]*|NURSE\s+ON\s+CALL[A-Za-z\s-]*)",
        ]
        
        # Compile all patterns
        self._compile_all_patterns()
    
    def _compile_all_patterns(self):
        """Compile all regex patterns for performance"""
        pattern_groups = [
            'patient_name_patterns', 'dob_patterns', 'gender_patterns', 'mrn_patterns',
            'medicare_patterns', 'ssn_patterns', 'order_number_patterns', 'order_date_patterns',
            'soc_patterns', 'cert_period_patterns', 'physician_patterns', 'phone_patterns',
            'address_patterns', 'diagnosis_patterns', 'provider_patterns'
        ]
        
        for group_name in pattern_groups:
            patterns = getattr(self, group_name)
            compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
            setattr(self, f'_compiled_{group_name}', compiled_patterns)
    
    def extract_patient_name(self, text: str) -> Dict[str, str]:
        """Extract patient first and last name"""
        for pattern in self._compiled_patient_name_patterns:
            match = pattern.search(text)
            if match:
                full_name = match.group(1).strip()
                if self._is_valid_patient_name(full_name):
                    return self._parse_full_name(full_name)
        return {'first_name': '', 'last_name': ''}
    
    def _parse_full_name(self, full_name: str) -> Dict[str, str]:
        """Parse full name into first and last name"""
        full_name = full_name.strip()
        
        # Handle "Last, First" format
        if ',' in full_name:
            parts = full_name.split(',', 1)
            last_name = parts[0].strip()
            first_name = parts[1].strip().split()[0] if parts[1].strip() else ''
            return {'first_name': first_name, 'last_name': last_name}
        
        # Handle "First Last" format
        parts = full_name.split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = parts[-1]
            return {'first_name': first_name, 'last_name': last_name}
        elif len(parts) == 1:
            return {'first_name': parts[0], 'last_name': ''}
        
        return {'first_name': '', 'last_name': ''}
    
    def extract_dob(self, text: str) -> str:
        """Extract date of birth"""
        for pattern in self._compiled_dob_patterns:
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                return self._normalize_date(date_str)
        return ''
    
    def extract_gender(self, text: str) -> str:
        """Extract gender/sex with validation to avoid form field labels"""
        for pattern in self._compiled_gender_patterns:
            match = pattern.search(text)
            if match:
                gender = match.group(1).upper()
                
                # Skip if this looks like a form field instruction
                context = text[max(0, match.start()-20):match.end()+20]
                if any(phrase in context.upper() for phrase in [
                    'M OR F', 'F OR M', 'MALE OR FEMALE', 'FEMALE OR MALE',
                    'CHECK ONE', 'CIRCLE ONE', 'SELECT', 'CHOOSE'
                ]):
                    continue
                
                # Additional validation
                if not self._is_valid_gender(gender):
                    continue
                
                # Return standardized gender
                if gender in ['M', 'MALE']:
                    return 'MALE'
                elif gender in ['F', 'FEMALE']:
                    return 'FEMALE'
        return ''
    
    def extract_mrn(self, text: str) -> str:
        """Extract medical record number"""
        for pattern in self._compiled_mrn_patterns:
            match = pattern.search(text)
            if match:
                # Use first captured group if available, else full match
                mrn = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if self._is_valid_mrn(mrn):
                    return mrn
        return ''
    
    def extract_medicare_number(self, text: str) -> str:
        """Extract Medicare number"""
        for pattern in self._compiled_medicare_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ''
    
    def extract_ssn(self, text: str) -> str:
        """Extract SSN"""
        for pattern in self._compiled_ssn_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ''
    
    def extract_order_number(self, text: str) -> str:
        """Extract order number"""
        for pattern in self._compiled_order_number_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ''
    
    def extract_order_date(self, text: str) -> str:
        """Extract order date"""
        for pattern in self._compiled_order_date_patterns:
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                return self._normalize_date(date_str)
        return ''
    
    def extract_soc_date(self, text: str) -> str:
        """Extract start of care date"""
        for pattern in self._compiled_soc_patterns:
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                return self._normalize_date(date_str)
        return ''
    
    def extract_certification_period(self, text: str) -> Dict[str, str]:
        """Extract certification period start and end dates"""
        for pattern in self._compiled_cert_period_patterns:
            match = pattern.search(text)
            if match:
                start_date = self._normalize_date(match.group(1))
                end_date = self._normalize_date(match.group(2))
                return {'start_date': start_date, 'end_date': end_date}
        return {'start_date': '', 'end_date': ''}
    
    def extract_physician_name(self, text: str) -> str:
        """Extract physician name"""
        for pattern in self._compiled_physician_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ''
    
    def extract_phone_number(self, text: str) -> str:
        """Extract phone number"""
        for pattern in self._compiled_phone_patterns:
            match = pattern.search(text)
            if match:
                if len(match.groups()) == 3:
                    return f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
                else:
                    return match.group(1).strip()
        return ''
    
    def extract_address(self, text: str) -> str:
        """Extract address"""
        for pattern in self._compiled_address_patterns:
            match = pattern.search(text)
            if match:
                address = match.group(1).strip()
                address = self._clean_extracted_address(address)
                if self._is_valid_address(address):
                    return address
        return ''
    
    def extract_primary_diagnosis(self, text: str) -> Dict[str, str]:
        """Extract primary diagnosis with ICD code"""
        for pattern in self._compiled_diagnosis_patterns:
            match = pattern.search(text)
            if match:
                icd_code = match.group(1)
                description = match.group(2).strip()
                return {'icd_code': icd_code, 'description': description}
        return {'icd_code': '', 'description': ''}
    
    def extract_provider_name(self, text: str) -> str:
        """Extract provider/agency name"""
        for pattern in self._compiled_provider_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ''
    
    def extract_diagnoses(self, text: str) -> List[EpisodeDiagnosis]:
        """Extract diagnoses list for compatibility with DataParser"""
        diagnosis = self.extract_primary_diagnosis(text)
        if diagnosis['icd_code'] or diagnosis['description']:
            episode_diagnosis = EpisodeDiagnosis(
                diagnosis_code=diagnosis['icd_code'],
                diagnosis_description=diagnosis['description'],
                diagnosis_type="primary",
                icd_version="ICD-10" if diagnosis['icd_code'].startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')) else "ICD-9"
            )
            return [episode_diagnosis]
        return []
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to MM/DD/YYYY format"""
        if not date_str:
            return ""
        
        # Remove time portion if present
        date_str = re.sub(r'\s+\d{1,2}:\d{2}.*', '', date_str)
        
        # Handle month name format (May 15, 1980)
        if re.match(r'[A-Za-z]+\s+\d{1,2},\s+\d{4}', date_str):
            try:
                from datetime import datetime
                parsed = datetime.strptime(date_str, '%B %d, %Y')
                return parsed.strftime('%m/%d/%Y')
            except:
                pass
        
        # Handle MM/DD/YYYY or MM-DD-YYYY
        date_match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
        if date_match:
            month, day, year = date_match.groups()
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
        
        # Handle MM/DD/YY
        date_match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})$', date_str)
        if date_match:
            month, day, year = date_match.groups()
            full_year = f"20{year}" if int(year) < 50 else f"19{year}"
            return f"{month.zfill(2)}/{day.zfill(2)}/{full_year}"
        
        return date_str
    
    def extract_all_fields(self, text: str) -> Dict[str, Any]:
        """Extract all fields from text in one pass"""
        if not text:
            return {}
        
        # Clean text for better matching
        text = self._clean_text(text)
        
        results = {}
        
        # Patient information
        patient_name = self.extract_patient_name(text)
        results['patient_fname'] = patient_name['first_name']
        results['patient_lname'] = patient_name['last_name']
        results['dob'] = self.extract_dob(text)
        
        # Extract gender with name-based inference as fallback
        extracted_gender = self.extract_gender(text)
        # Name-based inference can override weak or obviously wrong extraction
        if patient_name['first_name']:
            inferred_gender = self._infer_gender_from_name(patient_name['first_name'])
            if not extracted_gender:
                extracted_gender = inferred_gender
            elif inferred_gender and extracted_gender != inferred_gender:
                # Prefer name-based inference when they disagree (heuristic)
                extracted_gender = inferred_gender
        results['patient_sex'] = extracted_gender
        
        results['medical_record_no'] = self.extract_mrn(text)
        results['medicare_number'] = self.extract_medicare_number(text)
        results['ssn'] = self.extract_ssn(text)
        
        # Order information
        results['order_no'] = self.extract_order_number(text)
        results['order_date'] = self.extract_order_date(text)
        results['start_of_care'] = self.extract_soc_date(text)
        
        # Certification period
        cert_period = self.extract_certification_period(text)
        results['episode_start_date'] = cert_period['start_date']
        results['episode_end_date'] = cert_period['end_date']
        
        # Provider information
        results['physician_name'] = self.extract_physician_name(text)
        results['phone_number'] = self.extract_phone_number(text)
        results['address'] = self.extract_address(text)
        results['provider_name'] = self.extract_provider_name(text)
        
        # Diagnosis
        diagnosis = self.extract_primary_diagnosis(text)
        results['primary_diagnosis'] = diagnosis['description']
        results['icd_code'] = diagnosis['icd_code']
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving line boundaries for positional regexes."""
        # 1️⃣ Remove control chars but preserve newlines.
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', text)

        # 2️⃣ Collapse multiple spaces / tabs but leave newlines intact.
        text = re.sub(r'[ \t]+', ' ', text)

        # 3️⃣ Collapse multiple newlines to a single one to avoid giant gaps.
        text = re.sub(r'\r?\n+', '\n', text)

        # 4️⃣ Strip leading / trailing whitespace.
        return text.strip()
    
    def extract_field(self, text: str, field_name: str) -> str:
        """Generic field extraction method for backward compatibility"""
        all_fields = self.extract_all_fields(text)
        return all_fields.get(field_name, '')
    
    def _is_valid_patient_name(self, name: str) -> bool:
        """Validate if extracted name is a real patient name"""
        if not name or len(name.strip()) < 2:
            return False
        
        # Filter out template text and invalid names
        invalid_patterns = [
            r'first\s+name', r'last\s+name', r'patient\s+name',
            r'unspecified', r'unsp', r'hyperlipidemia', r'neuropathy',
            r'episode', r'diagnosis', r'client', r'template',
            r'MM/DD/YYYY', r'M\s+or\s+F', r'MALE\s+or\s+FEMALE',
            r'RN\s*$', r'MD\s*$', r'DO\s*$', r'^\s*[A-Z]{1,3}\s*$'
        ]
        
        name_upper = name.upper()
        for pattern in invalid_patterns:
            if re.search(pattern, name_upper):
                return False
        
        # Must contain at least one letter
        if not re.search(r'[A-Za-z]', name):
            return False
            
        return True
    
    def _is_valid_gender(self, gender: str) -> bool:
        """Validate if extracted gender is valid"""
        if not gender:
            return False
        
        gender_upper = gender.upper()
        
        # Filter out form field text
        invalid_patterns = [
            r'M\s+OR\s+F', r'F\s+OR\s+M', r'MALE\s+OR\s+FEMALE',
            r'FEMALE\s+OR\s+MALE', r'CHECK', r'CIRCLE', r'SELECT'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, gender_upper):
                return False
        
        return gender_upper in ['M', 'F', 'MALE', 'FEMALE']
    
    def _is_valid_mrn(self, mrn: str) -> bool:
        """Validate if extracted MRN looks like an actual MRN (letters+digits)"""
        if not mrn:
            return False
        
        mrn = mrn.strip().upper()

        # Accept either: 
        #   • alphanumeric MRN that contains at least one digit (existing rule), OR
        #   • digits-only MRN (commonly 9–10 digits coming from patient line)
        if len(mrn) < 6:
            return False

        has_letter = bool(re.search(r'[A-Z]', mrn))
        has_digit = bool(re.search(r'\d', mrn))

        if not has_digit:
            return False  # must contain digits

        if not has_letter:
            # Digits-only MRN – allow 6-15 digits, but reject if all digits are identical (e.g., 000000000)
            if not re.fullmatch(r'\d{6,15}', mrn):
                return False
            if re.match(r'^(\d)\1+$', mrn):
                return False
            return True

        # For alphanumeric MRNs keep previous rules: at least one letter & one digit
        
        # Discard obvious template words
        if mrn in {"MRN", "MEDICAL", "RECORD", "NUMBER"}:
            return False
        
        return True
    
    def _is_valid_address(self, address: str) -> bool:
        """Validate if extracted address is valid"""
        if not address or len(address.strip()) < 5:
            return False
        
        address_upper = address.upper()
        
        # Filter out system artifacts
        invalid_patterns = [
            r'WRAPPER', r'SESSION', r'CACHE', r'APPROVAL',
            r'^\d{10,}', r'ISAPPROVAL', r'SESSIONCACHEKEY',
            r'FAX\s+NUMBER', r'TIME\s+ZONE'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, address_upper):
                return False
        
        return True
    
    def _clean_extracted_address(self, address: str) -> str:
        """Clean extracted address of system artifacts"""
        if not address:
            return ""
        
        # Remove system artifacts
        address = re.sub(r'\d{10,}\s+isApprovalWrapper.*', '', address)
        address = re.sub(r'isApprovalWrapper.*', '', address)
        address = re.sub(r'sessionCacheKey.*', '', address)
        address = re.sub(r'Fax\s+Number\s+\d+', '', address)
        address = re.sub(r'\d{2}:\d{2}\s+[AP]M\s+Eastern\s+Time\s+Zone', '', address)
        
        return address.strip()
    
    def _infer_gender_from_name(self, first_name: str) -> str:
        """Infer gender from common first names"""
        if not first_name:
            return ''
        
        first_name = first_name.upper().strip()
        
        # Common female names
        female_names = {
            'MARIA', 'MARIE', 'MARY', 'OLIVIA', 'LUCIA', 'ANA', 'JOANN',
            'KATHLEEN', 'MELISSA', 'KAYLEIGH', 'KAYLEE', 'AMANDA', 'CONNIE',
            'VIRGINIA', 'RITA', 'NOEMIA', 'CANDIDA', 'ILDA', 'GUILHERMINA',
            'LOUISE', 'DALE'
        }
        
        # Common male names  
        male_names = {
            'DARWIN', 'EDWARD', 'JOAO', 'EDUARDO', 'WILLIAM', 'LAWRENCE',
            'ALFRED', 'JOHN'
        }
        
        if first_name in female_names:
            return 'FEMALE'
        elif first_name in male_names:
            return 'MALE'
        
        return '' 