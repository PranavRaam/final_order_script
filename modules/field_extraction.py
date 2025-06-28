"""
Advanced Field Extraction Module
Optimized regex patterns based on actual healthcare document formats
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
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
            r"([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*\(\s*[A-Z0-9]{3,}\s*\)",
            
            # "Rodrigues, Doris (MA250415113301)" - tight spacing - SECOND MOST RELIABLE  
            r"([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*\([A-Z0-9]{3,}\)",
            
            # Patient name in structured data context (e.g., "Name and Address Gender Date of Birth Phone Number Sousa, Joao Male")
            r"(?:Name|Patient).*?(?:Number|Address|Gender|Birth)\s+([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s+(?:Male|Female|M|F|\d)",
            
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
            # "Visit Date: 04/24/2025"
            r"Visit\s+Date:?:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            # "Date order received: 05/07/2025"
            r"Date\s+order\s+received:?:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            # with time
            r"Date\s+order\s+received:?:?\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M)",
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
        
        # CERTIFICATION/Episode PERIOD PATTERNS
        self.cert_period_patterns = [
            # Existing patterns retain
            r"Certification\s+Period:?\s*(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})",
            r"CERT:?\s*(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{1,2}/\d{1,2}/\d{4})\s+Through\s+(\d{1,2}/\d{1,2}/\d{4})",
            # Basic "Episode: 04/14/2025 - 06/12/2025"
            r"Episode:?\s*(\d{1,2}/\d{1,2}/\d{4})\s*(?:-|–|—|to|To|through)[:]?\s*(\d{1,2}/\d{1,2}/\d{4})",
            # Allow extra words between Episode label and first date (e.g., "Episode Dates 04/14/2025 - 06/12/2025")
            r"Episode\s+[A-Za-z\s]*?(\d{1,2}/\d{1,2}/\d{4})\s*(?:-|–|—|to|To|through)[:]?\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"Certification\s+Period:?:?\s*(\d{1,2}/\d{1,2}/\d{4})\s*[-–—]{1,2}\s*(\d{1,2}/\d{1,2}/\d{4})",
            # Generic 'From DATE To DATE' pattern (when preceded by Certification Period)
            r"Certification\s+Period[^\d]{0,20}(\d{1,2}/\d{1,2}/\d{4})\s*(?:-|to|To|through|–|—)\s*(\d{1,2}/\d{1,2}/\d{4})",
            # Service dates ISO pattern
            r"Service\s+Dates?:?\s*(\d{4}-\d{2}-\d{2})\s*(?:-|to|through|–|—|→|➡)\s*(\d{4}-\d{2}-\d{2})",
            # Episode/Certification with Unicode arrows or em dash between dates
            r"Episode\s+(?:Dates?|Period|Span|Range)?[:\s]*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s*[\u2190\u2192\u27A1\u2012\u2013\u2014\u2015-]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            # Month-name dates with dash/arrow
            r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*[\u2190\u2192\u27A1\u2012\u2013\u2014\u2015-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        ]
        
        # Compile lists for faster search
        self._compiled_cert_period_patterns = [re.compile(p, re.IGNORECASE) for p in self.cert_period_patterns]
        
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
            # Standard ICD-10 code format: letter + 2 digits + optional dot + additional characters
            r"([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\s+([A-Z][A-Za-z0-9\s,.\-'():/]+?)(?:\s+Start Effective Date|\s+\([EO]\)|\s*$)",
            
            # Numbered diagnosis lines: "1 M48.56XD COLLAPSED VERT, NEC, LUMBAR REGION, SUBS FOR FX W ROUTN HEAL"
            r"\d+\s+([A-Z]\d{2}\.?\d*[A-Z]*)\s+([A-Z][A-Za-z\s,]+)",
            
            # Primary/Secondary Diagnosis format: "Primary Diagnosis Code Description"
            r"(?:Primary|Secondary)\s+Diagnosis\s+Code\s+Description\s+([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\s+([A-Za-z0-9\s,.\-'():/]+)",
            
            # Diagnosis with colon: "Diagnosis: I10. Essential (primary) hypertension"
            r"(?:diagnosis|dx|icd):?\s*([A-Z]\d{2}\.?\d*[A-Z]*)\s*([A-Za-z\s,]+)",
            
            # Medical diagnosis format: "Medical Diagnosis: G20.A1"
            r"Medical\s+Diagnosis:\s*([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)",
            
            # Diagnosis in parentheses at end: "Essential (primary) hypertension (E)"
            r"([A-Z][A-Za-z0-9\s,.\-'():/]+?)\s+\(([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\)",
        ]
        
        # PROVIDER / AGENCY INFORMATION PATTERNS  (broadened)
        # ------------------------------------------------------------
        # 1. Explicit label lines (most reliable)
        #    e.g. "Agency Name: Bristol Home Health Care"
        #         "Agency: VNA of Boston"
        #         "Provider/Agency: Kindred at Home"
        # 2. Generic trailing keywords such as  "Home Health", "Hospice",
        #    "Healthcare", "Visiting Nurse", "VNA" etc.
        # 3. Upper-case lines that end with those keywords.
        self.provider_patterns = [
            # Labelled versions first so they win (but exclude simple "Provider Name" which is usually physician)
            r"(?:Agency\s+Name|Agency|Provider/Agency)\s*[:\-]?\s*([A-Z][A-Za-z0-9 &'.,-]{3,50})",
            
            # Generic name that ends with common agency words
            r"([A-Z][A-Za-z0-9 &'.,-]{3,40}\s+(?:Home\s+Health|Home\s*Care|Health\s*Care|Healthcare|Hospice|Visiting\s+Nurse|VNA)(?:\s+of\s+[A-Z][A-Za-z]+)?)",
            
            # Names where the keyword is *in the middle* e.g. "Kindred at Home"
            r"([A-Z][A-Za-z0-9 &'.,-]{3,40}\s+at\s+Home)",
            
            # Preserve legacy hard-coded matches
            r"(ACCENTCARE[A-Za-z\s,.-]{0,30}|Community\s+Nurse[A-Za-z\s,.-]{0,30}|NURSE\s+ON\s+CALL[A-Za-z\s-]{0,30})",
            
            # Specific well-known agency patterns
            r"(AlphaCare\s+Home\s+Health)",
            r"(Nightingale\s+Visiting\s+Nurse)",
            r"(VISITING\s+NURSE\s+HOME\s+AND\s+HOSPICE)",
            r"(FALL\s+RIVER\s+-\s+CENTERWELL\s+HOME\s+HEALTH)",
            r"(VNHH\s+HOSPICE)",
            r"(AMEDISYS\s+HOME\s+HEALTH)",
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
        
        candidate_dates = self._extract_all_dates(text)
        start, end = self._choose_episode_pair(candidate_dates)
        if start and end:
            return {'start_date': start, 'end_date': end}
        
        # 🔄 SOC inference fallback
        soc_date_str = self.extract_soc_date(text)
        if soc_date_str:
            try:
                soc_dt = datetime.strptime(soc_date_str, '%m/%d/%Y')
                end_dt = soc_dt + timedelta(days=59)
                return {
                    'start_date': soc_dt.strftime('%m/%d/%Y'),
                    'end_date': end_dt.strftime('%m/%d/%Y')
                }
            except Exception:
                pass
        
        return {'start_date': '', 'end_date': ''}
    
    def _extract_all_dates(self, text: str) -> List[datetime]:
        """Return a list of datetime objects for every date we can recognise in the text.
        Handles:
            • MM/DD/YYYY or MM-DD-YYYY
            • Month DD, YYYY  (e.g., May 08, 2025)
            • YYYY-MM-DD (ISO)
        """
        patterns = [
            re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'),          # 05/08/2025
            re.compile(r'([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})'),   # May 8, 2025
            re.compile(r'(\d{4})-(\d{2})-(\d{2})')                    # 2025-05-08
        ]

        month_lookup = {
            'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
            'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12
        }

        dates: List[datetime] = []
        for pat in patterns:
            for m in pat.finditer(text):
                try:
                    if len(m.groups()) == 3:
                        g1, g2, g3 = m.groups()
                        if pat.pattern.startswith('('):  # numeric month first
                            month, day, year = int(g1), int(g2), int(g3)
                        elif pat.pattern.startswith('([A-Za-z'):  # month name first
                            month = month_lookup.get(g1.upper(), 0)
                            day, year = int(g2), int(g3)
                        else:  # ISO
                            year, month, day = int(g1), int(g2), int(g3)
                        dates.append(datetime(year, month, day))
                except Exception:
                    continue
        return dates

    def _choose_episode_pair(self, dates: List[datetime]) -> Tuple[str, str]:
        """Given a list of dates, choose the pair that most likely represents a 60-day episode."""
        if len(dates) < 2:
            return '', ''
        dates = sorted(set(dates))
        best_pair = None
        best_diff = 9999
        for i in range(len(dates)):
            for j in range(i+1, len(dates)):
                diff = abs((dates[j] - dates[i]).days)
                if 40 <= diff <= 120:  # plausible episode length
                    # choose diff closest to 60
                    score = abs(diff - 60)
                    if score < best_diff:
                        best_diff = score
                        best_pair = (dates[i], dates[j])
                        if best_diff == 0:
                            break
        if best_pair:
            return best_pair[0].strftime('%m/%d/%Y'), best_pair[1].strftime('%m/%d/%Y')
        return '', ''
    
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
                provider_name = match.group(1).strip()
                if self._is_valid_provider_name(provider_name):
                    return provider_name
        return ''
    
    def extract_diagnoses(self, text: str) -> List[EpisodeDiagnosis]:
        """Extract all ICD-10 codes from diagnosis tables in the text."""
        diagnoses = []
        
        # Use all diagnosis patterns to find codes
        for pattern in self._compiled_diagnosis_patterns:
            for match in pattern.finditer(text):
                code = ""
                desc = ""
                
                # Handle different pattern group structures
                if match.lastindex == 2:
                    # Pattern with code and description groups
                    if re.match(r"[A-Z][0-9]{2}", match.group(1)):
            code = match.group(1).strip()
            desc = match.group(2).strip()
                    else:
                        # Swapped order - description then code
                        desc = match.group(1).strip()
                        code = match.group(2).strip()
                elif match.lastindex == 1:
                    # Pattern with only code group
                    code = match.group(1).strip()
                    
                if code and re.match(r"[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?", code):
                    # Clean up description
                    if desc:
                        desc = re.sub(r'\s+', ' ', desc).strip()
                        # Remove trailing markers
                        desc = re.sub(r'\s+\([EO]\)\s*$', '', desc)
                        desc = re.sub(r'\s+Start Effective Date.*$', '', desc)
                    
                diagnoses.append(EpisodeDiagnosis(
                    diagnosis_code=code,
                    diagnosis_description=desc,
                    diagnosis_type="other",
                        icd_version="ICD-10"
                    ))
        
        # Additional specific searches for common diagnosis patterns in documents
        additional_patterns = [
            # Format: "Code Description (E)" at end of line
            r"([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\s+([A-Za-z][A-Za-z0-9\s,.\-'():/]+?)\s+\([EO]\)",
            # Format: "Description Code" 
            r"([A-Za-z][A-Za-z0-9\s,.\-'():/]{10,}?)\s+([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\s*$",
        ]
        
        for pattern_str in additional_patterns:
            pattern = re.compile(pattern_str, re.MULTILINE | re.IGNORECASE)
            for match in pattern.finditer(text):
                if re.match(r"[A-Z][0-9]{2}", match.group(1)):
                    code = match.group(1).strip()
                    desc = match.group(2).strip() if match.lastindex >= 2 else ""
                else:
                    desc = match.group(1).strip()
                    code = match.group(2).strip()
                    
                if code and re.match(r"[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?", code):
                    # Clean up description
                    if desc:
                        desc = re.sub(r'\s+', ' ', desc).strip()
                        desc = re.sub(r'\s+\([EO]\)\s*$', '', desc)
                        
                    diagnoses.append(EpisodeDiagnosis(
                        diagnosis_code=code,
                        diagnosis_description=desc,
                        diagnosis_type="other",
                        icd_version="ICD-10"
                    ))
        
        # Remove duplicates based on diagnosis code
        seen_codes = set()
        unique_diagnoses = []
        for diagnosis in diagnoses:
            if diagnosis.diagnosis_code not in seen_codes:
                seen_codes.add(diagnosis.diagnosis_code)
                unique_diagnoses.append(diagnosis)
        
        # Fallback: if nothing found, use the old single extraction
        if not unique_diagnoses:
            diagnosis = self.extract_primary_diagnosis(text)
            if diagnosis['icd_code'] or diagnosis['description']:
                unique_diagnoses.append(EpisodeDiagnosis(
                    diagnosis_code=diagnosis['icd_code'],
                    diagnosis_description=diagnosis['description'],
                    diagnosis_type="primary",
                    icd_version="ICD-10" if diagnosis['icd_code'].startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')) else "ICD-9"
                ))
        
        return unique_diagnoses
    
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
        
        # Map provider/agency name to ordering facility for downstream CSV columns
        results['ordering_facility'] = results['provider_name']

        # -------------------------------------------------------------
        # 🏠  Address → City / State / ZIP parsing
        # -------------------------------------------------------------
        city = state = zip_code = ''
        if results['address']:
            city, state, zip_code = self._parse_city_state_zip(results['address'])

        # Store parsed components so StructuredParser can pick them up (after fallback)
        results['city'] = city
        results['state'] = state
        results['zip_code'] = zip_code
        
        # Fallback: if city/state still blank, scan whole text for first CITY, ST ZIP pattern
        if (not city or not state) and text:
            m = re.search(r"([A-Za-z][A-Za-z\s\.]+?),\s+([A-Z]{2})\s+(\d{4,5}(?:-\d{4})?)", text)
            if m:
                city = city or m.group(1).title().strip()
                state = state or m.group(2).upper()
                zip_code = zip_code or m.group(3)
                if re.fullmatch(r'\d{4}', zip_code):
                    zip_code = '0' + zip_code

            # Update results with fallback values
            results['city'] = city
            results['state'] = state
            results['zip_code'] = zip_code

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
            r'hypertension', r'diabetes', r'pneumonia', r'copd', r'chr\s+kidney',
            r'heart\s+failure', r'coronary\s+artery', r'malignant', r'neoplasm',
            r'chronic', r'acute', r'primary', r'secondary', r'essential',
            r'episode', r'diagnosis', r'client', r'template',
            r'MM/DD/YYYY', r'M\s+or\s+F', r'MALE\s+or\s+FEMALE',
            r'RN\s*$', r'MD\s*$', r'DO\s*$', r'^\s*[A-Z]{1,3}\s*$',
            # Add medical condition patterns that could match "condition, qualifier" format
            r'disease', r'disorder', r'syndrome', r'failure', r'block',
            r'anemia', r'thyroid', r'kidney', r'bone', r'prostate',
            r'aspirin', r'nicotine', r'dependence', r'history', r'use',
            r'metabolic', r'congestive', r'obstructive', r'respiratory',
            # Add more specific patterns that appear in diagnoses
            r'hyperkalemia', r'hypercalcemia', r'diastolic', r'systolic',
            r'parkinson', r'dementia', r'malignancy', r'cancer'
        ]
        
        name_lower = name.lower()  # Convert to lowercase for case-insensitive matching
        for pattern in invalid_patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return False
        
        # Must contain at least one letter and at least one vowel to avoid
        # catching stray field labels like "PRN" or "DX" etc.
        if not re.search(r'[A-Za-z]', name):
            return False

        if not re.search(r'[AEIOUaeiou]', name):
            return False
            
        # Additional check: if it contains ICD codes or medical codes, reject it
        if re.search(r'[A-Z]\d{2}\.?\d*', name):  # Matches patterns like "E78.5" or "I10"
            return False
            
        # Check if it looks like a medical term by checking for typical medical word endings
        medical_endings = [r'emia$', r'itis$', r'osis$', r'pathy$', r'trophy$', r'uria$', r'algia$']
        for ending in medical_endings:
            if re.search(ending, name_lower):
                return False
        
        # Special check: If it contains comma-separated medical terms, reject it
        # This handles cases like "Hyperlipidemia, unspecified"
        if ',' in name:
            parts = [part.strip().lower() for part in name.split(',')]
            medical_terms = {
                'unspecified', 'chronic', 'acute', 'primary', 'secondary', 
                'hyperlipidemia', 'hypertension', 'diabetes', 'essential',
                'malignant', 'benign', 'stage', 'grade', 'type', 'mild',
                'moderate', 'severe', 'with', 'without', 'complicated',
                'uncomplicated', 'controlled', 'uncontrolled'
            }
            if any(part in medical_terms for part in parts):
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
        """Validate if extracted address is reasonable"""
        if not address or len(address.strip()) < 5:
            return False
        
        # Filter out template text and invalid addresses
        invalid_patterns = [
            r'address', r'street', r'city', r'state', r'zip',
            r'template', r'form', r'field', r'unspecified',
            r'MM/DD/YYYY', r'^\s*[A-Z]{1,3}\s*$'
        ]
        
        address_upper = address.upper()
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

    # -----------------------------------------------------------------
    # 🌆  Helper: Parse City / State / ZIP from an address line
    # -----------------------------------------------------------------
    def _parse_city_state_zip(self, address: str) -> Tuple[str, str, str]:
        """Given a postal address string, attempt to extract the City, two-letter
        State code, and 5-digit ZIP (+optional 4). Returns 3 strings, which may
        be blank if parsing fails."""

        if not address:
            return '', '', ''

        # Normalize whitespace
        addr = re.sub(r"\s+", " ", address).strip()

        # Typical ending pattern: "CITY, ST 01234" or "CITY ST 01234"
        # Allow multi-word CITY (e.g., "FALL RIVER") and optional comma.
        m = re.search(r"([A-Za-z][A-Za-z\s\.]+?)\s*,?\s+([A-Z]{2})\s+(\d{4,5}(?:-\d{4})?)$", addr)
        if m:
            raw_city = m.group(1).title().strip()
            # Remove street-type tokens that may have bled into the match
            street_tokens = {
                'St', 'Street', 'Rd', 'Road', 'Ave', 'Avenue', 'Dr', 'Drive',
                'Ln', 'Lane', 'Blvd', 'Boulevard', 'Way', 'Hwy', 'Highway'
            }
            # Tokenize and filter
            words = [w for w in re.split(r'\s+', raw_city) if w]
            cleaned_words = [w for w in words if w not in street_tokens]
            # Improved heuristic: take last word unless it's a directional pair
            directional = {'North', 'South', 'East', 'West', 'New', 'Ft', 'Fort', 'St', 'Saint'}
            if len(cleaned_words) >= 2 and cleaned_words[-2] in directional:
                city = ' '.join(cleaned_words[-2:])  # e.g., "North Dartmouth"
            else:
                city = cleaned_words[-1] if cleaned_words else ''
            state = m.group(2).upper()
            zip_code = m.group(3)
            # Zero-pad 4-digit ZIPs (OCR sometimes drops leading zero)
            if re.fullmatch(r'\d{4}', zip_code):
                zip_code = '0' + zip_code
            return city, state, zip_code

        return '', '', '' 

    def _is_valid_provider_name(self, provider_name: str) -> bool:
        """Validate if extracted provider name is reasonable"""
        if not provider_name or len(provider_name.strip()) < 3:
            return False
        
        # Filter out noise text that commonly appears in documents
        invalid_patterns = [
            r'^signature\s*$', r'signature\s+signature', r'^\s*signature\s',
            r'^https?\s*$', r'^\s*https?://', r'^\s*www\.',
            r'^\s*[A-Z]\.\s*$', r'^\s*[A-Z]{1,3}\s*$',  # Single letters or short codes
            r'clinical\s+data', r'clinical\s+manager', r'branch\s+name',
            r'phone\s+number', r'address', r'time\s+zone',
            r'eastern\s+time', r'AM\s+Eastern', r'PM\s+Eastern',
            r'^\s*AM\s*$', r'^\s*PM\s*$',
            r'decline\s+in\s+mental', r'currently\s+taking',
            r'currently\s+reports', r'exhaustion',
            r'medications', r'behavioral\s+status',
            r'in\s+the\s+past\s+\d+\s+months',
            r'^\s*[,.\-\s]+$',  # Just punctuation
            r'^\s*\d+\s*$',  # Just numbers
            r'template', r'form', r'field', r'unspecified'
        ]
        
        provider_upper = provider_name.upper()
        for pattern in invalid_patterns:
            if re.search(pattern, provider_upper):
                return False
        
        # Must contain at least one letter
        if not re.search(r'[A-Za-z]', provider_name):
            return False
        
        # Should not be mostly punctuation or whitespace
        letter_count = len(re.findall(r'[A-Za-z]', provider_name))
        total_count = len(provider_name)
        if letter_count / total_count < 0.5:  # At least 50% letters
            return False
            
        return True 