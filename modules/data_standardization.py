"""
Data Standardization Module
Handles proper formatting and normalization of extracted healthcare data
"""

import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import logging

class DataStandardizer:
    """Standardizes and normalizes extracted healthcare data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # State abbreviation mappings
        self.state_mappings = {
            'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
            'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
            'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
            'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
            'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
            'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
            'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
            'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
            'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
            'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
            'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
            'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
            'wisconsin': 'WI', 'wyoming': 'WY'
        }
    
    def standardize_date(self, date_str: str) -> str:
        """Convert any date format to MM/DD/YYYY"""
        if not date_str or not date_str.strip():
            return ""
        
        # Clean the input
        date_str = date_str.strip()
        
        # Remove common prefixes/suffixes
        date_str = re.sub(r'^(DOB|Date of Birth|Born|Birthday):\s*', '', date_str, flags=re.IGNORECASE)
        date_str = re.sub(r'\s*(years? old|age \d+).*$', '', date_str, flags=re.IGNORECASE)
        
        # Try different date patterns
        date_patterns = [
            # MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
            (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})', lambda m: f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"),
            
            # MM/DD/YY, MM-DD-YY (assume 20xx for years 00-30, 19xx for 31-99)
            (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})$', lambda m: self._expand_year(m.group(1), m.group(2), m.group(3))),
            
            # YYYY/MM/DD, YYYY-MM-DD
            (r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', lambda m: f"{m.group(2).zfill(2)}/{m.group(3).zfill(2)}/{m.group(1)}"),
            
            # Month DD, YYYY or Month DD YYYY
            (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', 
             lambda m: self._month_name_to_date(m.group(1), m.group(2), m.group(3))),
            
            # Mon DD, YYYY or Mon DD YYYY (abbreviated months)
            (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+(\d{4})', 
             lambda m: self._month_abbr_to_date(m.group(1), m.group(2), m.group(3))),
            
            # DD Month YYYY
            (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 
             lambda m: self._month_name_to_date(m.group(2), m.group(1), m.group(3))),
             
            # DD Mon YYYY (abbreviated months)
            (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})', 
             lambda m: self._month_abbr_to_date(m.group(2), m.group(1), m.group(3)))
        ]
        
        for pattern, formatter in date_patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    result = formatter(match)
                    # Validate the result
                    if self._is_valid_date(result):
                        return result
                except Exception as e:
                    self.logger.debug(f"Date formatting error: {e}")
                    continue
        
        # If no pattern matches, try to extract just numbers and guess format
        numbers = re.findall(r'\d+', date_str)
        if len(numbers) == 3:
            try:
                # Try MM/DD/YYYY if first number <= 12
                if int(numbers[0]) <= 12 and int(numbers[1]) <= 31:
                    if len(numbers[2]) == 4:  # Full year
                        result = f"{numbers[0].zfill(2)}/{numbers[1].zfill(2)}/{numbers[2]}"
                    elif len(numbers[2]) == 2:  # Two-digit year
                        year = f"20{numbers[2]}" if int(numbers[2]) <= 30 else f"19{numbers[2]}"
                        result = f"{numbers[0].zfill(2)}/{numbers[1].zfill(2)}/{year}"
                    else:
                        return ""
                    
                    if self._is_valid_date(result):
                        return result
            except ValueError:
                pass
        
        return ""
    
    def standardize_gender(self, gender_str: str) -> str:
        """Standardize gender to MALE or FEMALE"""
        if not gender_str or not gender_str.strip():
            return ""

        # Remove common punctuation / whitespace and uppercase
        gender_str = gender_str.strip().upper().replace('.', '').replace(',', '')

        # Handle common combined placeholders like "M/F", "M OR F", "MALE/FEMALE"
        # These usually indicate instructions rather than actual value → return empty
        if '/' in gender_str or ' OR ' in gender_str:
            return ""

        # Map various formats to standard constants
        male_aliases = {
            'M', 'MALE', 'MAN', 'MASCULINE', 'BOY', 'HOMBRE'
        }
        female_aliases = {
            'F', 'FEMALE', 'WOMAN', 'FEMININE', 'GIRL', 'MUJER'
        }

        if gender_str in male_aliases:
            return 'MALE'
        if gender_str in female_aliases:
            return 'FEMALE'

        # If unrecognised, return empty so downstream validation can flag
        return ""
    
    def standardize_phone(self, phone_str: str) -> str:
        """Standardize phone number to (XXX) XXX-XXXX format"""
        if not phone_str or not phone_str.strip():
            return ""
        
        # Extract only digits
        digits = re.sub(r'\D', '', phone_str)
        
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        return phone_str.strip()  # Return original if can't format
    
    def standardize_ssn(self, ssn_str: str) -> str:
        """Standardize SSN to XXX-XX-XXXX format"""
        if not ssn_str or not ssn_str.strip():
            return ""
        
        # Extract only digits
        digits = re.sub(r'\D', '', ssn_str)
        
        if len(digits) == 9:
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        
        return ssn_str.strip()  # Return original if can't format
    
    def standardize_state(self, state_str: str) -> str:
        """Standardize state to 2-letter abbreviation"""
        if not state_str or not state_str.strip():
            return ""
        
        state_str = state_str.strip().lower()
        
        # If already 2 letters, return uppercase
        if len(state_str) == 2 and state_str.isalpha():
            return state_str.upper()
        
        # Check full state names
        return self.state_mappings.get(state_str, state_str.upper()[:2])
    
    def standardize_zip(self, zip_str: str) -> str:
        """Standardize ZIP code"""
        if not zip_str or not zip_str.strip():
            return ""
        
        # Extract digits and hyphens
        zip_clean = re.sub(r'[^\d\-]', '', zip_str)
        
        # Handle 5-digit or 9-digit ZIP
        if re.match(r'^\d{5}$', zip_clean):
            return zip_clean
        elif re.match(r'^\d{5}\d{4}$', zip_clean):
            return f"{zip_clean[:5]}-{zip_clean[5:]}"
        elif re.match(r'^\d{5}-\d{4}$', zip_clean):
            return zip_clean
        
        return zip_str.strip()
    
    def standardize_name(self, name_str: str) -> str:
        """Standardize name fields"""
        if not name_str or not name_str.strip():
            return ""
        
        # Remove extra whitespace and special characters
        name_str = re.sub(r'[^\w\s\-\']', ' ', name_str)
        name_str = re.sub(r'\s+', ' ', name_str).strip()
        
        # Title case but preserve certain patterns
        words = name_str.split()
        standardized_words = []
        
        for word in words:
            if word.upper() in ['II', 'III', 'IV', 'JR', 'SR']:
                standardized_words.append(word.upper())
            elif word.lower().startswith('mc') and len(word) > 2:
                # Handle McNames
                standardized_words.append('Mc' + word[2:].capitalize())
            elif word.lower().startswith("o'") and len(word) > 2:
                # Handle O'Names
                standardized_words.append("O'" + word[2:].capitalize())
            else:
                standardized_words.append(word.capitalize())
        
        return ' '.join(standardized_words)
    
    def _expand_year(self, month: str, day: str, year2: str) -> str:
        """Expand 2-digit year to 4-digit year"""
        year_int = int(year2)
        if year_int <= 30:
            full_year = f"20{year2}"
        else:
            full_year = f"19{year2}"
        
        return f"{month.zfill(2)}/{day.zfill(2)}/{full_year}"
    
    def _month_name_to_date(self, month_name: str, day: str, year: str) -> str:
        """Convert month name to MM/DD/YYYY"""
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        month_num = month_map.get(month_name.lower())
        if month_num:
            return f"{month_num}/{day.zfill(2)}/{year}"
        return ""
    
    def _month_abbr_to_date(self, month_abbr: str, day: str, year: str) -> str:
        """Convert month abbreviation to MM/DD/YYYY"""
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        month_num = month_map.get(month_abbr.lower())
        if month_num:
            return f"{month_num}/{day.zfill(2)}/{year}"
        return ""
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Validate MM/DD/YYYY date format"""
        try:
            datetime.strptime(date_str, '%m/%d/%Y')
            return True
        except ValueError:
            return False
    
    def standardize_all_patient_data(self, patient_data) -> None:
        """Apply standardization to all patient data fields"""
        # Names
        if patient_data.patient_fname:
            patient_data.patient_fname = self.standardize_name(patient_data.patient_fname)
        if patient_data.patient_lname:
            patient_data.patient_lname = self.standardize_name(patient_data.patient_lname)
        if patient_data.patient_mname:
            patient_data.patient_mname = self.standardize_name(patient_data.patient_mname)
        
        # Date of Birth
        if patient_data.dob:
            patient_data.dob = self.standardize_date(patient_data.dob)
        
        # Gender
        if patient_data.patient_sex:
            patient_data.patient_sex = self.standardize_gender(patient_data.patient_sex)
        
        # Contact info
        if patient_data.phone_number:
            patient_data.phone_number = self.standardize_phone(patient_data.phone_number)
        if patient_data.emergency_contact_phone:
            patient_data.emergency_contact_phone = self.standardize_phone(patient_data.emergency_contact_phone)
        
        # Address
        if patient_data.state:
            patient_data.state = self.standardize_state(patient_data.state)
        if patient_data.zip_code:
            patient_data.zip_code = self.standardize_zip(patient_data.zip_code)
        
        # SSN
        if patient_data.ssn:
            patient_data.ssn = self.standardize_ssn(patient_data.ssn)
    
    def standardize_all_order_data(self, order_data) -> None:
        """Apply standardization to all order data fields"""
        # Dates
        if order_data.order_date:
            order_data.order_date = self.standardize_date(order_data.order_date)
        if order_data.episode_start_date:
            order_data.episode_start_date = self.standardize_date(order_data.episode_start_date)
        if order_data.episode_end_date:
            order_data.episode_end_date = self.standardize_date(order_data.episode_end_date)
        if order_data.start_of_care:
            order_data.start_of_care = self.standardize_date(order_data.start_of_care)
        if order_data.signed_by_physician_date:
            order_data.signed_by_physician_date = self.standardize_date(order_data.signed_by_physician_date)
        
        # Names
        if order_data.physician_name:
            order_data.physician_name = self.standardize_name(order_data.physician_name)
        
        # Phone
        if order_data.physician_phone:
            order_data.physician_phone = self.standardize_phone(order_data.physician_phone)
    
    def standardize_mrn(self, mrn_str: str) -> str:
        """Return MRN if it matches expected pattern (2 letters + 9-13 digits) else blank."""
        if not mrn_str or not mrn_str.strip():
            return ""
        mrn_str = mrn_str.strip().upper().replace(" ", "")
        # Allow either standard alphanumeric format (2 letters + digits) or pure 9–10 digit identifiers
        if re.match(r'^[A-Z]{2}\d{9,13}$', mrn_str):
            return mrn_str
        if re.match(r'^\d{8,12}$', mrn_str):
            return mrn_str
        return "" 