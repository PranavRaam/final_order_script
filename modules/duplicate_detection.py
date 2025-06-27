"""
Duplicate Detection Module
Identifies and handles duplicate patient entries in extracted healthcare data
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from .data_structures import PatientData, ParsedResult
import logging
from difflib import SequenceMatcher

class DuplicateDetector:
    """Detects and handles duplicate patient entries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Similarity thresholds
        self.name_similarity_threshold = 0.85
        self.dob_exact_match_required = True
        self.mrn_exact_match_required = True
        self.ssn_exact_match_required = True
        
        # Weights for different matching criteria
        self.match_weights = {
            'name': 0.4,
            'dob': 0.3,
            'mrn': 0.2,
            'ssn': 0.1
        }
    
    def detect_duplicates(self, parsed_results: List[ParsedResult]) -> Dict[str, List[ParsedResult]]:
        """
        Detect duplicate patients in parsed results
        Returns a dictionary mapping unique patient keys to lists of duplicate results
        """
        patient_groups = {}
        
        for result in parsed_results:
            if not result.patient_data:
                continue
            
            # Generate potential duplicate keys for this patient
            duplicate_keys = self._generate_duplicate_keys(result.patient_data)
            
            # Check if this patient matches any existing groups
            matched_group = None
            for key in duplicate_keys:
                if key in patient_groups:
                    matched_group = key
                    break
            
            if matched_group:
                # Add to existing group
                patient_groups[matched_group].append(result)
            else:
                # Create new group with the first non-empty key
                primary_key = next((key for key in duplicate_keys if key), None)
                if primary_key:
                    patient_groups[primary_key] = [result]
        
        # Filter to only return groups with actual duplicates
        duplicates = {k: v for k, v in patient_groups.items() if len(v) > 1}
        
        if duplicates:
            self.logger.info(f"Found {len(duplicates)} groups with duplicate patients")
            for key, group in duplicates.items():
                self.logger.info(f"Duplicate group '{key}': {len(group)} entries")
        
        return duplicates
    
    def _generate_duplicate_keys(self, patient_data: 'PatientData') -> List[str]:
        """Generate potential duplicate keys for a patient"""
        keys = []
        
        # Normalize patient data for comparison
        normalized_name = self._normalize_name(f"{patient_data.patient_fname} {patient_data.patient_lname}".strip())
        normalized_dob = self._normalize_date(patient_data.dob or "")
        normalized_mrn = self._normalize_identifier(patient_data.medical_record_no or "")
        normalized_ssn = self._normalize_ssn(patient_data.ssn or "")
        
        # Generate keys based on different combinations
        if normalized_name and normalized_dob:
            keys.append(f"name_dob:{normalized_name}:{normalized_dob}")
        
        if normalized_mrn:
            keys.append(f"mrn:{normalized_mrn}")
        
        if normalized_ssn:
            keys.append(f"ssn:{normalized_ssn}")
        
        # Additional keys for partial matches
        if normalized_name:
            # Last name + DOB
            last_name = patient_data.patient_lname.strip() if patient_data.patient_lname else ""
            if last_name and normalized_dob:
                normalized_last = self._normalize_name(last_name)
                keys.append(f"lastname_dob:{normalized_last}:{normalized_dob}")
        
        return keys
    
    def _normalize_name(self, name: str) -> str:
        """Normalize patient name for comparison"""
        if not name:
            return ""
        
        # Convert to uppercase and remove extra whitespace
        name = re.sub(r'\s+', ' ', name.upper().strip())
        
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(MR|MRS|MS|DR|JR|SR|II|III|IV)\b\.?', '', name)
        
        # Remove punctuation except hyphens and apostrophes
        name = re.sub(r'[^\w\s\-\']', '', name)
        
        # Handle common name variations
        name = re.sub(r'\bWILLIAM\b', 'BILL', name)
        name = re.sub(r'\bROBERT\b', 'BOB', name)
        name = re.sub(r'\bMICHAEL\b', 'MIKE', name)
        name = re.sub(r'\bJAMES\b', 'JIM', name)
        
        return name.strip()
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date for comparison"""
        if not date_str:
            return ""
        
        # Extract digits only
        digits = re.sub(r'[^\d]', '', date_str)
        
        # Try to parse as MMDDYYYY or MMDDYY
        if len(digits) == 8:  # MMDDYYYY
            return f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
        elif len(digits) == 6:  # MMDDYY
            year = int(digits[4:])
            # Convert 2-digit year to 4-digit (assume 1900-2099)
            if year > 50:
                year += 1900
            else:
                year += 2000
            return f"{digits[:2]}/{digits[2:4]}/{year}"
        
        # Try other common formats
        date_match = re.search(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})', date_str)
        if date_match:
            month, day, year = date_match.groups()
            if len(year) == 2:
                year = int(year)
                if year > 50:
                    year += 1900
                else:
                    year += 2000
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
        
        return ""
    
    def _normalize_identifier(self, identifier: str) -> str:
        """Normalize identifiers like MRN"""
        if not identifier:
            return ""
        
        # Remove all non-alphanumeric characters and convert to uppercase
        normalized = re.sub(r'[^A-Za-z0-9]', '', identifier).upper()
        
        return normalized
    
    def _normalize_ssn(self, ssn: str) -> str:
        """Normalize SSN for comparison"""
        if not ssn:
            return ""
        
        # Extract digits only
        digits = re.sub(r'[^\d]', '', ssn)
        
        # Validate SSN format
        if len(digits) == 9 and not digits.startswith('000'):
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        
        return ""
    
    def find_similar_patients(self, patient_data: PatientData, existing_patients: List[PatientData]) -> List[Tuple[PatientData, float]]:
        """
        Find patients similar to the given patient data
        Returns list of (patient, similarity_score) tuples
        """
        similar_patients = []
        
        for existing_patient in existing_patients:
            similarity_score = self._calculate_similarity(patient_data, existing_patient)
            
            if similarity_score > 0.7:  # Threshold for considering similarity
                similar_patients.append((existing_patient, similarity_score))
        
        # Sort by similarity score (highest first)
        similar_patients.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patients
    
    def _calculate_similarity(self, patient1: PatientData, patient2: PatientData) -> float:
        """Calculate similarity score between two patients"""
        total_score = 0.0
        total_weight = 0.0
        
        # Name similarity
        if patient1.patient_fname and patient2.patient_fname:
            name_sim = self._calculate_name_similarity(patient1.patient_fname, patient2.patient_fname)
            total_score += name_sim * self.match_weights['name']
            total_weight += self.match_weights['name']
        
        # DOB exact match
        if patient1.dob and patient2.dob:
            dob_norm1 = self._normalize_date(patient1.dob)
            dob_norm2 = self._normalize_date(patient2.dob)
            if dob_norm1 and dob_norm2:
                dob_match = 1.0 if dob_norm1 == dob_norm2 else 0.0
                total_score += dob_match * self.match_weights['dob']
                total_weight += self.match_weights['dob']
        
        # MRN exact match
        if patient1.medical_record_no and patient2.medical_record_no:
            mrn_norm1 = self._normalize_identifier(patient1.medical_record_no)
            mrn_norm2 = self._normalize_identifier(patient2.medical_record_no)
            if mrn_norm1 and mrn_norm2:
                mrn_match = 1.0 if mrn_norm1 == mrn_norm2 else 0.0
                total_score += mrn_match * self.match_weights['mrn']
                total_weight += self.match_weights['mrn']
        
        # SSN exact match
        if patient1.ssn and patient2.ssn:
            ssn_norm1 = self._normalize_ssn(patient1.ssn)
            ssn_norm2 = self._normalize_ssn(patient2.ssn)
            if ssn_norm1 and ssn_norm2:
                ssn_match = 1.0 if ssn_norm1 == ssn_norm2 else 0.0
                total_score += ssn_match * self.match_weights['ssn']
                total_weight += self.match_weights['ssn']
        
        # Return weighted average
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        norm_name1 = self._normalize_name(name1)
        norm_name2 = self._normalize_name(name2)
        
        if not norm_name1 or not norm_name2:
            return 0.0
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, norm_name1, norm_name2).ratio()
        
        # Boost score for exact matches on last name
        name1_parts = norm_name1.split()
        name2_parts = norm_name2.split()
        
        if len(name1_parts) > 0 and len(name2_parts) > 0:
            # Assume last name is the last part
            if name1_parts[-1] == name2_parts[-1]:
                similarity = min(1.0, similarity + 0.2)
        
        return similarity
    
    def merge_duplicate_results(self, duplicate_group: List[ParsedResult]) -> ParsedResult:
        """
        Merge duplicate results into a single result
        Keeps the result with highest confidence, but merges missing data
        """
        if not duplicate_group:
            return None
        
        if len(duplicate_group) == 1:
            return duplicate_group[0]
        
        # Sort by confidence score (highest first)
        sorted_results = sorted(duplicate_group, 
                              key=lambda x: x.confidence_score or 0.0, 
                              reverse=True)
        
        # Start with the highest confidence result
        merged_result = sorted_results[0]
        
        # Merge missing fields from other results
        for other_result in sorted_results[1:]:
            if other_result.patient_data:
                merged_result.patient_data = self._merge_patient_data(
                    merged_result.patient_data, 
                    other_result.patient_data
                )
            
            if other_result.order_data:
                # For order data, we might want to keep separate orders
                # or merge them depending on the use case
                pass
        
        # Add note about merge
        if hasattr(merged_result, 'processing_notes'):
            merged_result.processing_notes = (merged_result.processing_notes or "") + \
                f" [MERGED from {len(duplicate_group)} duplicate entries]"
        
        self.logger.info(f"Merged {len(duplicate_group)} duplicate results for patient")
        
        return merged_result
    
    def _merge_patient_data(self, primary: PatientData, secondary: PatientData) -> PatientData:
        """Merge patient data, filling in missing fields from secondary"""
        if not primary:
            return secondary
        if not secondary:
            return primary
        
        # Create a copy of primary data
        merged = PatientData()
        
        # Copy all fields from primary, then fill in missing ones from secondary
        for field_name in primary.__dict__:
            primary_value = getattr(primary, field_name, None)
            secondary_value = getattr(secondary, field_name, None)
            
            # Use primary value if it exists and is not empty, otherwise use secondary
            if primary_value and str(primary_value).strip():
                setattr(merged, field_name, primary_value)
            elif secondary_value and str(secondary_value).strip():
                setattr(merged, field_name, secondary_value)
            else:
                setattr(merged, field_name, primary_value)
        
        return merged
    
    def generate_duplicate_report(self, duplicates: Dict[str, List[ParsedResult]]) -> str:
        """Generate a report of duplicate patients found"""
        if not duplicates:
            return "No duplicate patients found."
        
        report = f"DUPLICATE PATIENT REPORT\n{'='*50}\n\n"
        report += f"Found {len(duplicates)} groups with duplicate patients:\n\n"
        
        for i, (key, group) in enumerate(duplicates.items(), 1):
            report += f"{i}. Group: {key}\n"
            report += f"   Number of duplicates: {len(group)}\n"
            
            for j, result in enumerate(group):
                if result.patient_data:
                    name = result.patient_data.patient_fname or "Unknown"
                    dob = result.patient_data.dob or "Unknown"
                    mrn = result.patient_data.medical_record_no or "Unknown"
                    doc_id = result.doc_id or "Unknown"
                    confidence = result.confidence_score or 0.0
                    
                    report += f"   {j+1}. Document: {doc_id}\n"
                    report += f"      Name: {name}\n"
                    report += f"      DOB: {dob}\n"
                    report += f"      MRN: {mrn}\n"
                    report += f"      Confidence: {confidence:.2f}\n"
            
            report += "\n"
        
        return report 