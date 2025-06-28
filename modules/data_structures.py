"""
Data Structures Module - Healthcare Document Data Models
Defines all data classes and structures for patient and order information
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class EpisodeDiagnosis:
    """Represents a single episode diagnosis with ICD coding"""
    diagnosis_code: str = ""
    diagnosis_description: str = ""
    diagnosis_type: str = ""  # "primary", "secondary", "other"
    icd_version: str = ""  # "ICD-9", "ICD-10"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "diagnosis_code": self.diagnosis_code,
            "diagnosis_description": self.diagnosis_description,
            "diagnosis_type": self.diagnosis_type,
            "icd_version": self.icd_version
        }

@dataclass
class PatientData:
    """Comprehensive patient information structure with 25+ fields"""
    
    # Basic Demographics
    patient_fname: str = ""
    patient_lname: str = ""
    patient_mname: str = ""
    dob: str = ""
    patient_sex: str = ""
    ssn: str = ""
    
    # Medical Identifiers
    medical_record_no: str = ""
    account_number: str = ""
    
    # Contact Information
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    phone_number: str = ""
    email: str = ""
    
    # Emergency Contact
    emergency_contact_name: str = ""
    emergency_contact_phone: str = ""
    emergency_contact_relationship: str = ""
    
    # Insurance Information
    primary_insurance: str = ""
    secondary_insurance: str = ""
    policy_number: str = ""
    group_number: str = ""
    subscriber_id: str = ""
    
    # Provider Information
    provider_name: str = ""
    provider_npi: str = ""
    referring_physician: str = ""
    
    # Additional Demographics
    marital_status: str = ""
    preferred_language: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "patient_fname": self.patient_fname,
            "patient_lname": self.patient_lname,
            "patient_mname": self.patient_mname,
            "dob": self.dob,
            "patient_sex": self.patient_sex,
            "ssn": self.ssn,
            "medical_record_no": self.medical_record_no,
            "account_number": self.account_number,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "phone_number": self.phone_number,
            "email": self.email,
            "emergency_contact_name": self.emergency_contact_name,
            "emergency_contact_phone": self.emergency_contact_phone,
            "emergency_contact_relationship": self.emergency_contact_relationship,
            "primary_insurance": self.primary_insurance,
            "secondary_insurance": self.secondary_insurance,
            "policy_number": self.policy_number,
            "group_number": self.group_number,
            "subscriber_id": self.subscriber_id,
            "provider_name": self.provider_name,
            "provider_npi": self.provider_npi,
            "referring_physician": self.referring_physician,
            "marital_status": self.marital_status,
            "preferred_language": self.preferred_language
        }
    
    def get_full_name(self) -> str:
        """Get patient's full name"""
        parts = [self.patient_fname, self.patient_mname, self.patient_lname]
        return " ".join(part for part in parts if part.strip())
    
    def calculate_age(self) -> int:
        """Calculate patient age from DOB"""
        if not self.dob:
            return 0
        
        try:
            # Try different date formats
            for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y']:
                try:
                    birth_date = datetime.strptime(self.dob, fmt)
                    today = datetime.now()
                    age = today.year - birth_date.year
                    if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                        age -= 1
                    return max(0, age)
                except ValueError:
                    continue
            return 0
        except:
            return 0

@dataclass
class OrderData:
    """Comprehensive order and episode information structure with 30+ fields"""
    
    # Order Information
    order_no: str = ""
    order_date: str = ""
    
    # Episode Dates
    episode_start_date: str = ""
    episode_end_date: str = ""
    start_of_care: str = ""
    signed_by_physician_date: str = ""
    
    # Physician Information
    physician_name: str = ""
    physician_npi: str = ""
    physician_phone: str = ""
    ordering_facility: str = ""
    
    # Service Information
    service_type: str = ""
    frequency: str = ""
    duration: str = ""
    
    # Diagnosis Information
    episode_diagnoses: List[EpisodeDiagnosis] = field(default_factory=list)
    primary_diagnosis: str = ""
    secondary_diagnosis: str = ""
    
    # Clinical Information
    clinical_notes: str = ""
    authorization_number: str = ""
    equipment_needed: str = ""
    special_instructions: str = ""
    discharge_notes: str = ""
    goals_of_care: str = ""
    functional_limitations: str = ""
    safety_measures: str = ""
    caregiver_instructions: str = ""
    medication_reconciliation: str = ""
    follow_up_instructions: str = ""
    
    # Administrative
    billing_code: str = ""
    place_of_service: str = ""
    insurance_payer: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "order_no": self.order_no,
            "order_date": self.order_date,
            "episode_start_date": self.episode_start_date,
            "episode_end_date": self.episode_end_date,
            "start_of_care": self.start_of_care,
            "signed_by_physician_date": self.signed_by_physician_date,
            "physician_name": self.physician_name,
            "physician_npi": self.physician_npi,
            "physician_phone": self.physician_phone,
            "ordering_facility": self.ordering_facility,
            "service_type": self.service_type,
            "frequency": self.frequency,
            "duration": self.duration,
            "episode_diagnoses": [diagnosis.to_dict() for diagnosis in self.episode_diagnoses],
            "primary_diagnosis": self.primary_diagnosis,
            "secondary_diagnosis": self.secondary_diagnosis,
            "clinical_notes": self.clinical_notes,
            "authorization_number": self.authorization_number,
            "equipment_needed": self.equipment_needed,
            "special_instructions": self.special_instructions,
            "discharge_notes": self.discharge_notes,
            "goals_of_care": self.goals_of_care,
            "functional_limitations": self.functional_limitations,
            "safety_measures": self.safety_measures,
            "caregiver_instructions": self.caregiver_instructions,
            "medication_reconciliation": self.medication_reconciliation,
            "follow_up_instructions": self.follow_up_instructions,
            "billing_code": self.billing_code,
            "place_of_service": self.place_of_service,
            "insurance_payer": self.insurance_payer
        }
    
    def get_all_diagnoses_text(self) -> str:
        """Get all diagnoses as concatenated text"""
        diagnoses_text = []
        
        if self.primary_diagnosis:
            diagnoses_text.append(f"Primary: {self.primary_diagnosis}")
        
        if self.secondary_diagnosis:
            diagnoses_text.append(f"Secondary: {self.secondary_diagnosis}")
        
        for diagnosis in self.episode_diagnoses:
            if diagnosis.diagnosis_description:
                prefix = f"{diagnosis.diagnosis_type.title()}: " if diagnosis.diagnosis_type else ""
                code_suffix = f" ({diagnosis.diagnosis_code})" if diagnosis.diagnosis_code else ""
                diagnoses_text.append(f"{prefix}{diagnosis.diagnosis_description}{code_suffix}")
        
        return "; ".join(diagnoses_text)
    
    def get_date_range(self) -> str:
        """Get formatted date range for the episode"""
        if self.episode_start_date and self.episode_end_date:
            return f"{self.episode_start_date} to {self.episode_end_date}"
        elif self.episode_start_date:
            return f"From {self.episode_start_date}"
        elif self.episode_end_date:
            return f"Until {self.episode_end_date}"
        else:
            return ""

@dataclass
class ParsedResult:
    """Complete parsing result with enhanced metadata"""
    doc_id: str
    patient_data: PatientData
    order_data: OrderData
    source: str  # "structured" or "llm"
    status: str  # "parsed" or "failed"
    error: str = ""
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    extraction_method: str = ""
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "doc_id": self.doc_id,
            "patient_data": self.patient_data.to_dict(),
            "order_data": self.order_data.to_dict(),
            "source": self.source,
            "status": self.status,
            "error": self.error,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "extraction_method": self.extraction_method,
            "processing_time": self.processing_time
        }
    
    def get_summary(self) -> str:
        """Get a summary of the parsed result"""
        if self.status != "parsed":
            return f"Failed: {self.error}"
        
        patient_name = self.patient_data.get_full_name()
        order_info = f"Order #{self.order_data.order_no}" if self.order_data.order_no else "No order number"
        
        return f"{patient_name} - {order_info} (Confidence: {self.confidence_score:.2f})"

# Quality assessment helpers
def calculate_patient_completeness(patient: PatientData) -> float:
    """Calculate completeness score for patient data"""
    patient_dict = patient.to_dict()
    filled_fields = sum(1 for value in patient_dict.values() if str(value).strip())
    total_fields = len(patient_dict)
    return filled_fields / total_fields if total_fields > 0 else 0.0

def calculate_order_completeness(order: OrderData) -> float:
    """Calculate completeness score for order data"""
    order_dict = order.to_dict()
    
    # Count filled fields (excluding episode_diagnoses list)
    filled_fields = 0
    total_fields = 0
    
    for key, value in order_dict.items():
        if key == "episode_diagnoses":
            # Count diagnosis list separately
            total_fields += 1
            if isinstance(value, list) and len(value) > 0:
                filled_fields += 1
        else:
            total_fields += 1
            if str(value).strip():
                filled_fields += 1
    
    return filled_fields / total_fields if total_fields > 0 else 0.0

def get_critical_patient_fields() -> List[str]:
    """Get list of critical patient fields for validation"""
    return [
        'patient_fname',
        'patient_lname', 
        'dob',
        'medical_record_no',
        'patient_sex'
    ]

def get_critical_order_fields() -> List[str]:
    """Get list of critical order fields for validation"""
    return [
        'order_no',
        'order_date',
        'episode_start_date',
        'start_of_care',
        'physician_name',
        'primary_diagnosis'
    ]

def validate_critical_fields(patient: PatientData, order: OrderData) -> Dict[str, bool]:
    """Validate that critical fields are present"""
    results = {}
    
    # Check critical patient fields
    patient_dict = patient.to_dict()
    for field in get_critical_patient_fields():
        results[f"patient_{field}"] = bool(patient_dict.get(field, "").strip())
    
    # Check critical order fields
    order_dict = order.to_dict()
    for field in get_critical_order_fields():
        if field == "episode_diagnoses":
            results[f"order_{field}"] = len(order.episode_diagnoses) > 0
        else:
            results[f"order_{field}"] = bool(order_dict.get(field, "").strip())
    
    return results 