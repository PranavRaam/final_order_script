"""
Data Parser Module - Main Orchestration (Refactored)
Orchestrates the comprehensive healthcare document parsing using modular components
Keeps under 600 lines by importing functionality from specialized modules
"""

import os
import json
import csv
import logging
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import our modular components
from .data_structures import (
    PatientData, OrderData, EpisodeDiagnosis, ParsedResult,
    calculate_patient_completeness, calculate_order_completeness,
    validate_critical_fields
)
from .field_extraction import FieldExtractor
from .llm_parser import LLMParser
from .data_standardization import DataStandardizer
from .enhanced_field_extraction import EnhancedFieldExtractor
from .duplicate_detection import DuplicateDetector
from .text_preprocessor import preprocess_text, split_sections

# Parsing Configuration - Adjust these for better performance
PARSING_CONFIG = {
    # Success criteria thresholds
    'MIN_CONFIDENCE': 0.15,
    'MIN_COMPLETENESS': 0.1,
    'HIGH_CONFIDENCE_THRESHOLD': 0.3,
    'HIGH_COMPLETENESS_THRESHOLD': 0.2,
    
    # Fallback criteria (more lenient)
    'FALLBACK_MIN_CONFIDENCE': 0.1,
    'FALLBACK_MIN_COMPLETENESS': 0.15,
    
    # LLM settings
    'LLM_TIMEOUT': 30,
    'LLM_RETRY_COUNT': 2,
    'LLM_FAST_MODE': True,
    
    # Debug settings
    'VERBOSE_LOGGING': True,
    
    # Duplicate handling
    'MERGE_DUPLICATES': True,
    
    # If True, always attempt an LLM pass after structured parsing and merge
    'ALWAYS_RUN_LLM': False,
    # Minimum completeness required to skip the extra LLM pass (only relevant when ALWAYS_RUN_LLM=False)
    'TARGET_COMPLETENESS': 0.95,
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredParser:
    """Enhanced structured parser using FieldExtractor for pattern-based extraction"""
    
    def __init__(self):
        self.field_extractor = FieldExtractor()
        self.logger = logging.getLogger(__name__)
    
    def extract_patient_data(self, text: str) -> PatientData:
        """Extract comprehensive patient data using field patterns"""
        self.logger.debug("🔍 Extracting patient data using structured patterns")
        
        # Create PatientData object
        patient_data = PatientData()
        
        # Extract ALL patient fields
        patient_data.patient_fname = self.field_extractor.extract_field(text, 'patient_fname') or ""
        patient_data.patient_lname = self.field_extractor.extract_field(text, 'patient_lname') or ""
        patient_data.patient_mname = self.field_extractor.extract_field(text, 'patient_mname') or ""
        patient_data.dob = self.field_extractor.extract_field(text, 'dob') or ""
        patient_data.patient_sex = self.field_extractor.extract_field(text, 'patient_sex') or ""
        patient_data.ssn = self.field_extractor.extract_field(text, 'ssn') or ""
        patient_data.medical_record_no = self.field_extractor.extract_field(text, 'medical_record_no') or ""
        patient_data.account_number = self.field_extractor.extract_field(text, 'account_number') or ""
        patient_data.address = self.field_extractor.extract_field(text, 'address') or ""
        patient_data.city = self.field_extractor.extract_field(text, 'city') or ""
        patient_data.state = self.field_extractor.extract_field(text, 'state') or ""
        patient_data.zip_code = self.field_extractor.extract_field(text, 'zip_code') or ""
        patient_data.phone_number = self.field_extractor.extract_field(text, 'phone_number') or ""
        patient_data.email = self.field_extractor.extract_field(text, 'email') or ""
        patient_data.emergency_contact_name = self.field_extractor.extract_field(text, 'emergency_contact_name') or ""
        patient_data.emergency_contact_phone = self.field_extractor.extract_field(text, 'emergency_contact_phone') or ""
        patient_data.emergency_contact_relationship = self.field_extractor.extract_field(text, 'emergency_contact_relationship') or ""
        patient_data.primary_insurance = self.field_extractor.extract_field(text, 'primary_insurance') or ""
        patient_data.secondary_insurance = self.field_extractor.extract_field(text, 'secondary_insurance') or ""
        patient_data.policy_number = self.field_extractor.extract_field(text, 'policy_number') or ""
        patient_data.group_number = self.field_extractor.extract_field(text, 'group_number') or ""
        patient_data.subscriber_id = self.field_extractor.extract_field(text, 'subscriber_id') or ""
        patient_data.provider_name = self.field_extractor.extract_field(text, 'provider_name') or ""
        patient_data.provider_npi = self.field_extractor.extract_field(text, 'provider_npi') or ""
        patient_data.referring_physician = self.field_extractor.extract_field(text, 'referring_physician') or ""
        patient_data.marital_status = self.field_extractor.extract_field(text, 'marital_status') or ""
        patient_data.preferred_language = self.field_extractor.extract_field(text, 'preferred_language') or ""
        
        # Add calculated age if we have DOB
        if patient_data.dob:
            age = patient_data.calculate_age()
            if age > 0:
                self.logger.debug(f"Calculated age: {age} years")
        
        return patient_data
    
    def extract_order_data(self, text: str) -> OrderData:
        """Extract comprehensive order data using field patterns"""
        self.logger.debug("🔍 Extracting order data using structured patterns")
        
        # Create OrderData object
        order_data = OrderData()
        
        # Extract ALL order fields
        order_data.order_no = self.field_extractor.extract_field(text, 'order_no') or ""
        order_data.order_date = self.field_extractor.extract_field(text, 'order_date') or ""
        order_data.episode_start_date = self.field_extractor.extract_field(text, 'episode_start_date') or ""
        order_data.episode_end_date = self.field_extractor.extract_field(text, 'episode_end_date') or ""
        order_data.start_of_care = self.field_extractor.extract_field(text, 'start_of_care') or ""
        order_data.signed_by_physician_date = self.field_extractor.extract_field(text, 'signed_by_physician_date') or ""
        order_data.physician_name = self.field_extractor.extract_field(text, 'physician_name') or ""
        order_data.physician_npi = self.field_extractor.extract_field(text, 'physician_npi') or ""
        order_data.physician_phone = self.field_extractor.extract_field(text, 'physician_phone') or ""
        order_data.ordering_facility = self.field_extractor.extract_field(text, 'ordering_facility') or ""
        order_data.service_type = self.field_extractor.extract_field(text, 'service_type') or ""
        order_data.frequency = self.field_extractor.extract_field(text, 'frequency') or ""
        order_data.duration = self.field_extractor.extract_field(text, 'duration') or ""
        order_data.primary_diagnosis = self.field_extractor.extract_field(text, 'primary_diagnosis') or ""
        order_data.secondary_diagnosis = self.field_extractor.extract_field(text, 'secondary_diagnosis') or ""
        order_data.clinical_notes = self.field_extractor.extract_field(text, 'clinical_notes') or ""
        order_data.authorization_number = self.field_extractor.extract_field(text, 'authorization_number') or ""
        order_data.equipment_needed = self.field_extractor.extract_field(text, 'equipment_needed') or ""
        order_data.special_instructions = self.field_extractor.extract_field(text, 'special_instructions') or ""
        order_data.discharge_notes = self.field_extractor.extract_field(text, 'discharge_notes') or ""
        order_data.goals_of_care = self.field_extractor.extract_field(text, 'goals_of_care') or ""
        order_data.functional_limitations = self.field_extractor.extract_field(text, 'functional_limitations') or ""
        order_data.safety_measures = self.field_extractor.extract_field(text, 'safety_measures') or ""
        order_data.caregiver_instructions = self.field_extractor.extract_field(text, 'caregiver_instructions') or ""
        order_data.medication_reconciliation = self.field_extractor.extract_field(text, 'medication_reconciliation') or ""
        order_data.follow_up_instructions = self.field_extractor.extract_field(text, 'follow_up_instructions') or ""
        order_data.billing_code = self.field_extractor.extract_field(text, 'billing_code') or ""
        order_data.place_of_service = self.field_extractor.extract_field(text, 'place_of_service') or ""
        order_data.insurance_payer = self.field_extractor.extract_field(text, 'insurance_payer') or ""
        
        # Extract diagnoses separately
        diagnoses = self.field_extractor.extract_diagnoses(text)
        if diagnoses:
            order_data.episode_diagnoses = diagnoses
            self.logger.debug(f"Extracted {len(diagnoses)} diagnoses")
        
        return order_data

class DataParser:
    """Main data parser orchestrating structured and LLM-based extraction"""
    
    def __init__(self, field_extractor: FieldExtractor, 
                 llm_parser: Optional['LLMParser'] = None,
                 config: Optional[Dict] = None):
        self.field_extractor = field_extractor
        self.llm_parser = llm_parser
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Allow external config to override PARSING_CONFIG defaults
        if self.config:
            for k, v in self.config.items():
                if k in PARSING_CONFIG:
                    PARSING_CONFIG[k] = v
        
        # Initialize new components
        self.data_standardizer = DataStandardizer()
        self.enhanced_extractor = EnhancedFieldExtractor()
        self.duplicate_detector = DuplicateDetector()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'llm_fallback_used': 0,
            'duplicates_found': 0,
            'fields_enhanced': 0
        }
        
        # Initialize parsers
        self.structured_parser = StructuredParser()
        
        # Primary (fast) LLM parser
        self.llm_parser_fast = llm_parser
        # Slow LLM parser for fallback (if Ollama available)
        try:
            if llm_parser:
                self.llm_parser_slow = LLMParser(ollama_url=llm_parser.ollama_url, model_name=llm_parser.model_name, fast_mode=False)
            else:
                self.llm_parser_slow = None
        except Exception:
            self.llm_parser_slow = None
            
        self.logger.info(f"🚀 DataParser initialized (LLM fallback: {bool(llm_parser)})")
    
    def _calculate_confidence_score(self, patient: PatientData, order: OrderData, source: str) -> float:
        """Calculate confidence score based on completeness and source"""
        
        # Base confidence by source
        base_confidence = 0.8 if source == "structured" else 0.7
        
        # Calculate completeness scores
        patient_completeness = calculate_patient_completeness(patient)
        order_completeness = calculate_order_completeness(order)
        
        # Validate critical fields
        critical_validation = validate_critical_fields(patient, order)
        critical_score = sum(critical_validation.values()) / len(critical_validation)
        
        # Combined score
        data_quality = (patient_completeness * 0.4) + (order_completeness * 0.4) + (critical_score * 0.2)
        final_confidence = base_confidence * data_quality
        
        return min(final_confidence, 1.0)
    
    def _calculate_completeness_score(self, patient: PatientData, order: OrderData) -> float:
        """Calculate overall completeness score"""
        patient_completeness = calculate_patient_completeness(patient)
        order_completeness = calculate_order_completeness(order)
        return (patient_completeness * 0.6) + (order_completeness * 0.4)
    
    def parse_document(self, extraction_result: Dict) -> ParsedResult:
        """Parse a single document with enhanced accuracy"""
        doc_id = extraction_result.get('doc_id', 'unknown')
        text = extraction_result.get('text', '')
        status = extraction_result.get('status', 'failed')
        
        # Preprocess raw OCR text to improve downstream parsing
        if text and text.strip():
            text = preprocess_text(text)
        
        # Split into logical sections for focused parsing
        sections = split_sections(text)
        header_chunk = sections.get('HEADER', '')
        orders_chunk = sections.get('ORDERS', '')
        diagnoses_chunk = sections.get('DIAGNOSES', '')
        
        # Fallback: if a chunk is missing, use full text
        patient_text_for_structured = header_chunk if header_chunk else text
        order_text_for_structured = orders_chunk if orders_chunk else text

        start_time = time.time()
        self.logger.info(f"🔍 Starting parsing for document {doc_id}")
        
        if status != 'extracted' or not text.strip():
            return ParsedResult(
                doc_id=doc_id,
                patient_data=PatientData(),
                order_data=OrderData(),
                source="none",
                status="failed",
                error="No valid text extracted",
                confidence_score=0.0,
                completeness_score=0.0,
                extraction_method="none",
                processing_time=time.time() - start_time
            )
        
        try:
            # Step 1: Try structured parsing first
            self.logger.debug(f"🏗️ Attempting structured parsing", doc_id)
            
            patient_data = self.structured_parser.extract_patient_data(patient_text_for_structured)
            order_data = self.structured_parser.extract_order_data(order_text_for_structured)
            
            # Calculate initial scores
            confidence = self._calculate_confidence_score(patient_data, order_data, "structured")
            completeness = self._calculate_completeness_score(patient_data, order_data)
            
            # Debug logging to see what we extracted
            self.logger.info(f"📊 Structured parsing results for {doc_id}:")
            self.logger.info(f"   Patient name: fname='{patient_data.patient_fname}', lname='{patient_data.patient_lname}'")
            self.logger.info(f"   DOB: '{patient_data.dob}'")
            self.logger.info(f"   Confidence: {confidence:.3f}, Completeness: {completeness:.3f}")
            
            # Count non-empty fields for more detailed debug info
            patient_fields = sum(1 for v in patient_data.__dict__.values() if v is not None and str(v).strip())
            order_fields = sum(1 for v in order_data.__dict__.values() if v is not None and str(v).strip() and not isinstance(v, list))
            self.logger.info(f"   Fields filled: Patient={patient_fields}/27, Order={order_fields}/29")
            
            # Determine if structured parsing was successful
            has_name = bool(patient_data.patient_fname or patient_data.patient_lname)
            has_dob = bool(patient_data.dob)
            
            # More flexible success criteria - accept documents with either good name/dob OR high confidence
            basic_success = (
                confidence > PARSING_CONFIG['MIN_CONFIDENCE'] and  # Use config value
                completeness > PARSING_CONFIG['MIN_COMPLETENESS'] and  # Use config value
                (has_name or has_dob)  # Accept if we have EITHER name OR DOB
            )
            
            # High confidence documents can pass even without name/dob
            high_confidence_success = (
                confidence > PARSING_CONFIG['HIGH_CONFIDENCE_THRESHOLD'] and
                completeness > PARSING_CONFIG['HIGH_COMPLETENESS_THRESHOLD']
            )
            
            structured_success = basic_success or high_confidence_success
            
            if PARSING_CONFIG['VERBOSE_LOGGING']:
                self.logger.info(f"   Success criteria: confidence>{PARSING_CONFIG['MIN_CONFIDENCE']}={confidence > PARSING_CONFIG['MIN_CONFIDENCE']}, completeness>{PARSING_CONFIG['MIN_COMPLETENESS']}={completeness > PARSING_CONFIG['MIN_COMPLETENESS']}, has_name={has_name}, has_dob={has_dob}")
                self.logger.info(f"   Basic success: {basic_success}, High confidence success: {high_confidence_success}")
                self.logger.info(f"   Overall success: {structured_success}")
            
            if structured_success and not PARSING_CONFIG.get('ALWAYS_RUN_LLM', False) and completeness >= PARSING_CONFIG.get('TARGET_COMPLETENESS', 0.95):
                processing_time = time.time() - start_time
                self.logger.info(f"✅ Structured parsing successful for {doc_id} (confidence: {confidence:.2f}, completeness: {completeness:.2f}) – no LLM needed")
                return ParsedResult(
                    doc_id=doc_id,
                    patient_data=patient_data,
                    order_data=order_data,
                    source="structured",
                    status="parsed",
                    confidence_score=confidence,
                    completeness_score=completeness,
                    extraction_method="enhanced_structured",
                    processing_time=processing_time
                )
            
            # Step 2: Try LLM fallback if structured parsing failed
            if self.llm_parser_fast and self.llm_parser_fast.available and (not structured_success or PARSING_CONFIG.get('ALWAYS_RUN_LLM', False) or completeness < PARSING_CONFIG.get('TARGET_COMPLETENESS', 0.95)):
                self.logger.info(f"🤖 Structured parsing insufficient for {doc_id}, trying FAST LLM fallback")
                
                try:
                    # Feed only relevant chunks to LLM (header + orders) to improve accuracy and reduce cost
                    llm_input_text = f"{header_chunk}\n{orders_chunk}" if (header_chunk or orders_chunk) else text
                    llm_patient_data, llm_order_data = self.llm_parser_fast.parse_combined_with_llm(llm_input_text, doc_id)
                    
                    # Merge strategy: use structured as baseline, then fill any EMPTY structured field with LLM value
                    def _merge_data(target, source):
                        for attr, val in source.__dict__.items():
                            if isinstance(val, list):
                                if not getattr(target, attr):
                                    setattr(target, attr, val)
                            else:
                                if not getattr(target, attr):
                                    if val:
                                        setattr(target, attr, val)

                    merged_patient = PatientData(**patient_data.__dict__)
                    merged_order = OrderData(**order_data.__dict__)
                    _merge_data(merged_patient, llm_patient_data)
                    _merge_data(merged_order, llm_order_data)

                    merged_confidence = self._calculate_confidence_score(merged_patient, merged_order, "llm")
                    merged_completeness = self._calculate_completeness_score(merged_patient, merged_order)

                    # Extra pass: if episode dates still blank try targeted LLM extraction
                    if (not merged_order.episode_start_date or not merged_order.episode_end_date):
                        start_ep, end_ep = self.llm_parser_fast.parse_episode_dates(orders_chunk if orders_chunk else text, doc_id)
                        if not merged_order.episode_start_date and start_ep:
                            merged_order.episode_start_date = start_ep
                        if not merged_order.episode_end_date and end_ep:
                            merged_order.episode_end_date = end_ep
                        # Re-compute completeness after filling
                        merged_completeness = self._calculate_completeness_score(merged_patient, merged_order)

                    has_llm_data = merged_completeness > completeness or merged_confidence > confidence

                    if has_llm_data and (merged_confidence > confidence or merged_completeness > completeness):
                        processing_time = time.time() - start_time
                        self.logger.info(f"✅ LLM parsing successful for {doc_id} (confidence: {merged_confidence:.2f}, completeness: {merged_completeness:.2f})")
                        return ParsedResult(
                            doc_id=doc_id,
                            patient_data=merged_patient,
                            order_data=merged_order,
                            source="llm",
                            status="parsed",
                            confidence_score=merged_confidence,
                            completeness_score=merged_completeness,
                            extraction_method="enhanced_llm",
                            processing_time=processing_time
                        )
                    else:
                        self.logger.info(f"📊 LLM results not better than structured for {doc_id} (LLM: {merged_confidence:.2f}/{merged_completeness:.2f} vs Structured: {confidence:.2f}/{completeness:.2f})")
                        
                except Exception as e:
                    self.logger.error(f"🚨 LLM fallback failed for {doc_id}: {e}")
                    # Continue to use structured results even if LLM fails
            
            # Use structured results as fallback (more lenient acceptance)
            processing_time = time.time() - start_time
            
            if structured_success:
                self.logger.info(f"✅ Using structured parsing results for {doc_id}")
                return ParsedResult(
                    doc_id=doc_id,
                    patient_data=patient_data,
                    order_data=order_data,
                    source="structured",
                    status="parsed",
                    confidence_score=confidence,
                    completeness_score=completeness,
                    extraction_method="enhanced_structured",
                    processing_time=processing_time
                )
            else:
                # Use structured output with limited confidence - be more lenient
                if confidence > PARSING_CONFIG['FALLBACK_MIN_CONFIDENCE'] or completeness > PARSING_CONFIG['FALLBACK_MIN_COMPLETENESS']:  # Use config values
                    self.logger.warning(f"⚠️ Using structured output with limited confidence for {doc_id}")
                    return ParsedResult(
                        doc_id=doc_id,
                        patient_data=patient_data,
                        order_data=order_data,
                        source="structured",
                        status="parsed",
                        confidence_score=confidence,
                        completeness_score=completeness,
                        extraction_method="enhanced_structured_fallback",
                        error="Low confidence extraction" if confidence <= PARSING_CONFIG['HIGH_CONFIDENCE_THRESHOLD'] * 0.8 else "",
                        processing_time=processing_time
                    )
                else:
                    # Complete failure only if extremely low scores
                    self.logger.warning(f"❌ Failed to parse {doc_id}: Low confidence extraction")
                    return ParsedResult(
                        doc_id=doc_id,
                        patient_data=patient_data,
                        order_data=order_data,
                        source="structured",
                        status="failed",
                        confidence_score=confidence,
                        completeness_score=completeness,
                        extraction_method="enhanced_structured_failed",
                        error="Low confidence extraction",
                        processing_time=processing_time
                    )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Parsing error: {str(e)}"
            self.logger.error(error_msg, doc_id)
            return ParsedResult(
                doc_id=doc_id,
                patient_data=PatientData(),
                order_data=OrderData(),
                source="none",
                status="failed",
                error=error_msg,
                confidence_score=0.0,
                completeness_score=0.0,
                extraction_method="failed",
                processing_time=processing_time
            )
    
    def parse_documents(self, extraction_results: List[Dict]) -> List[ParsedResult]:
        """Parse multiple documents with enhanced processing"""
        total_docs = len(extraction_results)
        self.logger.info(f"🚀 Starting parsing of {total_docs} documents")
        
        results = []
        successful = 0
        structured_count = 0
        llm_count = 0
        total_processing_time = 0.0
        
        for i, extraction_result in enumerate(extraction_results, 1):
            doc_id = extraction_result.get('doc_id', f'doc_{i}')
            
            self.logger.info(f"📄 Processing document {i}/{total_docs}: {doc_id}")
            
            result = self.parse_document(extraction_result)
            results.append(result)
            total_processing_time += result.processing_time
            
            if result.status == 'parsed':
                successful += 1
                if result.source == 'structured':
                    structured_count += 1
                elif result.source == 'llm':
                    llm_count += 1
                    
                self.logger.info(f"✅ Parsed successfully for {doc_id} (method: {result.extraction_method}, confidence: {result.confidence_score:.2f})")
            else:
                self.logger.error(f"❌ Parsing failed: {result.error}", doc_id)
        
        # Summary statistics
        success_rate = (successful / total_docs) * 100 if total_docs > 0 else 0
        avg_confidence = sum(r.confidence_score for r in results if r.status == 'parsed') / max(successful, 1)
        avg_completeness = sum(r.completeness_score for r in results if r.status == 'parsed') / max(successful, 1)
        avg_processing_time = total_processing_time / total_docs if total_docs > 0 else 0
        
        self.logger.info(f"📊 Parsing complete:")
        self.logger.info(f"   Success rate: {success_rate:.1f}% ({successful}/{total_docs})")
        self.logger.info(f"   Average confidence: {avg_confidence:.3f}")
        self.logger.info(f"   Average completeness: {avg_completeness:.3f}")
        self.logger.info(f"   Average processing time: {avg_processing_time:.2f}s per document")
        self.logger.info(f"   Methods used: {structured_count} structured, {llm_count} LLM")
        
        return results

    def enhance_extracted_data(self, result: ParsedResult, text: str) -> ParsedResult:
        """Enhance extracted data with better field extraction and standardization"""
        if not result or not result.patient_data:
            return result
        
        enhanced_fields_count = 0
        
        # Extract critical missing fields using enhanced patterns
        enhanced_fields = self.enhanced_extractor.extract_all_critical_fields(text)
        
        # Enhance patient data with missing critical fields
        patient_data = result.patient_data
        
        # DOB enhancement
        if not patient_data.dob and enhanced_fields['dob']:
            patient_data.dob = enhanced_fields['dob']
            enhanced_fields_count += 1
        
        # Standardize DOB format
        if patient_data.dob:
            standardized_dob = self.data_standardizer.standardize_date(patient_data.dob)
            if standardized_dob != patient_data.dob:
                patient_data.dob = standardized_dob
                enhanced_fields_count += 1
        
        # SSN enhancement
        if not patient_data.ssn and enhanced_fields['ssn']:
            patient_data.ssn = enhanced_fields['ssn']
            enhanced_fields_count += 1
        
        # Standardize SSN format
        if patient_data.ssn:
            standardized_ssn = self.data_standardizer.standardize_ssn(patient_data.ssn)
            if standardized_ssn != patient_data.ssn:
                patient_data.ssn = standardized_ssn
                enhanced_fields_count += 1
        
        # MRN enhancement and validation
        if not patient_data.medical_record_no and enhanced_fields['mrn']:
            patient_data.medical_record_no = enhanced_fields['mrn']
            enhanced_fields_count += 1
        # Clean/validate MRN
        if patient_data.medical_record_no:
            standardized_mrn = self.data_standardizer.standardize_mrn(patient_data.medical_record_no)
            if standardized_mrn != patient_data.medical_record_no:
                patient_data.medical_record_no = standardized_mrn
                enhanced_fields_count += 1
        
        # 1) If gender is missing try to infer from first name.
        if not patient_data.patient_sex and patient_data.patient_fname:
            inferred_gender = self.data_standardizer._infer_gender_from_name(patient_data.patient_fname)
            if inferred_gender:
                patient_data.patient_sex = inferred_gender
                enhanced_fields_count += 1
        
        # 2) Standardise gender to MALE/FEMALE only.
        if patient_data.patient_sex:
            standardized_gender = self.data_standardizer.standardize_gender(patient_data.patient_sex)
            if standardized_gender != patient_data.patient_sex:
                patient_data.patient_sex = standardized_gender
                enhanced_fields_count += 1
        
        # Phone number standardization
        if patient_data.phone_number:
            standardized_phone = self.data_standardizer.standardize_phone(patient_data.phone_number)
            if standardized_phone != patient_data.phone_number:
                patient_data.phone_number = standardized_phone
                enhanced_fields_count += 1
        
        # Name standardization (first and last names separately)
        if patient_data.patient_fname:
            standardized_fname = self.data_standardizer.standardize_name(patient_data.patient_fname)
            if standardized_fname != patient_data.patient_fname:
                patient_data.patient_fname = standardized_fname
                enhanced_fields_count += 1
        
        if patient_data.patient_lname:
            standardized_lname = self.data_standardizer.standardize_name(patient_data.patient_lname)
            if standardized_lname != patient_data.patient_lname:
                patient_data.patient_lname = standardized_lname
                enhanced_fields_count += 1
        
        # State standardization
        if patient_data.state:
            standardized_state = self.data_standardizer.standardize_state(patient_data.state)
            if standardized_state != patient_data.state:
                patient_data.state = standardized_state
                enhanced_fields_count += 1
        
        # ZIP code standardization
        if patient_data.zip_code:
            standardized_zip = self.data_standardizer.standardize_zip(patient_data.zip_code)
            if standardized_zip != patient_data.zip_code:
                patient_data.zip_code = standardized_zip
                enhanced_fields_count += 1
        
        # Enhance order data if present
        if result.order_data:
            order_data = result.order_data
            
            # Episode dates enhancement
            episode_dates = enhanced_fields['episode_dates']
            if not order_data.episode_start_date and episode_dates.get('start'):
                order_data.episode_start_date = self.data_standardizer.standardize_date(episode_dates['start'])
                enhanced_fields_count += 1
            
            if not order_data.episode_end_date and episode_dates.get('end'):
                order_data.episode_end_date = self.data_standardizer.standardize_date(episode_dates['end'])
                enhanced_fields_count += 1
            
            # NPI enhancement
            if not order_data.physician_npi and enhanced_fields['npi']:
                order_data.physician_npi = enhanced_fields['npi']
                enhanced_fields_count += 1
            
            # Diagnosis enhancement
            if enhanced_fields['diagnoses'] and len(enhanced_fields['diagnoses']) > 0:
                if not order_data.episode_diagnoses:
                    order_data.episode_diagnoses = []
                
                # Add enhanced diagnoses that aren't already present
                existing_descriptions = {diag.diagnosis_description.lower() for diag in order_data.episode_diagnoses if diag.diagnosis_description}
                
                for enhanced_diag in enhanced_fields['diagnoses']:
                    if enhanced_diag.diagnosis_description.lower() not in existing_descriptions:
                        order_data.episode_diagnoses.append(enhanced_diag)
                        enhanced_fields_count += 1
            
            # Insurance enhancement
            if not order_data.insurance_payer and enhanced_fields['insurance']:
                order_data.insurance_payer = enhanced_fields['insurance']
                enhanced_fields_count += 1
        
        # Update statistics
        if enhanced_fields_count > 0:
            self.stats['fields_enhanced'] += enhanced_fields_count
            if PARSING_CONFIG['VERBOSE_LOGGING']:
                self.logger.info(f"Enhanced {enhanced_fields_count} fields for document {result.doc_id}")
        
        return result

    def process_documents(self, documents: List[Dict]) -> List[ParsedResult]:
        """Process multiple documents with enhanced extraction and duplicate detection"""
        results = []
        
        for doc in documents:
            try:
                # Process individual document
                result = self.parse_document(doc)
                if result:
                    # Enhance the result with better field extraction
                    enhanced_result = self.enhance_extracted_data(result, doc.get('text', ''))
                    results.append(enhanced_result)
                    
            except Exception as e:
                self.logger.error(f"Error processing document {doc.get('doc_id', 'unknown')}: {str(e)}")
                # Create failed result
                failed_result = ParsedResult(
                    doc_id=doc.get('doc_id', 'unknown'),
                    patient_data=PatientData(),
                    order_data=OrderData(),
                    source="none",
                    status="failed",
                    error=str(e),
                    confidence_score=0.0,
                    completeness_score=0.0,
                    extraction_method="error",
                    processing_time=0.0
                )
                results.append(failed_result)
        
        # Detect and handle duplicates
        duplicates = self.duplicate_detector.detect_duplicates(results)
        if duplicates:
            self.stats['duplicates_found'] = len(duplicates)
            
            # Generate duplicate report
            duplicate_report = self.duplicate_detector.generate_duplicate_report(duplicates)
            self.logger.warning(f"Duplicate patients detected:\n{duplicate_report}")
            
            # Optionally merge duplicates (can be configured)
            if PARSING_CONFIG.get('MERGE_DUPLICATES', False):
                # Remove duplicates from results and add merged versions
                merged_results = []
                processed_groups = set()
                
                for result in results:
                    # Check if this result is part of a duplicate group
                    is_duplicate = False
                    for group_key, group_results in duplicates.items():
                        if result in group_results and group_key not in processed_groups:
                            # Merge this group
                            merged_result = self.duplicate_detector.merge_duplicate_results(group_results)
                            if merged_result:
                                merged_results.append(merged_result)
                            processed_groups.add(group_key)
                            is_duplicate = True
                            break
                    
                    # If not a duplicate, add as-is
                    if not is_duplicate:
                        merged_results.append(result)
                
                results = merged_results
        
        # Update final statistics
        self.stats['total_processed'] = len(results)
        self.stats['successful_extractions'] = len([r for r in results if r.status == 'parsed'])
        self.stats['failed_extractions'] = len([r for r in results if r.status == 'failed'])
        
        return results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        total = self.stats['total_processed']
        successful = self.stats['successful_extractions']
        
        return {
            'total_documents_processed': total,
            'successful_extractions': successful,
            'failed_extractions': self.stats['failed_extractions'],
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'llm_fallback_usage': self.stats['llm_fallback_used'],
            'llm_fallback_rate': (self.stats['llm_fallback_used'] / total * 100) if total > 0 else 0,
            'duplicates_found': self.stats['duplicates_found'],
            'fields_enhanced': self.stats['fields_enhanced'],
            'average_enhancements_per_document': (self.stats['fields_enhanced'] / total) if total > 0 else 0
        }

# Helper functions for saving data (imported from separate reporting module)
def save_parsed_data(results: List[ParsedResult], output_dir: str) -> str:
    """Save parsed data with enhanced structure"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual JSON files for each document
    json_dir = os.path.join(output_dir, f"parsed_json_{timestamp}")
    os.makedirs(json_dir, exist_ok=True)
    
    for result in results:
        if result.status == 'parsed':
            json_file = os.path.join(json_dir, f"{result.doc_id}.json")
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to save JSON for {result.doc_id}: {e}")
    
    return json_dir

def save_failed_parses(results: List[ParsedResult], output_path: str):
    """Save failed parsing attempts with detailed error analysis"""
    failed_results = [r for r in results if r.status == 'failed']
    
    if not failed_results:
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'doc_id', 'error', 'extraction_method', 'confidence_score', 
            'completeness_score', 'processing_time', 'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in failed_results:
            writer.writerow({
                'doc_id': result.doc_id,
                'error': result.error,
                'extraction_method': result.extraction_method,
                'confidence_score': result.confidence_score,
                'completeness_score': result.completeness_score,
                'processing_time': result.processing_time,
                'timestamp': datetime.now().isoformat()
            })

def save_extraction_csv(results: List[ParsedResult], output_path: str):
    """Save all extracted data to CSV format with the specific required column structure"""
    successful_results = [r for r in results if r.status == 'parsed']
    
    # Define the required column headers exactly as specified
    required_columns = [
        'Document_ID', 'Timestamp', 'Patient_ID', 'Patient_Created', 'Order_Pushed',
        'Patient_First_Name', 'Patient_Last_Name', 'Patient_DOB', 'Patient_Sex', 
        'Medical_Record_No', 'Service_Line', 'Payer_Source', 'Physician_NPI', 
        'Agency_Name', 'Patient_Address', 'Patient_City', 'Patient_State', 
        'Patient_Zip', 'Patient_Phone', 'Patient_Email', 'Order_Number', 
        'Order_Date', 'Start_Of_Care', 'Episode_Start_Date', 'Episode_End_Date', 
        'Sent_To_Physician_Date', 'Signed_By_Physician_Date', 'Company_ID', 
        'PG_Company_ID', 'SOC_Episode', 'Start_Episode', 'End_Episode', 
        'Diagnosis_1', 'Diagnosis_2', 'Diagnosis_3', 'Diagnosis_4', 
        'Diagnosis_5', 'Diagnosis_6', 'API_Status', 'Error_Message', 'Remarks'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=required_columns)
        writer.writeheader()
        
        seen_doc_ids = set()

        for result in successful_results:
            # Skip duplicates (keep first occurrence)
            if result.doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(result.doc_id)

            patient = result.patient_data
            order = result.order_data
            
            # Helper to extract ICD code from any text
            def _extract_icd(text_val: str) -> str:
                if not text_val:
                    return ""
                # Direct code if text matches pattern already
                text_val = text_val.strip()
                icd_match = re.search(r"[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?", text_val)
                return icd_match.group(0) if icd_match else ""

            # Extract up to 6 diagnoses from the episode_diagnoses list
            diagnoses = []
            
            # Add primary and secondary diagnoses first
            if order.primary_diagnosis:
                diagnoses.append(_extract_icd(order.primary_diagnosis))
            if order.secondary_diagnosis:
                diagnoses.append(_extract_icd(order.secondary_diagnosis))
            
            # Add episode diagnoses
            for diagnosis in order.episode_diagnoses:
                code_val = ""
                if diagnosis.diagnosis_code:
                    code_val = diagnosis.diagnosis_code
                elif diagnosis.diagnosis_description:
                    code_val = _extract_icd(diagnosis.diagnosis_description)
                if code_val:
                    diagnoses.append(code_val)
            
            # Ensure we have exactly 6 diagnosis fields
            while len(diagnoses) < 6:
                diagnoses.append("")
            diagnoses = diagnoses[:6]  # Take only first 6
            
            # Preserve leading zeros for digit-only MRNs when writing to CSV (Excel tends to strip them)
            def _preserve_leading_zeros(val: str) -> str:
                if val and val.isdigit() and val.startswith('0'):
                    return "'" + val  # Apostrophe forces text format in Excel
                return val or ""

            mrn_out = _preserve_leading_zeros(patient.medical_record_no)

            # Map the extracted data to the required columns
            row = {
                'Document_ID': result.doc_id,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Patient_ID': _preserve_leading_zeros(patient.account_number) if patient.account_number else '',
                'Patient_Created': '',  # Not available in current data
                'Order_Pushed': '',  # Not available in current data
                'Patient_First_Name': patient.patient_fname,
                'Patient_Last_Name': patient.patient_lname,
                'Patient_DOB': patient.dob,
                'Patient_Sex': patient.patient_sex,
                'Medical_Record_No': mrn_out,
                'Service_Line': order.service_type,
                'Payer_Source': patient.primary_insurance,
                'Physician_NPI': order.physician_npi or patient.provider_npi,
                'Agency_Name': order.ordering_facility,
                'Patient_Address': patient.address,
                'Patient_City': patient.city,
                'Patient_State': patient.state,
                'Patient_Zip': patient.zip_code,
                'Patient_Phone': patient.phone_number,
                'Patient_Email': patient.email,
                'Order_Number': order.order_no,
                'Order_Date': order.order_date,
                'Start_Of_Care': order.start_of_care,
                'Episode_Start_Date': order.episode_start_date,
                'Episode_End_Date': order.episode_end_date,
                'Sent_To_Physician_Date': '',  # Not available in current data
                'Signed_By_Physician_Date': order.signed_by_physician_date,
                'Company_ID': getattr(order, 'company_id', '') or '',
                'PG_Company_ID': os.getenv('PG_ID', 'd10f46ad-225d-4ba2-882c-149521fcead5'),
                'SOC_Episode': order.start_of_care,  # Using start_of_care as SOC_Episode
                'Start_Episode': order.episode_start_date,  # Same as Episode_Start_Date
                'End_Episode': order.episode_end_date,  # Same as Episode_End_Date
                'Diagnosis_1': diagnoses[0],
                'Diagnosis_2': diagnoses[1],
                'Diagnosis_3': diagnoses[2],
                'Diagnosis_4': diagnoses[3],
                'Diagnosis_5': diagnoses[4],
                'Diagnosis_6': diagnoses[5],
                'API_Status': 'SUCCESS' if result.status == 'parsed' else 'FAILED',
                'Error_Message': result.error,
                'Remarks': f"Confidence: {result.confidence_score:.3f}, Completeness: {result.completeness_score:.3f}, Method: {result.extraction_method}"
            }
            
            writer.writerow(row)
    
    # Also save failed results with error information
    failed_results = [r for r in results if r.status == 'failed']
    if failed_results:
        failed_path = output_path.replace('.csv', '_failed.csv')
        with open(failed_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=required_columns)
            writer.writeheader()
            
            for result in failed_results:
                # Create empty row with just the essential information
                row = {col: '' for col in required_columns}
                row.update({
                    'Document_ID': result.doc_id,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'API_Status': 'FAILED',
                    'Error_Message': result.error,
                    'Remarks': f"Processing failed: {result.extraction_method}"
                })
                writer.writerow(row)

class EnhancedDataParser:
    """Enhanced data parser with comprehensive field extraction and dual parsing strategy"""
    
    def __init__(self, output_dir: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "phi"):
        self.output_dir = output_dir
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM parser if needed
        try:
            from modules.llm_parser import LLMParser
            llm_parser = LLMParser(ollama_url, fast_mode=True)
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM parser: {e}")
            llm_parser = None
        
        # Initialize the main data parser with correct parameters
        config = {
            'ollama_url': ollama_url,
            'ollama_model': ollama_model,
            'output_dir': output_dir
        }
        
        self.data_parser = DataParser(
            field_extractor=FieldExtractor(), 
            llm_parser=llm_parser,
            config=config
        )
        
        self.logger.info(f"🚀 EnhancedDataParser initialized")
        self.logger.info(f"📁 Output directory: {output_dir}")
        self.logger.info(f"🤖 LLM: {ollama_model} at {ollama_url}")
    
    def parse_document(self, text: str, filename: str) -> ParsedResult:
        """Parse a single document with enhanced accuracy"""
        # Create extraction result format expected by DataParser
        extraction_result = {
            'doc_id': filename,
            'text': text,
            'status': 'extracted'
        }
        
        # Use the main data parser
        result = self.data_parser.parse_document(extraction_result)
        
        # Log result
        if result.status == 'parsed':
            self.logger.info(f"✅ Successfully parsed {filename} - Confidence: {result.confidence_score:.2f}, Completeness: {result.completeness_score:.2f}")
        else:
            self.logger.warning(f"❌ Failed to parse {filename}: {result.error}")
        
        return result
    
    def save_parsed_results(self, results: List[ParsedResult], output_path: str):
        """Save parsed results to CSV"""
        save_extraction_csv(results, output_path)
        self.logger.info(f"💾 Saved {len(results)} parsed results to {output_path}") 