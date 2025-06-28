#!/usr/bin/env python3
"""
Healthcare Document Processing Pipeline - Main Entry Point
Integrated workflow that processes documents from SharePoint through text extraction to field parsing

Workflow:
1. Fetch documents from SharePoint (or load from local files)
2. Use streamlined pipeline: Module 3 (Text Extraction) -> Module 4 (Field Parsing)
3. Generate comprehensive reports
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import json
import re
import pandas as pd

# Module imports
from modules.input_reader import InputReader, save_validation_report, save_invalid_rows
from modules.document_fetcher import DocumentFetcher, save_failed_fetches, save_fetch_summary  
from modules.sharepoint_fetcher import SharePointFetcher
from modules.reporting import save_comprehensive_summary, generate_field_analysis_report, write_completeness_report
from modules.data_structures import ParsedResult, PatientData, OrderData
from modules.data_structures import EpisodeDiagnosis
from modules.data_parser import save_extraction_csv
from modules.validator import validate_results
from modules.duplicate_detection import DuplicateDetector
from modules.patient_pusher import PatientPusher
from modules.order_pusher import OrderPusher

# Import the new streamlined pipeline
from pipeline_run import PipelineRunner

# Configuration Constants
INPUT_DIR = "data/input.csv"  
OUTPUT_DIR = "output" 
MAX_ROWS = 100  
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi"  

# Physician Group / Company settings – update these three values once before running
PG_ID = "d10f46ad-225d-4ba2-882c-149521fcead5"  
PG_NAME = "Prima Care"
PG_NPI = "1265422596"

# Toggle this flag to enable/disable patient & order API pushes.
# Keep False during testing/dry-run; set to True for production pushes.
ENABLE_API_PUSH = True

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_output_directories():
    """Create output directories for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main output directory  
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Module-specific directories
    directories = {
        'module1_dir': os.path.join(output_dir, 'module1_csv_validation'),
        'module2_dir': os.path.join(output_dir, 'module2_pdf_downloads'),
        'module3_dir': os.path.join(output_dir, 'module3_text_extraction'), 
        'module4_dir': os.path.join(output_dir, 'module4_data_parsing'),
        'reports_dir': os.path.join(output_dir, 'reports')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_dir, directories

def main():
    """Main function to run the document processing pipeline with streamlined Module 3->4 integration"""
    start_time = time.time()
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    # Load company mapping
    try:
        with open('company.json', 'r', encoding='utf-8') as f_cmp:
            company_map = json.load(f_cmp)
            logger.info(f"🏢 Loaded {len(company_map)} agency→companyId mappings")
    except Exception as e:
        company_map = {}
        logger.warning(f"Could not load company.json mapping: {e}")
    
    # --------------------------------------------------
    # Load input CSV (Module 1 data) into a quick lookup
    # so we can back-fill missing patient/order fields later.
    # We use only basic pandas-free helper from InputReader.
    # --------------------------------------------------
    try:
        df_csv = pd.read_csv(config['input_file'])
        _csv_map = {str(row['ID']): row for _, row in df_csv.iterrows() if not pd.isna(row['ID'])}
        logger.info(f"📑 Loaded {len(_csv_map)} records from input CSV for fallback use")
    except Exception as e:
        _csv_map = {}
        logger.warning(f"Failed to load input CSV for fallback merging: {e}")
    
    # Setup directories
    directories = setup_directories(config['output_dir'])
    output_dir = directories['output_dir']
    output_csv_path = os.path.join(directories['module4_dir'], 'extracted_data.csv')
    
    logger.info("🚀 Starting Healthcare Document Processing Pipeline")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📊 Document limit: {config['max_rows']} documents")
    
    # ==========================================
    # MODULE 1: Fetch documents from SharePoint
    # ==========================================
    logger.info("📥 MODULE 1: Fetching documents from SharePoint...")
    
    # Initialize Module 1 components
    sharepoint_fetcher = SharePointFetcher(
        site_url=config['sharepoint_site_url'],
        folder_name=config['sharepoint_folder'],
        username=config['sharepoint_username'],
        password=config['sharepoint_password']
    )
    
    # Fetch documents with the configured limit
    fetch_limit = min(config['max_rows'], config.get('document_limit', 100))
    logger.info(f"📥 Fetching up to {fetch_limit} documents from SharePoint...")
    
    fetch_results = sharepoint_fetcher.fetch_documents(limit=fetch_limit)
    
    if not fetch_results:
        logger.error("❌ No documents were fetched. Exiting.")
        return
    
    # Apply MAX_ROWS limit to fetched results
    if len(fetch_results) > config['max_rows']:
        logger.info(f"📊 Limiting results to {config['max_rows']} documents (MAX_ROWS setting)")
        fetch_results = fetch_results[:config['max_rows']]
    
    # Filter successful fetches
    successful_fetches = [result for result in fetch_results if result.get('status') == 'success']
    failed_fetches = [result for result in fetch_results if result.get('status') != 'success']
    
    total_docs = len(fetch_results)
    logger.info(f"✅ MODULE 1 completed: {len(successful_fetches)}/{total_docs} documents fetched successfully")
    
    if failed_fetches:
        logger.warning(f"⚠️  {len(failed_fetches)} documents failed to fetch")
        # Save failed fetches for review
        failed_fetch_path = os.path.join(directories['module1_dir'], 'failed_fetches.json')
        save_failed_fetches(failed_fetches, failed_fetch_path)
    
    # Save a complete summary (both success & fail) for auditing
    fetch_summary_path = os.path.join(directories['module1_dir'], 'sharepoint_fetch_results.json')
    save_sharepoint_fetch_results(fetch_results, fetch_summary_path)

    if not successful_fetches:
        logger.error("❌ No documents were successfully fetched. Exiting.")
        return
        
    # ==========================================
    # STREAMLINED MODULE 3 -> MODULE 4 PIPELINE
    # ==========================================
    logger.info("🔄 STREAMLINED PIPELINE: Module 3 → Module 4 integration...")
    
    # Initialize the streamlined pipeline
    pipeline = PipelineRunner(
        input_dir="data",  # Not used when we pass documents directly
        output_dir=output_dir,
        ollama_url=config['ollama_url'],
        ollama_model=config['ollama_model']
    )
    
    # Convert SharePoint fetch results to the format expected by the pipeline
    # The pipeline expects documents with 'pdf_buffer' or 'content'
    documents_for_pipeline = []
    
    for fetch_result in successful_fetches:
        doc_id = fetch_result.get('doc_id', 'unknown')
        pdf_buffer = fetch_result.get('pdf_buffer')
        content = fetch_result.get('content')  # If text was pre-extracted
        
        if pdf_buffer:
            # We have PDF data, use it directly
            doc = {
                "doc_id": doc_id,
                "pdf_buffer": pdf_buffer,
                "source_path": f"sharepoint://{doc_id}",
                "file_size": len(pdf_buffer)
            }
        elif content:
            # We have pre-extracted text
            doc = {
                "doc_id": doc_id,
                "content": content,
                "source_path": f"sharepoint://{doc_id}",
                "text_length": len(content)
            }
        else:
            logger.warning(f"Skipping {doc_id}: No PDF buffer or content available")
            continue
            
        documents_for_pipeline.append(doc)
    
    logger.info(f"📄 Prepared {len(documents_for_pipeline)} documents for pipeline processing")
    
    # Run the streamlined pipeline
    try:
        # Step 1: Module 3 - Text extraction
        logger.info("🔄 Running Module 3: Text extraction...")
        extraction_results = pipeline.run_module3_text_extraction(documents_for_pipeline)
        
        # Persist Module 3 outputs for auditing
        extraction_json_path = os.path.join(directories['module3_dir'], 'extraction_results.json')
        save_extraction_results(extraction_results, extraction_json_path)

        # Also drop individual text files for quick inspection
        text_files_dir = os.path.join(directories['module3_dir'], 'text_files')
        save_individual_text_files(extraction_results, text_files_dir)
        
        # Step 2: Module 4 - Field extraction
        logger.info("🔄 Running Module 4: Field extraction...")
        parsed_results = pipeline.run_module4_field_extraction(extraction_results)
        
        # Step 3: Save results
        logger.info("💾 Saving results...")
        
        # FINAL: Only map order_date to episode_start_date if episode_start_date is empty
        for res in parsed_results:
            od = res.order_data
            # Lookup corresponding CSV row (if available) using doc_id mapping. Normalise to strip any extraction suffixes like _digital/_ocr
            csv_row = None
            if _csv_map:
                raw_doc_id = str(res.doc_id)
                csv_row = _csv_map.get(raw_doc_id)
                if csv_row is None:
                    doc_id_clean = re.sub(r'_(?:digital|ocr).*$', '', raw_doc_id, flags=re.IGNORECASE)
                    csv_row = _csv_map.get(doc_id_clean)
            if not od.episode_start_date and od.order_date:
                od.episode_start_date = od.order_date
            # Always set episode_end_date = episode_start_date + 59 days if episode_start_date is present and valid
            from datetime import datetime, timedelta
            try:
                if od.episode_start_date:
                    dt = datetime.strptime(od.episode_start_date, "%m/%d/%Y")
                    od.episode_end_date = (dt + timedelta(days=59)).strftime("%m/%d/%Y")
            except Exception:
                od.episode_end_date = ""

            # 3. Order Number assignment
            # Always prefer the numeric ID from the CSV row when available to keep a consistent 7-digit scheme
            if csv_row is not None and csv_row.get('ID'):
                od.order_no = re.sub(r'[^0-9]', '', str(csv_row.get('ID')).strip())
            else:
                # If CSV not available, fabricate order number (digits-only) from the doc_id
                if not od.order_no or not str(od.order_no).strip():
                    base_order_no = re.sub(r'_(?:digital|ocr).*$', '', str(res.doc_id), flags=re.IGNORECASE)
                    digits_only = re.sub(r'[^0-9]', '', base_order_no)
                    od.order_no = digits_only if digits_only else base_order_no

            # 4. Fallback for order_date
            if not od.order_date:
                if od.signed_by_physician_date:
                    od.order_date = od.signed_by_physician_date
                elif od.start_of_care:
                    od.order_date = od.start_of_care
                elif csv_row is not None:
                    rec_date = _normalize_date(csv_row.get('Received On'))
                    if rec_date:
                        od.order_date = rec_date
            else:
                od.order_date = _normalize_date(od.order_date)  # normalise any format

            # 5. Fallback for primary_diagnosis
            if not od.primary_diagnosis and len(od.episode_diagnoses) > 0:
                first_diag = od.episode_diagnoses[0]
                diag_desc = getattr(first_diag, 'diagnosis_description', '') or getattr(first_diag, 'diagnosis_code', '')
                od.primary_diagnosis = diag_desc

            # 5b. Ensure episode_diagnoses list is not empty (API requires at least one)
            if (not od.episode_diagnoses or len(od.episode_diagnoses) == 0) and od.primary_diagnosis:
                od.episode_diagnoses = [
                    EpisodeDiagnosis(
                        diagnosis_code="",
                        diagnosis_description=od.primary_diagnosis,
                        diagnosis_type="primary",
                        icd_version="ICD-10",
                    )
                ]

            # 6. --- CSV-based patient and order backfill ---
            if csv_row is not None:
                pat = res.patient_data
                # Patient name
                if not pat.patient_fname or not pat.patient_lname:
                    name_dict = _split_name(csv_row.get('Patient', ''))
                    if not pat.patient_fname:
                        pat.patient_fname = name_dict['fname']
                    if not pat.patient_lname:
                        pat.patient_lname = name_dict['lname']

                # DOB not in CSV; skip.

                # MRN (if CSV has it; assume column 'MRN' or 'ID')
                if not pat.medical_record_no:
                    mrn_val = csv_row.get('MRN') or csv_row.get('ID')
                    if mrn_val:
                        pat.medical_record_no = str(mrn_val)

                # Sex not in CSV; skip unless column exists
                if not pat.patient_sex and csv_row.get('Sex'):
                    pat.patient_sex = str(csv_row['Sex']).upper()

                # OrderDate fallback from 'Received On' column (assuming that is acceptable)
                if not od.order_date:
                    rec_date = _normalize_date(csv_row.get('Received On'))
                    if rec_date:
                        od.order_date = rec_date

                # Agency / Facility name (case-insensitive search)
                agency_name = None
                for key in csv_row.keys():
                    k_lower = str(key).lower().strip()
                    if k_lower in {
                        'facility name', 'facility_name', 'facility',
                        'agency name', 'agency_name'
                    }:
                        val = csv_row.get(key)
                        if val and str(val).strip():
                            agency_name = str(val).strip()
                            break

                if agency_name:
                    od.provider_name = agency_name
                    od.ordering_facility = agency_name

            # Fill SOC from SOE if SOC is empty but SOE is present
            if not od.start_of_care and od.episode_start_date:
                od.start_of_care = od.episode_start_date
        # --------------------------------------------------

        # >>> ADD: Module 5 – Validation
        logger.info("🔍 MODULE 5: Validating parsed results…")
        valid_results, invalid_details = validate_results(parsed_results)
        logger.info(
            f"✅ Validation complete: {len(valid_results)} valid, {len(invalid_details)} invalid")

        # Persist invalid rows for audit
        invalid_json_path = os.path.join(directories['module4_dir'], 'invalid_results.json')
        with open(invalid_json_path, 'w', encoding='utf-8') as f:
            json.dump(invalid_details, f, indent=2)
        logger.info(f"📁 Invalid result details saved to: {invalid_json_path}")

        # Save valid results for further offline testing (payload tests, etc.)
        valid_json_path = os.path.join(directories['module4_dir'], 'valid_results.json')
        with open(valid_json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in valid_results], f, indent=2)
        logger.info(f"📁 Valid parsed results saved to: {valid_json_path}")

        # Save *all* parsed results (including those that failed validation) for full auditing/testing
        all_json_path = os.path.join(directories['module4_dir'], 'all_results.json')
        with open(all_json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in parsed_results], f, indent=2)
        logger.info(f"📁 All parsed results saved to: {all_json_path}")

        # >>> ADD: Module 6 – Duplicate Detection (run on VALID results)
        logger.info("🔄 MODULE 6: Detecting duplicate patients…")
        dup_detector = DuplicateDetector()
        duplicates = dup_detector.detect_duplicates(valid_results)
        logger.info(f"🔎 Duplicate groups found: {len(duplicates)}")

        duplicate_report_path = os.path.join(directories['reports_dir'], 'duplicate_patients_report.txt')
        duplicate_report_text = dup_detector.generate_duplicate_report(duplicates)
        with open(duplicate_report_path, 'w', encoding='utf-8') as f:
            f.write(duplicate_report_text)
        logger.info(f"📁 Duplicate patient report saved to: {duplicate_report_path}")

        # >>> ADD: Module 7 & 8 – Patient and Order Push (optional)
        api_tracking_records = []
        if ENABLE_API_PUSH:
            logger.info("🚀 MODULE 7 & 8: Pushing patients and orders to API… (ENABLE_API_PUSH=True)")
            patient_pusher = PatientPusher()
            order_pusher = OrderPusher()

            # Use the global constants defined at the top (can be overridden via env vars)
            global PG_ID, PG_NAME, PG_NPI

            for res in valid_results:
                patient = res.patient_data
                order = res.order_data

                # Build extra fields for patient creation
                # Attempt to resolve companyId using agency/provider/facility name
                agency_src = (order.ordering_facility or order.provider_name or '').strip()
                company_id_val = None
                if agency_src and company_map:
                    # Exact match first
                    company_id_val = company_map.get(agency_src)
                    if not company_id_val:
                        # Try case-insensitive match
                        for k, v in company_map.items():
                            if k.lower() == agency_src.lower():
                                company_id_val = v
                                break
                extra_patient_fields = {
                    'pgCompanyId': PG_ID,
                    'companyId': company_id_val or '',
                    'physicianGroup': PG_NAME,
                    'physicianGroupNPI': PG_NPI,
                }
                extra_patient_fields = {k: v for k, v in extra_patient_fields.items() if v}

                if not ENABLE_API_PUSH:
                    # Dry-run – record that we skipped
                    api_tracking_records.append({
                        'doc_id': res.doc_id,
                        'patient_status': 'SKIPPED',
                        'order_status': 'SKIPPED',
                        'remarks': 'Dry-run: API push disabled'
                    })
                    continue

                # ---- Actual push section ----
                success_pat, pat_resp, pat_status = patient_pusher.push_patient(patient, **extra_patient_fields)
                if not success_pat:
                    api_tracking_records.append({
                        'doc_id': res.doc_id,
                        'patient_status': pat_status,
                        'order_status': '',
                        'remarks': 'Patient push failed'
                    })
                    continue

                patient_id = pat_resp.get('id') or pat_resp.get('patientId') or pat_resp.get('patient_id')
                if not patient_id:
                    api_tracking_records.append({
                        'doc_id': res.doc_id,
                        'patient_status': pat_status,
                        'order_status': '',
                        'remarks': 'No patient id returned'
                    })
                    continue

                # Build order extra fields
                extra_order_fields = {
                    'pgCompanyId': PG_ID,
                    'companyId': company_id_val or ''
                }
                extra_order_fields = {k: v for k, v in extra_order_fields.items() if v}

                success_ord, ord_resp, ord_status = order_pusher.push_order(order, patient_id, **extra_order_fields)

                api_tracking_records.append({
                    'doc_id': res.doc_id,
                    'patient_status': pat_status,
                    'order_status': ord_status,
                    'remarks': 'success' if success_ord else 'order push failed'
                })

        # Save API tracking CSV
        api_tracking_path = os.path.join(directories['module4_dir'], 'api_tracking.csv')
        if api_tracking_records:
            pd.DataFrame(api_tracking_records).to_csv(api_tracking_path, index=False)
            logger.info(f"📁 API tracking CSV saved to: {api_tracking_path}")

        # Save to the proper module4 directory that main.py created (moved here to ensure updated values)
        save_extraction_csv(parsed_results, output_csv_path)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return
    
    # ==========================================
    # GENERATE REPORTS
    # ==========================================
    logger.info("📊 Generating comprehensive reports...")
    
    # Generate comprehensive summary
    summary_path = os.path.join(directories['reports_dir'], 'comprehensive_summary.txt')
    save_comprehensive_summary(parsed_results, summary_path)
    
    # Generate field analysis report
    analysis_path = os.path.join(directories['reports_dir'], 'field_analysis_report.txt')
    generate_field_analysis_report(parsed_results, analysis_path)

    # Generate completeness report
    completeness_csv = os.path.join(directories['reports_dir'], 'completeness_report.csv')
    write_completeness_report(parsed_results, completeness_csv)
    logger.info(f"📝 Completeness report saved to: {completeness_csv}")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    end_time = time.time()
    processing_time = end_time - start_time
    
    successful_parses = len([r for r in parsed_results if r.status == 'parsed'])
    total_parses = len(parsed_results)
    success_rate = (successful_parses / total_parses) * 100 if total_parses > 0 else 0
    
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   - Documents fetched: {len(successful_fetches)}/{total_docs}")
    logger.info(f"   - Documents parsed: {successful_parses}/{total_parses}")
    logger.info(f"   - Success rate: {success_rate:.1f}%")
    logger.info(f"   - Processing time: {processing_time:.2f} seconds")
    logger.info(f"")
    logger.info(f"📁 Output Files Generated:")
    logger.info(f"   📊 Main Results:")
    logger.info(f"      - Extracted Data CSV: {output_csv_path}")
    logger.info(f"   📋 Module Results:")
    logger.info(f"      - All Parsed Results: {all_json_path}")
    logger.info(f"      - Valid Results: {valid_json_path}")
    logger.info(f"      - Invalid Results: {invalid_json_path}")
    if failed_fetches:
        failed_fetch_path = os.path.join(directories['module1_dir'], 'failed_fetches.json')
        logger.info(f"      - Failed Fetches: {failed_fetch_path}")
    logger.info(f"   📈 Reports:")
    logger.info(f"      - Comprehensive Summary: {summary_path}")
    logger.info(f"      - Field Analysis Report: {analysis_path}")
    logger.info(f"      - Completeness Report: {completeness_csv}")
    logger.info(f"      - Duplicate Patient Report: {duplicate_report_path}")
    logger.info(f"")
    logger.info(f"🗂️  All files saved in: {output_dir}")

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables or defaults"""
    return {
        'input_file': os.getenv('INPUT_FILE', INPUT_DIR),
        'output_dir': os.getenv('OUTPUT_DIR', OUTPUT_DIR),
        'max_rows': int(os.getenv('MAX_ROWS', MAX_ROWS)),
        'ollama_url': os.getenv('OLLAMA_URL', OLLAMA_URL),
        'ollama_model': os.getenv('OLLAMA_MODEL', OLLAMA_MODEL),
        'sharepoint_site_url': os.getenv('SHAREPOINT_SITE_URL', ''),
        'sharepoint_folder': os.getenv('SHAREPOINT_FOLDER', ''),
        'sharepoint_username': os.getenv('SHAREPOINT_USERNAME', ''),
        'sharepoint_password': os.getenv('SHAREPOINT_PASSWORD', ''),
        'document_limit': int(os.getenv('DOCUMENT_LIMIT', '100'))
    }

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def setup_directories(output_base_dir: str) -> Dict[str, str]:
    """Setup output directories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"processing_{timestamp}")
    
    directories = {
        'output_dir': output_dir,
        'module1_dir': os.path.join(output_dir, 'module1_csv_validation'),
        'module2_dir': os.path.join(output_dir, 'module2_pdf_downloads'),
        'module3_dir': os.path.join(output_dir, 'module3_text_extraction'), 
        'module4_dir': os.path.join(output_dir, 'module4_data_parsing'),
        'reports_dir': os.path.join(output_dir, 'reports')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def save_extraction_results(extraction_results: List[Dict], output_path: str):
    """Save text extraction results to JSON file for reference"""
    try:
        # Convert extraction results to a JSON-serializable format
        serializable_results = []
        for result in extraction_results:
            serializable_result = {
                'doc_id': result.get('doc_id', 'unknown'),
                'status': result.get('status', 'unknown'),
                'text_length': len(result.get('text', '')) if result.get('text') else 0,
                'processing_time': result.get('processing_time', 0),
                'source_path': result.get('source_path', ''),
                'extracted_text': result.get('text', ''),  # Include the actual text content
                'error_message': result.get('error_message', '') if result.get('status') == 'failed' else None
            }
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logging.getLogger(__name__).info(f"📁 Text extraction results saved to: {output_path}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save extraction results: {e}")

def save_sharepoint_fetch_results(fetch_results: List[Dict], output_path: str):
    """Save SharePoint fetch results to JSON file for reference"""
    try:
        # Convert fetch results to a JSON-serializable format
        serializable_results = []
        for result in fetch_results:
            serializable_result = {
                'doc_id': result.get('doc_id', 'unknown'),
                'status': result.get('status', 'unknown'),
                'file_size': result.get('file_size', 0) if 'pdf_buffer' in result else None,
                'content_length': len(result.get('content', '')) if result.get('content') else None,
                'fetch_time': result.get('fetch_time', ''),
                'source_url': result.get('source_url', ''),
                'error_message': result.get('error_message', '') if result.get('status') != 'success' else None
            }
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logging.getLogger(__name__).info(f"📁 SharePoint fetch results saved to: {output_path}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save SharePoint fetch results: {e}")

def save_failed_fetches(failed_fetches: List[Dict], output_path: str):
    """Save failed fetch results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(failed_fetches, f, indent=2)

def save_individual_text_files(extraction_results: List[Dict], output_dir: str):
    """Save individual text files for easier review"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        for result in extraction_results:
            doc_id = result.get('doc_id', 'unknown')
            text = result.get('text', '')
            if text:
                filename = f"{doc_id}.txt"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
        logging.getLogger(__name__).info(f"📁 Individual text files saved to: {output_dir}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save individual text files: {e}")

# --------------------------------------------------
# Utility helpers for fallback merging
# --------------------------------------------------


def _normalize_date(date_str: str) -> str:
    """Return date in MM/DD/YYYY or empty string if parse fails."""
    if not date_str:
        return ""
    date_str = str(date_str).strip()
    for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%y"]:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%m/%d/%Y")
        except ValueError:
            continue
    # Try formats like 21-May-2025
    try:
        dt = datetime.strptime(date_str, "%d-%b-%Y")
        return dt.strftime("%m/%d/%Y")
    except ValueError:
        return ""


_NAME_SPLIT_RE = re.compile(r"\s*,\s*")


def _split_name(csv_name: str) -> Dict[str, str]:
    """Split 'Last, First Middle' into dict with fname/lname."""
    if not csv_name:
        return {"fname": "", "lname": ""}
    parts = _NAME_SPLIT_RE.split(csv_name.strip(), maxsplit=1)
    if len(parts) == 2:
        lname, rest = parts
        fname = rest.split()[0] if rest else ""
    else:
        split = csv_name.split()
        fname = split[0]
        lname = split[-1] if len(split) > 1 else ""
    return {"fname": fname.title(), "lname": lname.title()}

if __name__ == "__main__":
    main() 