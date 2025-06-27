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
from datetime import datetime
from typing import List, Dict, Any
import time
import json

# Module imports
from modules.input_reader import InputReader, save_validation_report, save_invalid_rows
from modules.document_fetcher import DocumentFetcher, save_failed_fetches, save_fetch_summary  
from modules.sharepoint_fetcher import SharePointFetcher
from modules.reporting import save_comprehensive_summary, generate_field_analysis_report, write_completeness_report
from modules.data_structures import ParsedResult, PatientData, OrderData

# Import the new streamlined pipeline
from pipeline_run import PipelineRunner

# Configuration Constants
INPUT_DIR = "data/input.csv"  # Default input CSV file
OUTPUT_DIR = "output"  # Default output directory
MAX_ROWS = 100  # Process all rows, set to number for testing
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi"  # Default to phi model

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
    
    # Setup directories
    directories = setup_directories(config['output_dir'])
    output_dir = directories['output_dir']
    
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
        
        # Step 2: Module 4 - Field extraction
        logger.info("🔄 Running Module 4: Field extraction...")
        parsed_results = pipeline.run_module4_field_extraction(extraction_results)
        
        # Step 3: Save results
        logger.info("💾 Saving results...")
        output_csv = pipeline.save_results(parsed_results)
        
        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {output_csv}")

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
    logger.info(f"   - Output CSV: {output_csv}")
    logger.info(f"   - Summary report: {summary_path}")
    logger.info(f"   - Analysis report: {analysis_path}")

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

def save_failed_fetches(failed_fetches: List[Dict], output_path: str):
    """Save failed fetch results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(failed_fetches, f, indent=2)

if __name__ == "__main__":
    main() 