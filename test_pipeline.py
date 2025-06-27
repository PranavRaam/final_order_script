#!/usr/bin/env python3
"""
Test script for the Module 3 → Module 4 pipeline
Tests the pipeline with existing data to ensure it works correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline_run import PipelineRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the pipeline with existing data"""
    
    # Check if we have any PDF files to test with
    data_dir = Path("data")
    if not data_dir.exists():
        logger.warning("No 'data' directory found. Creating test data...")
        data_dir.mkdir(exist_ok=True)
        
        # Create a simple test PDF or copy from existing output
        logger.info("Please add some PDF files to the 'data' directory and run again.")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in 'data' directory")
        logger.info("Please add some PDF files to test with")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files to test with")
    
    try:
        # Run the pipeline
        pipeline = PipelineRunner(
            input_dir="data",
            output_dir="output",
            ollama_url="http://localhost:11434",
            ollama_model="phi"
        )
        
        output_csv = pipeline.run_pipeline()
        
        # Verify the output
        if Path(output_csv).exists():
            logger.info(f"✅ Test successful! Output file created: {output_csv}")
            
            # Read and display a sample of the results
            import pandas as pd
            df = pd.read_csv(output_csv)
            logger.info(f"📊 Generated {len(df)} rows with {len(df.columns)} columns")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            # Show first few rows
            logger.info("📄 Sample data:")
            print(df.head(3).to_string())
            
            return True
        else:
            logger.error("❌ Test failed: Output file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def test_with_mock_data():
    """Test the pipeline with mock data (no PDF files needed)"""
    logger.info("🧪 Testing with mock data...")
    
    # Create mock documents
    mock_documents = [
        {
            "doc_id": "test_001",
            "content": """
            Patient: Smith, John A
            Date of Birth: 01/15/1950
            Gender: MALE
            Medical Record No: MA123456789012
            Phone: (555) 123-4567
            Address: 123 Main St, Boston, MA 02101
            
            Order Number: 1234567890
            Order Date: 05/21/2025
            Start of Care: 05/22/2025
            Episode Start: 05/22/2025
            Episode End: 07/21/2025
            
            Diagnosis: I10 Essential hypertension
            """,
            "source_path": "mock_data/test_001.txt"
        },
        {
            "doc_id": "test_002", 
            "content": """
            CLIENT: JONES, MARY B (MA987654321098)
            DOB: 03/20/1945
            Sex: F
            Phone: (555) 987-6543
            Address: 456 Oak Ave, Worcester, MA 01602
            
            Order #9876543210
            Order Date: 05/20/2025
            SOC Date: 05/21/2025
            Certification Period: 05/21/2025 to 07/20/2025
            
            Primary Diagnosis: E11.9 Type 2 diabetes mellitus without complications
            """,
            "source_path": "mock_data/test_002.txt"
        }
    ]
    
    try:
        # Initialize pipeline
        pipeline = PipelineRunner(
            input_dir="data",
            output_dir="output",
            ollama_url="http://localhost:11434",
            ollama_model="phi"
        )
        
        # Test Module 3 with mock data
        logger.info("🔄 Testing Module 3 with mock data...")
        from modules.text_extractor import TextExtractor
        
        text_extractor = TextExtractor(save_extracted_text=False)
        extraction_results = text_extractor.extract_text_batch_for_module4(mock_documents)
        
        logger.info(f"✅ Module 3 processed {len(extraction_results)} mock documents")
        
        # Test Module 4 with extraction results
        logger.info("🔄 Testing Module 4 with extraction results...")
        from modules.data_parser import DataParser, FieldExtractor
        from modules.llm_parser import LLMParser
        
        field_extractor = FieldExtractor()
        try:
            llm_parser = LLMParser("http://localhost:11434", fast_mode=True)
        except:
            llm_parser = None
            
        data_parser = DataParser(
            field_extractor=field_extractor,
            llm_parser=llm_parser
        )
        
        parsed_results = data_parser.parse_documents(extraction_results)
        
        logger.info(f"✅ Module 4 processed {len(parsed_results)} documents")
        
        # Save results
        output_csv = pipeline.processing_dir / "test_extracted_data.csv"
        from modules.data_parser import save_extraction_csv
        save_extraction_csv(parsed_results, str(output_csv))
        
        logger.info(f"✅ Mock test successful! Results saved to: {output_csv}")
        
        # Display results
        successful = len([r for r in parsed_results if r.status == 'parsed'])
        logger.info(f"📊 Mock test results: {successful}/{len(parsed_results)} successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting pipeline tests...")
    
    # Test 1: With real PDF files
    logger.info("=" * 50)
    logger.info("TEST 1: Real PDF files")
    logger.info("=" * 50)
    
    if test_pipeline():
        logger.info("✅ Test 1 PASSED")
    else:
        logger.info("❌ Test 1 FAILED")
    
    # Test 2: With mock data
    logger.info("=" * 50)
    logger.info("TEST 2: Mock data")
    logger.info("=" * 50)
    
    if test_with_mock_data():
        logger.info("✅ Test 2 PASSED")
    else:
        logger.info("❌ Test 2 FAILED")
    
    logger.info("=" * 50)
    logger.info("🧪 All tests completed!")

if __name__ == "__main__":
    main() 