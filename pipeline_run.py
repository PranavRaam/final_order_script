#!/usr/bin/env python3
"""
Pipeline Runner: Module 3 → Module 4 Integration
Connects text extraction (Module 3) with field extraction (Module 4) seamlessly.
No intermediate CSV files - direct memory handoff for perfect accuracy.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.light_text_extractor import TextExtractor
from modules.data_parser import DataParser, FieldExtractor, save_extraction_csv
from modules.llm_parser import LLMParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Seamless pipeline from Module 3 to Module 4"""
    
    def __init__(self, 
                 input_dir: str = "data",
                 output_dir: str = "output",
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "phi"):
        """
        Initialize the pipeline runner
        
        Args:
            input_dir: Directory containing PDF files to process
            output_dir: Directory to save results
            ollama_url: Ollama server URL for LLM fallback
            ollama_model: Ollama model name for LLM processing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.processing_dir = self.output_dir / f"processing_{timestamp}"
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Pipeline initialized")
        logger.info(f"📁 Input directory: {self.input_dir}")
        logger.info(f"📁 Output directory: {self.processing_dir}")
        logger.info(f"🤖 LLM: {ollama_model} at {ollama_url}")
    
    def load_documents(self) -> list:
        """
        Load PDF documents from input directory
        
        Returns:
            List of document dictionaries with doc_id and pdf_buffer
        """
        docs = []
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return docs
        
        logger.info(f"📚 Loading {len(pdf_files)} PDF documents")
        
        for pdf_path in pdf_files:
            try:
                with open(pdf_path, "rb") as f:
                    doc = {
                        "doc_id": pdf_path.stem,  # Remove .pdf extension
                        "pdf_buffer": f.read(),
                        "source_path": str(pdf_path),
                        "file_size": pdf_path.stat().st_size
                    }
                    docs.append(doc)
                    logger.debug(f"Loaded: {pdf_path.name} ({doc['file_size']} bytes)")
                    
            except Exception as e:
                logger.error(f"Failed to load {pdf_path}: {e}")
                continue
        
        logger.info(f"✅ Successfully loaded {len(docs)} documents")
        return docs
    
    def run_module3_text_extraction(self, documents: list) -> list:
        """
        Run Module 3: Text extraction from PDFs
        
        Args:
            documents: List of document dictionaries with pdf_buffer
            
        Returns:
            List of extraction results ready for Module 4
        """
        logger.info("🔄 Starting Module 3: Text Extraction")
        
        # Initialize text extractor
        text_extractor = TextExtractor(
            save_extracted_text=False,  # Keep in memory for efficiency
            output_dir=str(self.processing_dir / "module_3_text_extractor")
        )
        
        # Extract text from all documents
        extraction_results = text_extractor.extract_text_batch_for_module4(documents)
        
        # Log summary
        successful = len([r for r in extraction_results if r['status'] == 'extracted'])
        failed = len([r for r in extraction_results if r['status'] == 'failed'])
        
        logger.info(f"✅ Module 3 complete: {successful} successful, {failed} failed")
        
        return extraction_results
    
    def run_module4_field_extraction(self, extraction_results: list) -> list:
        """
        Run Module 4: Field extraction and parsing
        
        Args:
            extraction_results: List of text extraction results from Module 3
            
        Returns:
            List of parsed results with extracted fields
        """
        logger.info("🔄 Starting Module 4: Field Extraction")
        
        # Initialize field extractor and LLM parser
        field_extractor = FieldExtractor()
        
        try:
            llm_parser = LLMParser(ollama_url=self.ollama_url, model_name=self.ollama_model, fast_mode=True)
            logger.info(f"🤖 LLM parser initialized with {self.ollama_model}")
        except Exception as e:
            logger.warning(f"LLM parser not available: {e}")
            llm_parser = None
        
        # Initialize data parser
        data_parser = DataParser(
            field_extractor=field_extractor,
            llm_parser=llm_parser,
            config={
                'ollama_url': self.ollama_url,
                'ollama_model': self.ollama_model,
                'output_dir': str(self.processing_dir),
                'ALWAYS_RUN_LLM': True
            }
        )
        
        # Parse all documents
        parsed_results = data_parser.process_documents(extraction_results)
        
        # Log summary
        successful = len([r for r in parsed_results if r.status == 'parsed'])
        failed = len([r for r in parsed_results if r.status == 'failed'])
        
        logger.info(f"✅ Module 4 complete: {successful} successful, {failed} failed")
        
        return parsed_results
    
    def save_results(self, parsed_results: list) -> str:
        """
        Save parsed results to CSV
        
        Args:
            parsed_results: List of parsed results from Module 4
            
        Returns:
            Path to the saved CSV file
        """
        logger.info("💾 Saving results to CSV")
        
        output_csv = self.processing_dir / "extracted_data.csv"
        save_extraction_csv(parsed_results, str(output_csv))
        
        logger.info(f"✅ Results saved to: {output_csv}")
        return str(output_csv)
    
    def run_pipeline(self) -> str:
        """
        Run the complete pipeline: Module 3 → Module 4
        
        Returns:
            Path to the final CSV file
        """
        logger.info("🚀 Starting complete pipeline")
        
        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            raise ValueError("No documents to process")
        
        # Step 2: Module 3 - Text extraction
        extraction_results = self.run_module3_text_extraction(documents)
        
        # Step 3: Module 4 - Field extraction
        parsed_results = self.run_module4_field_extraction(extraction_results)
        
        # Step 4: Save results
        output_csv = self.save_results(parsed_results)
        
        # Final summary
        successful = len([r for r in parsed_results if r.status == 'parsed'])
        total = len(parsed_results)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        logger.info("🎉 Pipeline complete!")
        logger.info(f"📊 Final results: {successful}/{total} successful ({success_rate:.1f}%)")
        logger.info(f"📁 Output: {output_csv}")
        
        return output_csv

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Module 3 → Module 4 pipeline")
    parser.add_argument("--input-dir", default="data", help="Input directory with PDF files")
    parser.add_argument("--output-dir", default="output", help="Output directory for results")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--ollama-model", default="phi", help="Ollama model name")
    
    args = parser.parse_args()
    
    try:
        # Run pipeline
        pipeline = PipelineRunner(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model
        )
        
        output_csv = pipeline.run_pipeline()
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {output_csv}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 