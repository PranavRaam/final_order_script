"""
Mock SharePoint Fetcher Module
Simulates SharePoint document fetching for testing the integrated workflow
Uses existing local files to test the pipeline functionality
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

class SharePointFetcher:
    """Mock SharePoint fetcher that uses local files for testing"""
    
    def __init__(self, site_url: str = "", folder_name: str = "", 
                 username: str = "", password: str = ""):
        """Initialize the mock SharePoint fetcher"""
        self.site_url = site_url
        self.folder_name = folder_name
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
        
        # For testing, we'll use extracted text files as mock documents
        self.mock_data_dir = "logs/module_3_text_extractor"
        
        self.logger.info("🔧 Mock SharePoint Fetcher initialized")
        self.logger.info(f"   Using local files from: {self.mock_data_dir}")
    
    def fetch_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Mock fetch documents method that returns existing text files as document objects
        
        Args:
            limit: Maximum number of documents to fetch
            
        Returns:
            List of document dictionaries with mock document data
        """
        self.logger.info(f"🔍 Mock fetching documents (limit: {limit or 'unlimited'})")
        
        documents = []
        
        try:
            # Find existing text files to use as mock documents
            text_pattern = os.path.join(self.mock_data_dir, "extracted_texts_*/932*.txt")
            text_files = glob.glob(text_pattern)
            
            if not text_files:
                self.logger.warning(f"⚠️ No mock documents found in {self.mock_data_dir}")
                return []
            
            # Apply limit if specified
            if limit:
                text_files = text_files[:limit]
            
            self.logger.info(f"📄 Found {len(text_files)} mock documents")
            
            for i, text_file in enumerate(text_files, 1):
                try:
                    # Extract document ID from filename
                    filename = os.path.basename(text_file)
                    doc_id = filename.replace('.txt', '')
                    
                    # Read the text content
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Create mock document structure
                    document = {
                        'doc_id': doc_id,
                        'filename': filename,
                        'status': 'success',
                        'size': len(text_content),
                        'content_type': 'text/plain',
                        'url': f"mock://sharepoint/{filename}",
                        'local_path': text_file,
                        'pdf_buffer': None,  # No PDF buffer for text files
                        'content': text_content,  # Pre-extracted text for Module 3
                        'metadata': {
                            'source': 'mock_sharepoint',
                            'extraction_method': 'pre_extracted',
                            'file_type': 'text',
                            'original_pdf': f"{doc_id}.pdf"
                        }
                    }
                    
                    documents.append(document)
                    self.logger.debug(f"   📄 Mock document {i}: {doc_id} ({len(text_content)} chars)")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing mock document {text_file}: {str(e)}")
                    # Create failed document entry
                    failed_doc = {
                        'doc_id': os.path.basename(text_file).replace('.txt', ''),
                        'filename': os.path.basename(text_file),
                        'status': 'failed',
                        'error': f"Mock processing error: {str(e)}",
                        'metadata': {
                            'source': 'mock_sharepoint',
                            'error_type': 'processing_error'
                        }
                    }
                    documents.append(failed_doc)
            
            successful_docs = len([d for d in documents if d['status'] == 'success'])
            self.logger.info(f"✅ Mock fetch completed: {successful_docs}/{len(documents)} documents successful")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ Mock SharePoint fetch failed: {str(e)}")
            return []
    
    def test_connection(self) -> bool:
        """Test mock connection - always returns True for testing"""
        self.logger.info("🔗 Testing mock SharePoint connection...")
        
        # Check if mock data directory exists
        if os.path.exists(self.mock_data_dir):
            self.logger.info("✅ Mock connection successful")
            return True
        else:
            self.logger.error(f"❌ Mock data directory not found: {self.mock_data_dir}")
            return False
    
    def get_document_count(self) -> int:
        """Get total count of available mock documents"""
        try:
            text_pattern = os.path.join(self.mock_data_dir, "extracted_texts_*/932*.txt")
            text_files = glob.glob(text_pattern)
            count = len(text_files)
            self.logger.info(f"📊 Total mock documents available: {count}")
            return count
        except Exception as e:
            self.logger.error(f"❌ Error counting mock documents: {str(e)}")
            return 0
    
    def cleanup(self):
        """Cleanup method for consistency with real SharePoint fetcher"""
        self.logger.info("🧹 Mock SharePoint fetcher cleanup completed")


def create_mock_pdf_documents(count: int = 5) -> List[Dict[str, Any]]:
    """
    Create mock PDF document objects for testing when no text files are available
    
    Args:
        count: Number of mock documents to create
        
    Returns:
        List of mock document dictionaries
    """
    documents = []
    
    for i in range(1, count + 1):
        doc_id = f"mock_doc_{i:03d}"
        document = {
            'doc_id': doc_id,
            'filename': f"{doc_id}.pdf",
            'status': 'success',
            'size': 1024 * (i + 10),  # Mock file size
            'content_type': 'application/pdf',
            'url': f"mock://sharepoint/{doc_id}.pdf",
            'local_path': None,
            'pdf_buffer': b'%PDF-1.4 Mock PDF Content',  # Mock PDF buffer
            'metadata': {
                'source': 'mock_sharepoint',
                'created_date': f"2024-01-{i:02d}",
                'file_type': 'pdf'
            }
        }
        documents.append(document)
    
    return documents 