"""
Module 2: Document Fetcher
Fetches PDF documents from the Doctor Alliance API using doc_ids from Module 1.

Features:
- API integration with proper authentication
- Base64 decoding to binary PDF
- Retry logic for transient errors
- Comprehensive error handling and logging
- Optional disk saving for debugging
- Structured output for Module 3
"""

import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class DocumentFetcher:
    """
    Fetches PDF documents from Doctor Alliance API
    """
    
    def __init__(self, save_to_disk: bool = False, output_dir: str = "tmp/pdf_docs"):
        """
        Initialize DocumentFetcher
        
        Args:
            save_to_disk: Whether to save PDFs to disk for debugging
            output_dir: Directory to save PDFs if save_to_disk is True
        """
        self.auth_token = os.getenv('AUTH_TOKEN')
        self.doc_api_url = "https://api.doctoralliance.com/document/getfile"
        self.save_to_disk = save_to_disk
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Validate auth token
        if not self.auth_token:
            raise ValueError("AUTH_TOKEN not found in environment variables. Please check your .env file.")
        
        # Create output directory if saving to disk
        if self.save_to_disk:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"PDF output directory created: {self.output_dir}")
    
    def _extract_pdf_from_response(self, data: Dict, doc_id: str) -> Tuple[bool, Optional[bytes], str]:
        """
        Extract PDF data from API response - handles multiple response formats
        
        Args:
            data: JSON response data
            doc_id: Document ID for logging
            
        Returns:
            Tuple of (success, pdf_buffer, error_message)
        """
        # Log the response structure for debugging
        self.logger.debug(f"Response keys for doc_id {doc_id}: {list(data.keys())}")
        
        doc_b64 = None
        found_field = None
        
        # Handle the actual API response format: {"value": {"documentBuffer": "..."}, "isSuccess": true}
        if 'value' in data and isinstance(data['value'], dict):
            value_data = data['value']
            if 'documentBuffer' in value_data and value_data['documentBuffer']:
                doc_b64 = value_data['documentBuffer']
                found_field = 'value.documentBuffer'
                self.logger.debug(f"Found PDF data in {found_field} for doc_id {doc_id}")
        
        # If not found in the expected location, try other common field names
        if not doc_b64:
            # Try multiple possible field names for PDF data
            pdf_field_names = [
                'document',           # Expected field name
                'documentBuffer',     # Alternative (direct)
                'pdf_data',          # Alternative
                'file_data',         # Alternative
                'content',           # Alternative
                'data',              # Alternative
                'file',              # Alternative
                'pdf',               # Alternative
                'base64',            # Alternative
                'fileContent',       # Alternative
                'documentContent'    # Alternative
            ]
            
            # Search for PDF data in response
            for field_name in pdf_field_names:
                if field_name in data and data[field_name]:
                    doc_b64 = data[field_name]
                    found_field = field_name
                    break
        
        if not doc_b64:
            # Log available fields for debugging
            available_fields = {k: type(v).__name__ for k, v in data.items()}
            if 'value' in data and isinstance(data['value'], dict):
                value_fields = {k: type(v).__name__ for k, v in data['value'].items()}
                self.logger.error(f"No PDF data found for doc_id {doc_id}. Root fields: {available_fields}, Value fields: {value_fields}")
            else:
                self.logger.error(f"No PDF data found for doc_id {doc_id}. Available fields: {available_fields}")
            
            # Check if any field contains base64-like data
            all_fields = [(k, v) for k, v in data.items()]
            if 'value' in data and isinstance(data['value'], dict):
                all_fields.extend([(f"value.{k}", v) for k, v in data['value'].items()])
            
            for key, value in all_fields:
                if isinstance(value, str) and len(value) > 100:
                    # Check if it looks like base64
                    try:
                        # Try to decode a small portion
                        test_decode = base64.b64decode(value[:100])
                        if test_decode:
                            self.logger.warning(f"Potential base64 data found in field '{key}' for doc_id {doc_id}")
                            doc_b64 = value
                            found_field = key
                            break
                    except:
                        continue
        
        if not doc_b64:
            return False, None, f"No PDF data found in response. Available fields: {list(data.keys())}"
        
        # Try to decode Base64
        try:
            pdf_buffer = base64.b64decode(doc_b64)
            
            if len(pdf_buffer) == 0:
                return False, None, f"Empty PDF buffer after decoding from field '{found_field}'"
            
            # Verify it's a PDF by checking magic bytes
            if not pdf_buffer.startswith(b'%PDF'):
                self.logger.warning(f"Decoded data for doc_id {doc_id} doesn't start with PDF magic bytes")
                # Still return it - might be valid PDF with different format
            
            self.logger.debug(f"Successfully extracted PDF from field '{found_field}' for doc_id {doc_id}")
            return True, pdf_buffer, ""
            
        except Exception as e:
            return False, None, f"Base64 decode error from field '{found_field}': {str(e)}"
    
    def _make_request(self, doc_id: str, max_retries: int = 3) -> Tuple[bool, Optional[bytes], str, int]:
        """
        Make API request to fetch document
        
        Args:
            doc_id: Document ID to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (success, pdf_buffer, error_message, status_code)
        """
        headers = {
            'Authorization': f'Bearer {self.auth_token}',
            'Accept': 'application/json'
        }
        
        url = f"{self.doc_api_url}?docId.id={doc_id}"
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(f"Fetching doc_id {doc_id}, attempt {attempt + 1}")
                
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Check if the API returned an error response
                        if 'isSuccess' in data and not data['isSuccess']:
                            # Handle API-level errors
                            error_code = data.get('errorCode', 'UNKNOWN_ERROR')
                            error_message = data.get('errorMessage', 'No error message provided')
                            
                            # Log the specific error details
                            self.logger.warning(f"API error for doc_id {doc_id}: {error_code} - {error_message}")
                            
                            # Return appropriate error based on error code
                            if error_code in ['NOT_FOUND', 'DOCUMENT_NOT_FOUND']:
                                return False, None, f"Document not found: {error_message}", 404
                            elif error_code in ['ACCESS_DENIED', 'UNAUTHORIZED']:
                                return False, None, f"Access denied: {error_message}", 401
                            elif error_code in ['INVALID_REQUEST', 'BAD_REQUEST']:
                                return False, None, f"Invalid request: {error_message}", 400
                            else:
                                return False, None, f"API error ({error_code}): {error_message}", 200
                        
                        # If isSuccess is true or not present, try to extract PDF
                        success, pdf_buffer, error = self._extract_pdf_from_response(data, doc_id)
                        
                        if success:
                            return True, pdf_buffer, "", 200
                        else:
                            return False, None, error, 200
                        
                    except json.JSONDecodeError:
                        return False, None, "Invalid JSON response", 200
                    except Exception as e:
                        return False, None, f"Response processing error: {str(e)}", 200
                
                # Handle specific error codes
                elif response.status_code == 404:
                    return False, None, "Document not found", 404
                
                elif response.status_code == 401:
                    return False, None, "Unauthorized - check AUTH_TOKEN", 401
                
                elif response.status_code >= 500:
                    # Server errors - retry
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"Server error {response.status_code} for doc_id {doc_id}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, None, f"Server error: {response.status_code}", response.status_code
                
                else:
                    return False, None, f"HTTP error: {response.status_code}", response.status_code
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Timeout for doc_id {doc_id}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, None, "Request timeout", 0
                    
            except requests.exceptions.ConnectionError:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Connection error for doc_id {doc_id}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, None, "Connection error", 0
                    
            except Exception as e:
                return False, None, f"Unexpected error: {str(e)}", 0
        
        return False, None, "Max retries exceeded", 0
    
    def _save_pdf_to_disk(self, doc_id: str, pdf_buffer: bytes) -> Optional[str]:
        """
        Save PDF buffer to disk
        
        Args:
            doc_id: Document ID for filename
            pdf_buffer: Binary PDF data
            
        Returns:
            File path if saved successfully, None otherwise
        """
        try:
            file_path = self.output_dir / f"{doc_id}.pdf"
            with open(file_path, 'wb') as f:
                f.write(pdf_buffer)
            self.logger.debug(f"PDF saved to disk: {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save PDF {doc_id} to disk: {str(e)}")
            return None
    
    def fetch_document(self, doc_id: str, filename: str = None) -> Dict:
        """
        Fetch a single document by doc_id
        
        Args:
            doc_id: Document ID to fetch
            filename: Optional filename (not used, for compatibility)
            
        Returns:
            Dictionary with success, local_path, and error information
        """
        if not doc_id:
            return {
                'success': False,
                'local_path': None,
                'error': 'Missing doc_id',
                'pdf_buffer': None
            }
        
        # Use the existing _make_request method
        success, pdf_buffer, error, status_code = self._make_request(doc_id)
        
        if success:
            local_path = None
            
            # Save to disk if requested
            if self.save_to_disk:
                local_path = self._save_pdf_to_disk(doc_id, pdf_buffer)
            
            return {
                'success': True,
                'local_path': local_path,
                'error': '',
                'pdf_buffer': pdf_buffer,
                'file_size': len(pdf_buffer),
                'doc_id': doc_id
            }
        else:
            return {
                'success': False,
                'local_path': None,
                'error': error,
                'pdf_buffer': None,
                'status_code': status_code,
                'doc_id': doc_id
            }

    def fetch_documents(self, records: List[Dict]) -> Dict:
        """
        Fetch documents for all records
        
        Args:
            records: List of validated records from Module 1
            
        Returns:
            Dictionary with success/failed documents and summary
        """
        successful_docs = []
        failed_docs = []
        
        total_records = len(records)
        self.logger.info(f"Starting document fetch for {total_records} records")
        
        for i, record in enumerate(records, 1):
            doc_id = str(record.get('doc_id', '')).strip()
            
            if not doc_id:
                failed_docs.append({
                    'doc_id': '',
                    'error': 'Missing doc_id',
                    'status_code': 0,
                    'patient': record.get('patient', ''),
                    'received_on': record.get('received_on', '')
                })
                continue
            
            self.logger.info(f"Processing {i}/{total_records}: doc_id {doc_id}")
            
            # Fetch document
            result = self.fetch_document(doc_id)
            
            if result['success']:
                # Create successful record
                doc_record = {
                    'doc_id': doc_id,
                    'pdf_buffer': result['pdf_buffer'],
                    'file_name': f"{doc_id}.pdf",
                    'status': 'success',
                    'error': '',
                    'file_size': result['file_size'],
                    'patient': record.get('patient', ''),
                    'received_on': record.get('received_on', ''),
                    'original_record': record
                }
                
                successful_docs.append(doc_record)
                
                self.logger.info(f"✅ Successfully fetched doc_id {doc_id} ({result['file_size']} bytes)")
                
            else:
                # Create failed record
                failed_record = {
                    'doc_id': doc_id,
                    'error': result['error'],
                    'status_code': result['status_code'],
                    'patient': record.get('patient', ''),
                    'received_on': record.get('received_on', ''),
                    'original_record': record
                }
                
                failed_docs.append(failed_record)
                self.logger.error(f"❌ Failed to fetch doc_id {doc_id}: {result['error']} (status: {result['status_code']})")
        
        # Create summary
        summary = {
            'total': total_records,
            'success': len(successful_docs),
            'failed': len(failed_docs),
            'success_rate': round((len(successful_docs) / total_records * 100), 2) if total_records > 0 else 0
        }
        
        self.logger.info(f"Document fetch complete: {summary['success']}/{summary['total']} successful ({summary['success_rate']}%)")
        
        return {
            'success': successful_docs,
            'failed': failed_docs,
            'summary': summary
        }

def save_failed_fetches(failed_docs: List[Dict], output_path: str):
    """
    Save failed document fetches to CSV
    
    Args:
        failed_docs: List of failed document records
        output_path: Path to save the CSV file
    """
    if not failed_docs:
        return
    
    # Create DataFrame with relevant columns
    df_failed = pd.DataFrame(failed_docs)
    
    # Select and order columns that exist
    available_columns = df_failed.columns.tolist()
    desired_columns = ['doc_id', 'error', 'status_code', 'patient', 'received_on']
    columns = [col for col in desired_columns if col in available_columns]
    
    # Add missing columns with default values
    for col in desired_columns:
        if col not in df_failed.columns:
            if col == 'status_code':
                df_failed[col] = 0
            else:
                df_failed[col] = ''
    
    # Select columns in desired order
    df_failed = df_failed[desired_columns]
    
    # Save to CSV
    df_failed.to_csv(output_path, index=False)
    
def save_fetch_summary(result: Dict, output_path: str):
    """
    Save document fetch summary report
    
    Args:
        result: Result dictionary from fetch_documents or summary from main.py
        output_path: Path to save the report
    """
    # Handle different summary formats
    if 'summary' in result:
        # Format from fetch_documents method
        summary = result['summary']
        successful_docs = result.get('success', [])
        failed_docs = result.get('failed', [])
    else:
        # Format from main.py
        summary = {
            'total': result.get('total_attempted', 0),
            'success': result.get('successful_downloads', 0),
            'failed': result.get('failed_downloads', 0),
            'success_rate': result.get('success_rate', 0)
        }
        successful_docs = []
        failed_docs = []
    
    with open(output_path, 'w') as f:
        f.write("Module 2 - Document Fetcher Summary Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Documents: {summary['total']}\n")
        f.write(f"Successfully Fetched: {summary['success']}\n")
        f.write(f"Failed to Fetch: {summary['failed']}\n")
        f.write(f"Success Rate: {summary['success_rate']}%\n\n")
        
        # Only show detailed failed documents if available
        if 'failed' in result and result['failed']:
            f.write("Failed Documents:\n")
            f.write("-" * 30 + "\n")
            for failed in result['failed']:
                f.write(f"Doc ID: {failed.get('doc_id', 'Unknown')}\n")
                f.write(f"Error: {failed.get('error', 'Unknown error')}\n")
                f.write(f"Status Code: {failed.get('status_code', 'Unknown')}\n")
                f.write(f"Patient: {failed.get('patient', 'Unknown')}\n")
                f.write("-" * 30 + "\n")
        
        # Only show successful documents summary if available
        if successful_docs:
            f.write("\nSuccessful Documents Summary:\n")
            f.write("-" * 30 + "\n")
            total_size = sum(doc.get('file_size', 0) for doc in successful_docs)
            f.write(f"Total PDF Size: {total_size / (1024*1024):.2f} MB\n")
            
            if successful_docs:
                avg_size = total_size / len(successful_docs)
                f.write(f"Average PDF Size: {avg_size / 1024:.2f} KB\n")
        else:
            f.write("\nNote: Detailed document information not available in this summary format.\n")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Example test data
    test_records = [
        {
            "doc_id": "8461903",
            "patient": "Martin, Sharon",
            "received_on": "03/04/2024",
            "status": "Signed"
        }
    ]
    
    try:
        # Initialize fetcher
        fetcher = DocumentFetcher(save_to_disk=True)
        
        # Fetch documents
        result = fetcher.fetch_documents(test_records)
        
        print(f"Results: {result['summary']}")
        
    except Exception as e:
        print(f"Error: {e}") 