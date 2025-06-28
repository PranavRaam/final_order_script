"""
Module 3: Text Extractor
Extracts clean text from PDF documents (digital or scanned) for LLM processing.

Features:
- Smart document type detection (digital vs scanned)
- Fast pdfplumber extraction for digital PDFs
- OCR fallback for scanned/signed documents
- Text quality scoring and validation
- Multi-language OCR support
- Smart page filtering (skip blank/noise pages)
- Comprehensive error handling and logging
"""

import io
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import concurrent.futures  # NEW
import multiprocessing  # NEW
import pypdfium2 as pdfium  # NEW: fast PDF → image renderer
from easyocr import Reader as EasyOCRReader  # NEW OCR engine
import numpy as np  # Required by easyocr

# Initialise easyocr reader once (CPU mode)
_EASY_READER = EasyOCRReader(['en'], gpu=False)

class TextExtractor:
    """
    Extracts text from PDF documents with intelligent fallback mechanisms
    """
    
    def __init__(self, save_extracted_text: bool = False, output_dir: str = "extracted_texts"):
        """
        Initialize TextExtractor
        
        Args:
            save_extracted_text: Whether to save extracted text to disk for debugging
            output_dir: Directory to save extracted text files
        """
        self.save_extracted_text = save_extracted_text
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if saving text
        if self.save_extracted_text:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Text output directory created: {self.output_dir}")
        
        # Configure Tesseract for better OCR (if available)
        self._configure_tesseract()
    
    def _configure_tesseract(self):
        """Configure Tesseract OCR settings"""
        try:
            # Test if Tesseract is available
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            
            # Improved OCR configuration for medical documents
            # Removed problematic character whitelist that was causing "No closing quotation" errors
            self.ocr_config = r'--oem 3 --psm 6'
            
            self.logger.debug("Tesseract OCR configured successfully")
            
        except Exception as e:
            self.tesseract_available = False
            self.logger.warning(f"Tesseract not available: {e}")
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """
        Calculate text quality score (0.0 to 1.0)
        Higher score = better quality text
        
        Args:
            text: Extracted text
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        text = text.strip()
        total_chars = len(text)
        
        if total_chars < 10:
            return 0.1
        
        # Count different character types
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        punctuation = sum(1 for c in text if c in '.,;:()[]{}/-_@#$%^&*+=<>?!"\'')
        
        # Calculate ratios
        letter_ratio = letters / total_chars
        digit_ratio = digits / total_chars
        space_ratio = spaces / total_chars
        punct_ratio = punctuation / total_chars
        
        # Quality scoring
        quality_score = 0.0
        
        # Good letter ratio (medical docs should have plenty of text)
        if 0.4 <= letter_ratio <= 0.8:
            quality_score += 0.4
        elif letter_ratio > 0.2:
            quality_score += 0.2
        
        # Reasonable digit ratio (dates, IDs, etc.)
        if 0.05 <= digit_ratio <= 0.3:
            quality_score += 0.2
        
        # Good space ratio (readable text)
        if 0.1 <= space_ratio <= 0.25:
            quality_score += 0.2
        
        # Some punctuation (structured text)
        if 0.02 <= punct_ratio <= 0.15:
            quality_score += 0.1
        
        # Length bonus (longer text usually better)
        if total_chars > 500:
            quality_score += 0.1
        elif total_chars > 200:
            quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    def _extract_with_pdfplumber(self, pdf_buffer: bytes, doc_id: str) -> Tuple[str, float]:
        """
        Extract text using pdfplumber (for digital PDFs)
        
        Args:
            pdf_buffer: PDF binary data
            doc_id: Document ID for logging
            
        Returns:
            Tuple of (extracted_text, quality_score)
        """
        try:
            extracted_text = ""
            
            with io.BytesIO(pdf_buffer) as pdf_stream:
                with pdfplumber.open(pdf_stream) as pdf:
                    total_pages = len(pdf.pages)
                    self.logger.debug(f"Processing {total_pages} pages with pdfplumber for doc_id {doc_id}")
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n\n"
                                self.logger.debug(f"Extracted {len(page_text)} chars from page {page_num}")
                        except Exception as e:
                            self.logger.warning(f"Failed to extract text from page {page_num} of doc_id {doc_id}: {e}")
                            continue
            
            # Calculate quality score
            quality_score = self._calculate_text_quality_score(extracted_text)
            
            self.logger.debug(f"pdfplumber extraction for doc_id {doc_id}: {len(extracted_text)} chars, quality: {quality_score:.2f}")
            
            return extracted_text.strip(), quality_score
            
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed for doc_id {doc_id}: {e}")
            return "", 0.0
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _is_scanned_pdf(self, pdf_buffer: bytes) -> bool:
        """Advanced scanned-PDF detection inspired by final_version.py logic."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_buffer)) as pdf:
                total_text = ""
                pages_checked = min(3, len(pdf.pages))
                for page in pdf.pages[:pages_checked]:
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    total_text += txt.strip()

                # No text at all → scanned
                if not total_text:
                    return True
                # Very short text → likely scanned
                if len(total_text) < 100:
                    return True
                # Repetition analysis
                lines = [ln.strip() for ln in total_text.split("\n") if ln.strip()]
                if lines:
                    repetition_ratio = len(set(lines)) / len(lines)
                    if repetition_ratio < 0.3:
                        return True
                # Look for meaningful keywords to prove it *is* digital text
                indicators = [
                    "patient", "name", "date of birth", "dob", "address", "diagnosis", "medical", "record",
                ]
                has_keywords = any(ind in total_text.lower() for ind in indicators)
                if not has_keywords and len(total_text) < 500:
                    return True
            return False
        except Exception as e:
            self.logger.debug(f"Scanned-PDF detection failed – assuming scanned: {e}")
            return True
    
    def _extract_with_ocr(self, pdf_buffer: bytes, doc_id: str) -> Tuple[str, float]:
        """Parallel OCR extraction (faster) using PyMuPDF + Tesseract with easyOCR fallback."""
        if not self.tesseract_available:
            # Fallback directly to easyOCR if Tesseract not present
            return self._ocr_with_easyocr(pdf_buffer, doc_id)
        try:
            pdf_document = fitz.open(stream=pdf_buffer, filetype="pdf")
            page_count = len(pdf_document)

            def page_to_image(idx: int):
                # Try pdfium render first (faster & sharper)
                try:
                    page = pdfium.PdfDocument(pdf_buffer).get_page(idx)
                    pil_img = page.render(scale=2.5).to_pil()
                    return pil_img
                except Exception:
                    # fallback to PyMuPDF
                    page = pdf_document[idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
                    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            def ocr_tesseract(pil_img: Image.Image):
                try:
                    return pytesseract.image_to_string(pil_img, config=self.ocr_config)
                except Exception:
                    try:
                        return pytesseract.image_to_string(pil_img)
                    except Exception:
                        return ""

            # parallel OCR
            max_workers = min(4, multiprocessing.cpu_count(), page_count)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exec:
                images = list(exec.map(page_to_image, range(page_count)))
                texts = list(exec.map(ocr_tesseract, images))
            
            pdf_document.close()
            full_text = "\n".join(texts)
            cleaned = re.sub(r"\s+", " ", full_text).strip()
            quality = self._calculate_text_quality_score(cleaned)

            # Fallback to easyOCR if quality very low
            if quality < 0.2:
                self.logger.debug(f"Tesseract OCR low quality ({quality:.2f}) → trying easyOCR", doc_id)
                cleaned_easy, q_easy = self._ocr_with_easyocr(pdf_buffer, doc_id)
                if q_easy > quality:
                    cleaned, quality = cleaned_easy, q_easy
            return cleaned, quality
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {doc_id}: {e}")
            # last chance easyOCR
            return self._ocr_with_easyocr(pdf_buffer, doc_id)

    def _ocr_with_easyocr(self, pdf_buffer: bytes, doc_id: str) -> Tuple[str, float]:
        """OCR using easyocr reader on rendered images."""
        try:
            pdf = pdfium.PdfDocument(pdf_buffer)
            texts = []
            for page_idx in range(len(pdf)):
                page = pdf.get_page(page_idx)
                pil_img = page.render(scale=2.5).to_pil()
                result = _EASY_READER.readtext(np.array(pil_img))
                page_text = " ".join([r[1] for r in result])
                texts.append(page_text)
            pdf.close()
            joined = " \n".join(texts)
            cleaned = re.sub(r"\s+", " ", joined).strip()
            quality = self._calculate_text_quality_score(cleaned)
            return cleaned, quality
        except Exception as exc:
            self.logger.error(f"easyOCR extraction failed for {doc_id}: {exc}")
            return "", 0.0
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove form feed characters
        text = text.replace('\x0c', '\n')
        
        # Normalize spaces and tabs but PRESERVE newlines for readability
        # Replace sequences of spaces / tabs with a single space, while leaving line breaks intact
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove excessive newlines (keep paragraph structure)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common header/footer patterns (basic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely artifacts
            if len(line) > 2:  
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _save_extracted_text(self, doc_id: str, text: str, is_scanned: bool):
        """
        Save extracted text to disk for debugging
        
        Args:
            doc_id: Document ID
            text: Extracted text
            is_scanned: Whether document was processed with OCR
        """
        try:
            filename = f"{doc_id}_{'ocr' if is_scanned else 'digital'}.txt"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Document ID: {doc_id}\n")
                f.write(f"Extraction Method: {'OCR' if is_scanned else 'Digital'}\n")
                f.write(f"Extraction Date: {datetime.now().isoformat()}\n")
                f.write(f"Text Length: {len(text)} characters\n")
                f.write("-" * 50 + "\n\n")
                f.write(text)
            
            self.logger.debug(f"Extracted text saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save extracted text for doc_id {doc_id}: {e}")
    
    def _extract_with_pymupdf(self, pdf_buffer: bytes, doc_id: str) -> Tuple[str, float]:
        """Light-weight text extraction using PyMuPDF (fitz). Slightly different engine than pdfplumber.
        Returns the extracted text and quality score."""
        try:
            extracted_text = ""
            with fitz.open(stream=pdf_buffer, filetype="pdf") as doc:
                total_pages = doc.page_count
                self.logger.debug(f"PyMuPDF extracting {total_pages} pages for doc_id {doc_id}")
                for page_num in range(total_pages):
                    try:
                        page = doc.load_page(page_num)
                        page_text = page.get_text("text")  # simple layout text
                        if page_text:
                            extracted_text += page_text + "\n\n"
                    except Exception as e:
                        self.logger.warning(f"PyMuPDF failed on page {page_num+1} of {doc_id}: {e}")
                        continue
            quality_score = self._calculate_text_quality_score(extracted_text)
            self.logger.debug(f"PyMuPDF extraction for {doc_id}: {len(extracted_text)} chars, quality {quality_score:.2f}")
            return extracted_text.strip(), quality_score
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed for {doc_id}: {e}")
            return "", 0.0
    
    def extract_text(self, pdf_path: str = None, pdf_buffer: bytes = None, doc_id: str = None) -> Dict:
        """
        Extract text from a single PDF document with intelligent fallback
        
        Args:
            pdf_path: Path to PDF file (if saved to disk)
            pdf_buffer: PDF binary data
            doc_id: Document ID for logging
            
        Returns:
            Dictionary with success, text, and error information
        """
        if not pdf_path and not pdf_buffer:
            return {
                'success': False,
                'text': '',
                'error': 'No PDF data provided (neither path nor buffer)',
                'quality_score': 0.0,
                'extraction_method': 'none',
                'is_scanned': False
            }
        
        # If we have a path but no buffer, read the file
        if pdf_path and not pdf_buffer:
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_buffer = f.read()
            except Exception as e:
                return {
                    'success': False,
                    'text': '',
                    'error': f'Failed to read PDF file: {str(e)}',
                    'quality_score': 0.0,
                    'extraction_method': 'none',
                    'is_scanned': False
                }
        
        if not doc_id:
            doc_id = Path(pdf_path).stem if pdf_path else 'unknown'
        
        try:
            # 1️⃣ Try pdfplumber first (fast) for digital PDFs, UNLESS advanced scan detector says otherwise
            if self._is_scanned_pdf(pdf_buffer):
                # Skip digital attempts if clearly scanned
                best_text, best_quality, method = "", 0.0, "none"
            else:
            text_plumber, q_plumber = self._extract_with_pdfplumber(pdf_buffer, doc_id)
            best_text, best_quality, method = text_plumber, q_plumber, 'pdfplumber'
            
            # 2️⃣ If quality mediocre, attempt PyMuPDF and keep higher-quality result
            if 0 < best_quality < 0.6:
                text_fitz, q_fitz = self._extract_with_pymupdf(pdf_buffer, doc_id)
                if q_fitz > best_quality:
                    best_text, best_quality, method = text_fitz, q_fitz, 'pymupdf'
            
            # 3️⃣ Decide if we need OCR (treat as scanned) based on final quality
            is_scanned = best_quality < 0.25
            if is_scanned:
                ocr_text, ocr_quality = self._extract_with_ocr(pdf_buffer, doc_id)
                if ocr_quality > best_quality:  # Only overwrite if OCR better
                    best_text, best_quality, method = ocr_text, ocr_quality, 'ocr'
            
            cleaned_text = self._clean_extracted_text(best_text)
            return {
                'success': True,
                'text': cleaned_text,
                'quality_score': best_quality,
                'extraction_method': method,
                'is_scanned': is_scanned
            }
            
        except Exception as e:
            error_msg = f"Unexpected extraction error: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'text': '',
                'quality_score': 0.0,
                'extraction_method': 'none',
                'error': error_msg,
                'is_scanned': False
            }

    def extract_text_from_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract text from a list of PDF documents
        
        Args:
            documents: List of document dictionaries from Module 2
            
        Returns:
            List of extraction results
        """
        results = []
        total_docs = len(documents)
        
        self.logger.info(f"Starting text extraction for {total_docs} documents")
        
        for i, doc in enumerate(documents, 1):
            doc_id = doc.get('doc_id', 'unknown')
            pdf_buffer = doc.get('pdf_buffer')
            
            self.logger.info(f"Processing {i}/{total_docs}: doc_id {doc_id}")
            
            if not pdf_buffer:
                result = {
                    'doc_id': doc_id,
                    'text': '',
                    'is_scanned': False,
                    'status': 'failed',
                    'error': 'No PDF buffer provided',
                    'quality_score': 0.0,
                    'extraction_method': 'none'
                }
                results.append(result)
                continue
            
            try:
                # Step 1: Try digital extraction first
                extraction_result = self.extract_text(pdf_buffer=pdf_buffer, doc_id=doc_id)
                
                # Step 4: Clean the text
                cleaned_text = self._clean_extracted_text(extraction_result['text'])
                
                # Step 5: Determine final status
                if len(cleaned_text.strip()) > 50 and extraction_result['quality_score'] > 0.2:
                    status = 'extracted'
                    error = ''
                else:
                    status = 'failed'
                    error = f'Poor text quality (score: {extraction_result['quality_score']:.2f}) or insufficient content'
                
                result = {
                    'doc_id': doc_id,
                    'text': cleaned_text,
                    'is_scanned': extraction_result['is_scanned'],
                    'status': status,
                    'error': error,
                    'quality_score': extraction_result['quality_score'],
                    'extraction_method': extraction_result['extraction_method'],
                    'original_document': doc
                }
                
                results.append(result)
                
                if status == 'extracted':
                    self.logger.info(f"✅ Successfully extracted {len(cleaned_text)} chars from doc_id {doc_id} (method: {extraction_result['extraction_method']})")
                else:
                    self.logger.error(f"❌ Failed to extract text from doc_id {doc_id}: {error}")
                
            except Exception as e:
                error_msg = f"Unexpected error during text extraction: {str(e)}"
                self.logger.error(f"❌ Error processing doc_id {doc_id}: {error_msg}")
                
                result = {
                    'doc_id': doc_id,
                    'text': '',
                    'is_scanned': False,
                    'status': 'failed',
                    'error': error_msg,
                    'quality_score': 0.0,
                    'extraction_method': 'none',
                    'original_document': doc
                }
                results.append(result)
        
        # Summary
        successful = len([r for r in results if r['status'] == 'extracted'])
        failed = len([r for r in results if r['status'] == 'failed'])
        
        self.logger.info(f"Text extraction complete: {successful}/{total_docs} successful ({successful/total_docs*100:.1f}%)")
        
        return results

    def extract_text_batch_for_module4(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract text from documents and return structured data for direct Module 4 consumption
        This avoids saving/reading individual text files for better efficiency
        
        Args:
            documents: List of document dictionaries from Module 2 or pre-extracted text
            
        Returns:
            List of structured extraction results ready for Module 4
        """
        results = []
        total_docs = len(documents)
        
        self.logger.info(f"🔄 Module 3: Processing {total_docs} documents for direct Module 4 handoff")
        
        # Track statistics
        successful_extractions = 0
        failed_extractions = 0
        extraction_methods = {'digital': 0, 'ocr': 0, 'pre_extracted': 0}
        quality_scores = []
        
        for i, doc in enumerate(documents, 1):
            doc_id = doc.get('doc_id', 'unknown')
            pdf_buffer = doc.get('pdf_buffer')
            pre_extracted_text = doc.get('content')  # For pre-extracted text from mock fetcher
            
            self.logger.info(f"  Processing {i}/{total_docs}: {doc_id}")
            
            # Handle pre-extracted text (from mock SharePoint fetcher)
            if pre_extracted_text:
                quality_score = self._calculate_text_quality_score(pre_extracted_text)
                result = {
                    'doc_id': doc_id,
                    'text': pre_extracted_text,
                    'status': 'extracted',
                    'error': '',
                    'quality_score': quality_score,
                    'extraction_method': 'pre_extracted',
                    'is_scanned': False,  # Assume pre-extracted text is clean
                    'text_length': len(pre_extracted_text),
                    'metadata': doc
                }
                results.append(result)
                successful_extractions += 1
                quality_scores.append(quality_score)
                extraction_methods['pre_extracted'] += 1
                continue
            
            # Handle PDF buffer extraction
            if not pdf_buffer:
                result = {
                    'doc_id': doc_id,
                    'text': '',
                    'status': 'failed',
                    'error': 'No PDF buffer or pre-extracted text provided',
                    'quality_score': 0.0,
                    'extraction_method': 'none',
                    'is_scanned': False,
                    'text_length': 0,
                    'metadata': doc
                }
                results.append(result)
                failed_extractions += 1
                continue
            
            try:
                # Extract text using the standard method
                extraction_result = self.extract_text(pdf_buffer=pdf_buffer, doc_id=doc_id)
                
                # Structure result for Module 4
                if extraction_result['success']:
                    result = {
                        'doc_id': doc_id,
                        'text': extraction_result['text'],
                        'status': 'extracted',
                        'error': '',
                        'quality_score': extraction_result['quality_score'],
                        'extraction_method': extraction_result['extraction_method'],
                        'is_scanned': extraction_result['is_scanned'],
                        'text_length': len(extraction_result['text']),
                        'metadata': doc
                    }
                    successful_extractions += 1
                    quality_scores.append(extraction_result['quality_score'])
                    extraction_methods[extraction_result['extraction_method']] += 1
                else:
                    result = {
                        'doc_id': doc_id,
                        'text': '',
                        'status': 'failed',
                        'error': extraction_result['error'],
                        'quality_score': extraction_result['quality_score'],
                        'extraction_method': extraction_result['extraction_method'],
                        'is_scanned': extraction_result['is_scanned'],
                        'text_length': 0,
                        'metadata': doc
                    }
                    failed_extractions += 1
                
                results.append(result)
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(f"❌ Error processing {doc_id}: {error_msg}")
                
                result = {
                    'doc_id': doc_id,
                    'text': '',
                    'status': 'failed',
                    'error': error_msg,
                    'quality_score': 0.0,
                    'extraction_method': 'none',
                    'is_scanned': False,
                    'text_length': 0,
                    'metadata': doc
                }
                results.append(result)
                failed_extractions += 1
        
        # Log summary statistics
        success_rate = (successful_extractions / total_docs) * 100 if total_docs > 0 else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        total_text_length = sum(r['text_length'] for r in results if r['status'] == 'extracted')
        
        self.logger.info(f"✅ Module 3 Summary:")
        self.logger.info(f"  - Successfully extracted: {successful_extractions}/{total_docs} ({success_rate:.1f}%)")
        self.logger.info(f"  - Failed extractions: {failed_extractions}")
        self.logger.info(f"  - Digital PDFs: {extraction_methods['digital']}")
        self.logger.info(f"  - OCR (Scanned): {extraction_methods['ocr']}")
        self.logger.info(f"  - Pre-extracted: {extraction_methods['pre_extracted']}")
        self.logger.info(f"  - Average quality score: {avg_quality:.3f}")
        self.logger.info(f"  - Total text extracted: {total_text_length:,} characters")
        
        return results

def save_failed_extractions(failed_extractions: List[Dict], output_path: str):
    """
    Save failed text extractions to CSV
    
    Args:
        failed_extractions: List of failed extraction records
        output_path: Path to save the CSV file
    """
    if not failed_extractions:
        return
    
    # Create DataFrame with relevant columns
    df_failed = pd.DataFrame(failed_extractions)
    
    # Select and order columns
    columns = ['doc_id', 'error', 'quality_score', 'extraction_method']
    df_failed = df_failed[columns]
    
    # Save to CSV
    df_failed.to_csv(output_path, index=False)

def save_extraction_summary(results: List[Dict], output_path: str):
    """
    Save text extraction summary report
    
    Args:
        results: List of extraction results
        output_path: Path to save the report
    """
    successful = [r for r in results if r['status'] == 'extracted']
    failed = [r for r in results if r['status'] == 'failed']
    
    with open(output_path, 'w') as f:
        f.write("Module 3 - Text Extractor Summary Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Documents: {len(results)}\n")
        f.write(f"Successfully Extracted: {len(successful)}\n")
        f.write(f"Failed Extractions: {len(failed)}\n")
        f.write(f"Success Rate: {len(successful)/len(results)*100:.1f}%\n\n")
        
        if successful:
            f.write("Extraction Methods:\n")
            f.write("-" * 30 + "\n")
            digital_count = len([r for r in successful if not r['is_scanned']])
            ocr_count = len([r for r in successful if r['is_scanned']])
            f.write(f"Digital PDFs: {digital_count}\n")
            f.write(f"OCR (Scanned): {ocr_count}\n\n")
            
            f.write("Quality Statistics:\n")
            f.write("-" * 30 + "\n")
            avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
            f.write(f"Average Quality Score: {avg_quality:.3f}\n")
            
            total_chars = sum(len(r['text']) for r in successful)
            avg_chars = total_chars / len(successful)
            f.write(f"Average Text Length: {avg_chars:.0f} characters\n")
            f.write(f"Total Extracted Text: {total_chars:,} characters\n\n")
        
        if failed:
            f.write("Failed Extractions:\n")
            f.write("-" * 30 + "\n")
            for failed_item in failed:
                f.write(f"Doc ID: {failed_item['doc_id']}\n")
                f.write(f"Error: {failed_item['error']}\n")
                f.write(f"Quality Score: {failed_item['quality_score']:.3f}\n")
                f.write("-" * 30 + "\n")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Example test - would normally come from Module 2
    test_documents = [
        {
            "doc_id": "test_123",
            "pdf_buffer": b"sample_pdf_data",  # Would be real PDF buffer
            "file_name": "test_123.pdf"
        }
    ]
    
    try:
        # Initialize extractor
        extractor = TextExtractor(save_extracted_text=True)
        
        # Extract text
        results = extractor.extract_text_from_documents(test_documents)
        
        print(f"Extraction results: {len(results)} processed")
        
    except Exception as e:
        print(f"Error: {e}") 