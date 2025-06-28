import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


class TextExtractor:
    """Light-weight PDF text extractor for clean US clinical PDFs.

    Strategy:
    1. Try fast text extraction with pdfplumber.
    2. If the PDF appears scanned (no text layer / too little text) or
       pdfplumber returns <200 chars, fall back to Tesseract OCR.
    3. Return a dict compatible with the old heavy extractor so that the
       rest of the pipeline keeps working untouched.
    """

    def __init__(self, save_extracted_text: bool = False, output_dir: str = "extracted_texts", ocr_dpi: int = 200):
        self.save_extracted_text = save_extracted_text
        self.output_dir = output_dir
        self.ocr_dpi = ocr_dpi
        self.logger = logging.getLogger(__name__)
        if self.save_extracted_text:
            os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _is_scanned_pdf(self, pdf_bytes: bytes) -> bool:
        """Heuristic to decide if a PDF is scanned (image-only)."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_text = "".join((page.extract_text() or "") for page in pdf.pages[:3])

            if not total_text:
                return True
            if len(total_text) < 100:
                return True

            lines = [ln.strip() for ln in total_text.split("\n") if ln.strip()]
            if lines and (len(set(lines)) / len(lines)) < 0.3:
                return True

            indicators = ["patient", "name", "dob", "date of birth", "address", "diagnosis", "medical", "record"]
            if (not any(ind in total_text.lower() for ind in indicators)) and len(total_text) < 500:
                return True

            return False
        except Exception as e:
            self.logger.debug(f"Scanned-PDF detection failed (assuming scanned): {e}")
            return True

    def _ocr_extract(self, pdf_bytes: bytes) -> str:
        """OCR the entire PDF with PyMuPDF rendering + Tesseract."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        def _page_ocr(idx: int) -> str:
            page = doc[idx]
            matrix = fitz.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img, config="--oem 3 --psm 6")

        max_workers = min(4, os.cpu_count() or 1, len(doc))
        if len(doc) == 1:
            texts = [_page_ocr(0)]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                texts = list(executor.map(_page_ocr, range(len(doc))))

        doc.close()
        return "\n".join(texts)

    def _calculate_quality_score(self, text: str) -> float:
        """Very simple length-based quality metric (0-1)."""
        if not text:
            return 0.0
        return min(len(text) / 5000.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_text(self, *, pdf_path: str = None, pdf_buffer: bytes = None, doc_id: str = None) -> Dict:
        """Extract text from a single PDF and return a rich result dict."""
        try:
            # Load file if a path is supplied
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    pdf_buffer = f.read()

            if not pdf_buffer:
                raise ValueError("No PDF data provided")

            is_scanned = self._is_scanned_pdf(pdf_buffer)
            extraction_method = "ocr" if is_scanned else "digital"

            # Fast path: pdfplumber for digital PDFs
            if extraction_method == "digital":
                with pdfplumber.open(io.BytesIO(pdf_buffer)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)

                # Fallback to OCR if the digital text is suspiciously short
                if not text or len(text.strip()) < 200:
                    self.logger.debug(f"{doc_id}: Weak digital text, switching to OCR")
                    text = self._ocr_extract(pdf_buffer)
                    extraction_method = "ocr"
                    is_scanned = True
            else:
                # OCR path for scanned PDFs
                text = self._ocr_extract(pdf_buffer)

            # Basic clean-up
            cleaned_text = re.sub(r"\s+", " ", text).strip()
            quality_score = self._calculate_quality_score(cleaned_text)

            # Optional save for inspection
            if self.save_extracted_text and cleaned_text:
                out_path = os.path.join(self.output_dir, f"{doc_id or 'document'}.txt")
                try:
                    with open(out_path, "w", encoding="utf-8") as fp:
                        fp.write(cleaned_text)
                except Exception as exc:
                    self.logger.debug(f"Could not save extracted text for {doc_id}: {exc}")

            success = bool(cleaned_text)
            error = "" if success else "No text extracted"

            return {
                "doc_id": doc_id,
                "text": cleaned_text,
                "success": success,
                "error": error,
                "quality_score": quality_score,
                "extraction_method": extraction_method,
                "is_scanned": is_scanned,
            }

        except Exception as e:
            self.logger.error(f"Failed to extract text for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "text": "",
                "success": False,
                "error": str(e),
                "quality_score": 0.0,
                "extraction_method": "none",
                "is_scanned": False,
            }

    def extract_text_batch_for_module4(self, documents: List[Dict]) -> List[Dict]:
        """Process a list of docs and produce Module-4-ready results."""
        results: List[Dict] = []
        for doc in documents:
            doc_id = doc.get("doc_id", "unknown")
            pdf_buffer = doc.get("pdf_buffer")
            pre_text = doc.get("content")  # Pre-provided text, skip extraction

            if pre_text:
                quality = self._calculate_quality_score(pre_text)
                results.append({
                    "doc_id": doc_id,
                    "text": pre_text,
                    "status": "extracted",
                    "error": "",
                    "quality_score": quality,
                    "extraction_method": "pre_extracted",
                    "is_scanned": False,
                    "text_length": len(pre_text),
                    "metadata": doc,
                })
                continue

            extraction_result = self.extract_text(pdf_buffer=pdf_buffer, doc_id=doc_id)
            status = "extracted" if extraction_result["success"] else "failed"

            results.append({
                "doc_id": doc_id,
                "text": extraction_result["text"],
                "status": status,
                "error": extraction_result["error"],
                "quality_score": extraction_result["quality_score"],
                "extraction_method": extraction_result["extraction_method"],
                "is_scanned": extraction_result["is_scanned"],
                "text_length": len(extraction_result["text"]),
                "metadata": doc,
            })

        return results 