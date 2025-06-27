"""
Module 1: Input Reader - Practical Version
Purpose: Load CSV, validate required fields, normalize data, return clean results.
"""

import pandas as pd
import re
import chardet
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required fields for validation
REQUIRED_FIELDS = ["doc_id", "patient", "received_on", "status"]

# Column mapping for header normalization
COLUMN_MAP = {
    "id": "doc_id",
    "document_id": "doc_id",
    "doc id": "doc_id",
    "patient": "patient",
    "patient_name": "patient",
    "patient name": "patient",
    "name": "patient",
    "received on": "received_on",
    "received_on": "received_on",
    "receivedon": "received_on",
    "date_received": "received_on",
    "status": "status",
    "doc_status": "status",
    "document_status": "status",
    "physician": "physician",
    "doc type": "doc_type",
    "doc_type": "doc_type",
    "document_type": "doc_type",
    "facility": "facility",
    "facility type": "facility_type",
    "facility_type": "facility_type",
    "agency": "agency",
    "soc": "soc",
    "start_of_care": "soc",
    "start of care": "soc"
}

# Common date formats
DATE_FORMATS = [
    "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
    "%m/%d/%y", "%m-%d-%y", "%m.%d.%y", 
    "%Y-%m-%d", "%Y/%m/%d",
    "%B %d, %Y", "%b %d, %Y"
]


def detect_encoding(filepath: str) -> str:
    """Detect file encoding, fallback to UTF-8"""
    try:
        with open(filepath, 'rb') as f:
            sample = f.read(10000)
        result = chardet.detect(sample)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        
        if confidence < 0.7:
            logger.warning(f"Low confidence encoding detection, using UTF-8")
            return 'utf-8'
        
        logger.info(f"Detected encoding: {encoding}")
        return encoding
    except Exception as e:
        logger.error(f"Encoding detection failed: {e}, using UTF-8")
        return 'utf-8'


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names using COLUMN_MAP"""
    column_mapping = {}
    
    for col in df.columns:
        # Clean column name for matching
        clean_col = str(col).lower().strip().replace('_', ' ').replace('-', ' ')
        clean_col = re.sub(r'\s+', ' ', clean_col)
        
        if clean_col in COLUMN_MAP:
            column_mapping[col] = COLUMN_MAP[clean_col]
            logger.info(f"Mapped column: '{col}' -> '{COLUMN_MAP[clean_col]}'")
    
    return df.rename(columns=column_mapping)


def normalize_date(date_str: str) -> Optional[str]:
    """Normalize date to MM/DD/YYYY format"""
    if pd.isna(date_str) or not str(date_str).strip():
        return None
    
    date_str = str(date_str).strip()
    
    # Try each date format
    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%m/%d/%Y")
        except ValueError:
            continue
    
    # Try pandas fallback
    try:
        parsed_date = pd.to_datetime(date_str, infer_datetime_format=True)
        return parsed_date.strftime("%m/%d/%Y")
    except Exception:
        logger.warning(f"Could not parse date: {date_str}")
        return None


def normalize_patient_name(name: str) -> str:
    """Normalize patient name to 'Last, First' Title Case"""
    if pd.isna(name) or not str(name).strip():
        return ""
    
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
    
    # If already in "Last, First" format
    if ',' in name:
        parts = name.split(',', 1)
        last = parts[0].strip().title()
        first = parts[1].strip().title() if len(parts) > 1 else ""
        return f"{last}, {first}".rstrip(', ')
    
    # If in "First Last" format
    name_parts = name.split()
    if len(name_parts) >= 2:
        last_name = name_parts[-1].title()
        first_names = " ".join(name_parts[:-1]).title()
        return f"{last_name}, {first_names}"
    elif len(name_parts) == 1:
        return name_parts[0].title()
    
    return name.title()


def validate_and_clean_row(row: Dict, row_number: int) -> Tuple[bool, Dict, List[str]]:
    """
    Validate and clean a single row
    
    Returns:
        (is_valid, cleaned_row, errors)
    """
    errors = []
    cleaned_row = {}
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        value = row.get(field, "")
        if pd.isna(value) or not str(value).strip():
            errors.append(f"Missing {field}")
        else:
            # Clean the field
            cleaned_value = str(value).strip()
            
            # Special handling for specific fields
            if field == "received_on":
                normalized_date = normalize_date(cleaned_value)
                if normalized_date:
                    cleaned_row[field] = normalized_date
                else:
                    errors.append(f"Invalid date format in {field}")
                    cleaned_row[field] = cleaned_value
            elif field == "patient":
                cleaned_row[field] = normalize_patient_name(cleaned_value)
            else:
                cleaned_row[field] = cleaned_value
    
    # Add other fields if present
    for field, value in row.items():
        if field not in REQUIRED_FIELDS and not pd.isna(value):
            cleaned_row[field] = str(value).strip()
    
    # Add row number for tracking
    cleaned_row["row_number"] = row_number
    
    is_valid = len(errors) == 0
    return is_valid, cleaned_row, errors


def load_csv_file(filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load CSV file with encoding detection and error handling"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect encoding
    encoding = detect_encoding(str(filepath))
    
    try:
        # Load CSV
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            dtype=str,
            na_values=['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a'],
            keep_default_na=True,
            nrows=max_rows
        )
        
        logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Normalize column names
        df = normalize_column_names(df)
        
        return df
        
    except UnicodeDecodeError:
        # Try fallback encodings
        for fallback_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(filepath, encoding=fallback_encoding, dtype=str, nrows=max_rows)
                logger.info(f"Successfully loaded with fallback encoding: {fallback_encoding}")
                return normalize_column_names(df)
            except Exception:
                continue
        raise Exception("Could not read file with any encoding")
    
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def read_input_file(filepath: str, max_rows: Optional[int] = None) -> Dict:
    """
    Main function to read and process input CSV file
    
    Args:
        filepath: Path to CSV file
        max_rows: Optional limit on rows to process
    
    Returns:
        Dictionary with valid_rows, invalid_rows, and summary
    """
    try:
        # Load the CSV file
        df = load_csv_file(filepath, max_rows)
        
        # Process each row
        valid_rows = []
        invalid_rows = []
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            is_valid, cleaned_row, errors = validate_and_clean_row(row_dict, idx + 1)
            
            if is_valid:
                valid_rows.append(cleaned_row)
            else:
                invalid_rows.append({
                    "row_number": idx + 1,
                    "errors": errors,
                    "raw": row_dict
                })
        
        # Create summary
        summary = {
            "total": len(df),
            "valid": len(valid_rows),
            "invalid": len(invalid_rows),
            "success_rate": round((len(valid_rows) / len(df)) * 100, 1) if len(df) > 0 else 0
        }
        
        logger.info(f"Processing complete: {summary['valid']}/{summary['total']} valid rows ({summary['success_rate']}%)")
        
        return {
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


def save_validation_report(result: Dict, output_path: str):
    """Save a simple validation report"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Input Reader - Validation Report\n")
        f.write("=" * 40 + "\n\n")
        
        summary = result['summary']
        f.write(f"Total rows: {summary['total']}\n")
        f.write(f"Valid rows: {summary['valid']}\n")
        f.write(f"Invalid rows: {summary['invalid']}\n")
        f.write(f"Success rate: {summary['success_rate']}%\n\n")
        
        if result['invalid_rows']:
            f.write("Invalid Rows:\n")
            f.write("-" * 20 + "\n")
            for invalid_row in result['invalid_rows'][:20]:  # Limit to first 20
                f.write(f"Row {invalid_row['row_number']}: {', '.join(invalid_row['errors'])}\n")
            
            if len(result['invalid_rows']) > 20:
                f.write(f"... and {len(result['invalid_rows']) - 20} more invalid rows\n")
    
    logger.info(f"Validation report saved to: {output_path}")


def save_invalid_rows(invalid_rows: List[Dict], output_path: str):
    """Save invalid rows to CSV for debugging"""
    if not invalid_rows:
        logger.info("No invalid rows to save")
        return
    
    # Flatten invalid rows for CSV export
    flattened_rows = []
    for invalid_row in invalid_rows:
        flat_row = invalid_row['raw'].copy()
        flat_row['validation_errors'] = '; '.join(invalid_row['errors'])
        flat_row['original_row_number'] = invalid_row['row_number']
        flattened_rows.append(flat_row)
    
    df_invalid = pd.DataFrame(flattened_rows)
    df_invalid.to_csv(output_path, index=False)
    logger.info(f"Saved {len(invalid_rows)} invalid rows to: {output_path}")


# Public API for other modules
class InputReader:
    """Simple class interface for other modules"""
    
    @staticmethod
    def process_file(filepath: str, max_rows: Optional[int] = None) -> Dict:
        """Process a CSV file and return structured results"""
        return read_input_file(filepath, max_rows)
    
    @staticmethod
    def get_valid_records(filepath: str, max_rows: Optional[int] = None) -> List[Dict]:
        """Get only valid records from a CSV file"""
        result = read_input_file(filepath, max_rows)
        return result['valid_rows']
    
    @staticmethod
    def validate_file(filepath: str, max_rows: Optional[int] = None) -> bool:
        """Check if file has any valid records"""
        try:
            result = read_input_file(filepath, max_rows)
            return result['summary']['valid'] > 0
        except Exception:
            return False 