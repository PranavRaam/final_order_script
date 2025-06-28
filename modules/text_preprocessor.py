import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Common OCR misreads and their corrections
COMMON_CORRECTIONS = {
    'Ptient': 'Patient',
    'Pt ': 'Patient ',
    'DOB.': 'DOB:',
    'Da te': 'Date',
    'Mrn': 'MRN',
}

# Regex patterns that denote boiler-plate headers/footers we want to strip
HEADER_FOOTER_PATTERNS: List[re.Pattern] = [
    re.compile(r'^\s*Page\s+\d+\s*(of\s*\d+)?', re.IGNORECASE),
    re.compile(r'^\s*Fax(ed)?\b.*$', re.IGNORECASE),
    re.compile(r'^\s*CONFIDENTIAL\b.*$', re.IGNORECASE),
]

SECTION_PATTERNS = {
    '<HEADER>': [re.compile(r'^(PATIENT INFORMATION|PATIENT DETAILS|PATIENT:)\b', re.IGNORECASE)],
    '<ORDERS>': [re.compile(r'^(ORDERS?|ORDER INFORMATION)\b', re.IGNORECASE)],
    '<DIAGNOSES>': [re.compile(r'^(DIAGNOSIS|DIAGNOSES|DX:)\b', re.IGNORECASE)],
}

SECTION_LABELS = ['<HEADER>', '<ORDERS>', '<DIAGNOSES>']


def _apply_common_corrections(line: str) -> str:
    for wrong, correct in COMMON_CORRECTIONS.items():
        line = line.replace(wrong, correct)
    return line


def _strip_headers_footers(lines: List[str]) -> List[str]:
    cleaned = []
    for ln in lines:
        skip = False
        for pattern in HEADER_FOOTER_PATTERNS:
            if pattern.match(ln):
                skip = True
                break
        if not skip:
            cleaned.append(ln)
    return cleaned


def _dehyphenate(text: str) -> str:
    # Join word broken by hyphen at end of line
    return re.sub(r'-\n\s*', '', text)


def _insert_section_tags(lines: List[str]) -> List[str]:
    tagged = []
    for ln in lines:
        added_tag = False
        for tag, patterns in SECTION_PATTERNS.items():
            if any(pat.match(ln) for pat in patterns):
                tagged.append(tag)
                added_tag = True
                break
        tagged.append(ln)
    return tagged


def preprocess_text(raw_text: str) -> str:
    """Clean OCR text and add lightweight section tags.

    Steps:
    1. De-hyphenate words broken at line ends
    2. Split into lines, strip boilerplate headers/footers
    3. Apply common OCR corrections
    4. Insert simple section tags (<HEADER>, <ORDERS>, <DIAGNOSES>)
    5. Join back into a single string
    """
    if not raw_text:
        return ""

    # Step 1: de-hyphenate at line breaks
    text = _dehyphenate(raw_text)

    # Step 2: line-wise processing
    lines = text.splitlines()
    lines = _strip_headers_footers(lines)

    # Step 3: OCR corrections
    lines = [_apply_common_corrections(ln) for ln in lines]

    # Step 4: add tags
    lines = _insert_section_tags(lines)

    # Step 5: collapse multiple blanks and join
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # compress excessive blanks
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)  # collapse spaces

    logger.debug("Pre-processing completed: original %d chars → cleaned %d chars", len(raw_text), len(cleaned_text))
    return cleaned_text


def split_sections(cleaned_text: str) -> dict:
    """Split cleaned text into sections keyed by label without angle brackets.

    Returns a dict like {'HEADER': 'text...', 'ORDERS': 'text...', 'DIAGNOSES': 'text...'}
    Unknown text that appears before any tag or between tags but not within
    a labelled block is ignored for now (can be added under 'OTHER' if needed).
    """
    sections = {}
    current_label = None
    buffer_lines: List[str] = []

    for line in cleaned_text.splitlines():
        if line.strip() in SECTION_LABELS:
            # flush previous buffer
            if current_label is not None:
                sections[current_label] = "\n".join(buffer_lines).strip()
            # start new section
            current_label = line.strip().strip('<>').upper()
            buffer_lines = []
        else:
            buffer_lines.append(line)
    # flush last buffer
    if current_label is not None:
        sections[current_label] = "\n".join(buffer_lines).strip()
    return sections 