# 🏥 Patient Migration RPA Bot - End-to-End Workflow (Modular Breakdown)

---

## 🚀 Objective

Build a robust, production-grade script (or bot) that:

* Reads input patient/order metadata from a CSV
* Fetches and extracts document text via API
* Uses a **local LLM** (e.g., `phi` via **Ollama**) to extract key patient & order details
* Checks for existing patients via **MRN** (Medical Record Number)
* Pushes clean, validated patient & order data to a new platform via API
* Logs all actions, errors, and statuses in structured output CSVs

---

## 🔧 API Configuration

```python
TOKEN = os.getenv("AUTH_TOKEN")
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
DOC_STATUS_URL = "https://api.doctoralliance.com/document/get?docId.id="
PATIENT_CREATE_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/create"
ORDER_PUSH_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Order"
```

---

## �� Modular Breakdown

### **Module 1: `input_reader.py`**

**Purpose**: Read and preprocess the input CSV

* ✅ Reads input CSV (`primacare.csv`)
* ✅ Parses: `doc_id`, `agency`, `received_on`, `SOC`, `cert_period`, `cert_to`
* ✅ Normalizes dates to `MM/DD/YYYY`
* ✅ Skips invalid or incomplete rows

---

### **Module 2: `document_fetcher.py`**

**Purpose**: Fetch documents using API based on `doc_id`

* ✅ API GET: `https://api.doctoralliance.com/document/getfile?docId.id={doc_id}`
* ✅ Returns: Base64 PDF buffer and DA Backoffice ID

---

### **Module 3: `text_extractor.py`**

**Purpose**: Extract text from PDF

* ✅ Detects if PDF is **scanned** or **digital**
* ✅ For scanned: Uses **OCR via Tesseract** (with PyMuPDF)
* ✅ For digital: Uses **PDFPlumber**
* ✅ Returns cleaned, normalized text

---

### **Module 4: `llm_parser.py`**

**Purpose**: Extract structured patient/order data using local LLM

* ✅ Calls **local Phi model** via **Ollama API**
* ✅ Feeds cleaned document text and well-structured prompts
* ✅ Expects strict JSON with all key fields:

  * `patientFName`, `patientLName`, `dob`, `medicalRecordNo`, etc.
  * `episodeDiagnoses`, `orderNo`, `orderDate`, etc.
* ✅ Post-processes:

  * Normalize gender to `MALE/FEMALE`
  * Normalize all date fields to `MM/DD/YYYY`
  * Format name as `LastName, MiddleName, FirstName`

---

### **Module 5: `validator.py`**

**Purpose**: Ensure extracted data meets requirements

* ✅ Checks:

  * Required fields (e.g., MRN, DOB, Name, etc.) are present
  * Valid date format
  * Proper sex values (only `MALE`, `FEMALE`)
* ✅ Fails early and logs if fields are invalid or missing

---

### **Module 6: `deduplication.py`**

**Purpose**: Prevent duplicate patient creation

* ✅ Deduplication **only via MRN**
* ✅ API GET: `/api/Patient/company/pg/{PG_ID}`
* ✅ Searches list for matching `medicalRecordNo`
* ✅ Returns `patient_id` if already exists

---

### **Module 7: `patient_pusher.py`**

**Purpose**: Create new patient if not exists

* ⬜ Adds:

  * `pgCompanyId`, `companyId`, `physicianGroup`, `physicianGroupNPI`
* ⬜ POST to: `/api/Patient/create`
* ⬜ Logs success/failure with status code and message

---

### **Module 8: `order_pusher.py`**

**Purpose**: Push order details to API

* ⬜ Ensures episode dates (SOC, SOE, EOE) are available
* ⬜ POST to: `/api/Order`
* ⬜ Handles `201`, `409`, and logs response

---

### **Module 9: `logger.py`**

**Purpose**: Unified, color-coded, timestamped logging system

* ⬜ Logs:

  * Info, Success, Warning, Error, Progress
* ⬜ Saves to `logs/processing_log_*.txt`

---

### **Module 10: `output_writer.py`**

**Purpose**: Save all output to CSV

* ⬜ Output 1: `csv_outputs/extracted_patients_*.csv`
* ⬜ Output 2: `api_outputs/api_push_details_*.csv`
* ⬜ Includes all fields from input + API status, errors, timestamps

---

### **Module 11: `main.py`**

**Purpose**: Orchestrates the workflow end-to-end

1. ⬜ Load input CSV
2. ⬜ For each row:

   * Fetch document
   * Extract text (OCR or digital)
   * Feed to LLM via Ollama
   * Parse & validate JSON output
   * Check for existing patient (MRN)
   * Create patient if needed
   * Push order
   * Write result to logs and output CSVs

---

## 📊 Final Workflow Summary

### 📅 Patient Push

* Check if patient exists using `MRN`
* If not, create patient using `/api/Patient/create`
* Store `patient_id` for further reference

### 📧 Order Push

* Always tied to the `patient_id`
* Pushes order to `/api/Order`
* Includes date fields, status fields, physician signature status

📉 One Patient ➔ Many Orders possible

---

## 🔧 Local LLM Setup (Ollama)

### 1. Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Phi Model:

```bash
ollama pull phi
```

### 3. Run the model:

```bash
ollama run phi
```

### 4. API Call Example:

```python
import requests
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "phi",
    "prompt": prompt,
    "stream": False
})
data = response.json()["response"]
```

---

## ✅ Final Output

You will get:

* A patient push tracking CSV (`api_push_details_*.csv`)
* A final patient data CSV (`extracted_patients_*.csv`)
* A detailed log file per run (`logs/processing_log_*.txt`)
* Accurate MRN-based deduplication
* State-of-the-art LLM extraction using local model

---

## 📦 Optional Enhancements

* [ ] CLI flags (`--csv`, `--rows`, `--log`)
* [ ] Email/Slack summary alert after run
* [ ] Retry logic on failed requests
* [ ] Unit tests for LLM parsing, MRN matching, validators
