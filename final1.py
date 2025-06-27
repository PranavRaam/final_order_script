import csv
import json
import base64
import io
import re
import os
from datetime import datetime, timedelta
import requests
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TOKEN = os.getenv("AUTH_TOKEN")  # Set this in .env
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
DOC_STATUS_URL = "https://api.doctoralliance.com/document/get?docId.id="
PATIENT_CREATE_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/create"
ORDER_PUSH_URL = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Order"
PG_ID = "161d97e7-4d84-4ed0-8d99-4147b75f8988"

# Load mappings
with open("output.json") as f:
    company_map = json.load(f)

if os.path.exists("hawthorn_internalmedicine.json") and os.path.getsize("hawthorn_internalmedicine.json") > 0:
    with open("hawthorn_internalmedicine.json") as f:
        created_patients = json.load(f)
else:
    created_patients = {}

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

AUDIT_PATIENTS_FILE = "audit_patients.csv"
AUDIT_ORDERS_FILE = "audit_orders.csv"

def get_pdf_text(doc_id):
    response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS)
    if response.status_code != 200:
        raise Exception("Failed to fetch document")
    daId = response.json()["value"]["patientId"]["id"]
    document_buffer = response.json()["value"]["documentBuffer"]
    pdf_bytes = base64.b64decode(document_buffer)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    edited = re.sub(r'\b\d[A-Z][A-Z0-9]\d[A-Z][A-Z0-9]\d[A-Z]{2}(?:\d{2})?\b', '', text)
    # print(edited)
    return [edited, daId]

def extract_patient_data(text):
    query = """
        You are an expert in medical documentation. Extract the following fields from the attached medical PDF and return them in the specified JSON format. Use the exact field names below, and provide values based strictly on the document content. This JSON will be sent directly to an API, so ensure the format is correct and do not return any additional text.
        Required JSON format:
        {
        "patientFName": "",
        "patientLName": "",
        "dob": "",
        "patientSex": "",
        "medicalRecordNo": "",
        "billingProvider": "",
        "npi": "",
        "nameOfAgency": "",
        "episodeDiagnoses": [
            {
            "startOfCare": "",
            "startOfEpisode": "",
            "endOfEpisode": ""
            "firstDiagnosis": "",
            "secondDiagnosis": "",
            "thirdDiagnosis": "",
            "fourthDiagnosis": "",
            "fifthDiagnosis": "",
            "sixthDiagnosis": ""
            }
        ]
        }
        Clarifications:
        Extract the following fields based only on explicit labels in the document:
        - startOfCare: Look for labels like "SOC", "Start of Care Date", or "SOC Date".
        - startOfEpisode: From labels like "Start Date", "Episode Start Date", "Certification Period From", or "From Date". If in a "X - Y" format, use the date before the dash.
        - endOfEpisode: From labels like "End Date", "Episode End Date", "Certification Period To", or "To Date". If in a "X - Y" format, use the date after the dash.
        - Episode dates may appear under headings like "Episode Dates".
        - Diagnosis Codes: Extract up to 6 ICD-10-CM codes (1 primary + up to 5 secondary). Format: starts with a letter, followed by 2 digits, optional dot, and up to 4 alphanumerics. Leave blank if not found.
        - Billing Provider: Name of the primary physician.
        - Do not infer values. Preserve original date formats. Return only the structured JSON.
    """
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json.loads(match.group()) if match else {}

def extract_order_data(text):
    query = """
        You are an expert in medical documentation. Extract the following fields from the attached medical PDF. Use the exact field names provided below. These fields must be extracted accurately, as they will be used to populate an API request.
    Return the data in the following format, using the field names exactly and preserving the order:

    1. orderNo - Unique order number identifying the medical document
    2. orderDate - Date the order was created
    3. startOfCare - Start of care date for the patient
    4. episodeStartDate - Start date of the care episode
    5. episodeEndDate - End date of the care episode
    6. documentID - Unique identifier for the document
    7. mrn - Medical Record Number of the patient
    8. patientName - Full name of the patient
    9. sentToPhysicianDate - Date the document was sent to the physician
    10. sentToPhysicianStatus - Boolean indicating whether the document was sent to the physician
    11. patientId - Unique patient identifier
    12. companyId - Identifier for the company managing the care
    13. bit64Url - 64-bit encoded URL for secure access to the order
    14. documentName - The type or title of the document, typically located at the top of the document, such as "Certification", "Recertification", "Orders" (Can be any order), "485", "Plan of Care", etc.

    Instructions:
    - Return a single JSON object with exact field names and order.
    - startOfEpisode: Match labels like "Start Date", "Episode Start Date", "Certification period From", "From Date", or the date before a dash (e.g., "04/01/2025 - 05/30/2025").
    - endOfEpisode: Match "End Date", "Episode End Date", "Certification period To", "To Date", or the date after the dash.
    - The start and end of episode should be exactly 59 or 89 days apart (as seen in the source).
    - Use MM/DD/YYYY format. Pad single digits with 0 (e.g., "4/6/2025" → "04/06/2025").
    - Return missing fields as "" (empty string) or null, not None.
    - All booleans must be true or false, not strings.
    - Do not infer values; only extract what's explicitly present.
    """
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content([text, query])
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json.loads(match.group()) if match else {}

def fetch_signed_date(doc_id):
    response = requests.get(f"{DOC_STATUS_URL}{doc_id}", headers=HEADERS)
    if response.status_code == 200:
        value = response.json().get("value", {})
        if value.get("documentStatus") == "Signed":
            raw_date = value.get("physicianSigndate", "")
            try:
                return datetime.fromisoformat(raw_date).strftime("%m/%d/%Y") if raw_date else None
            except ValueError:
                return None
    return None

def get_patient_details_from_api(patient_id):
    url = f"https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/get-patient/{patient_id}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"⚠ Failed to fetch patient details for patient ID: {patient_id}")
        return {}


def process_dates_for_patient(patient_data, doc_id, audit_reason=None):
    try:
        episode_info = patient_data.get("episodeDiagnoses", [{}])[0] or {}
    except:
        with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, "No Episode Diagnosis"])
        return None

    def safe_parse_date(date_str):
        try:
            date_str = str(date_str).strip()
            if not date_str:
                return None
            return datetime.strptime(date_str, "%m/%d/%Y")
        except Exception:
            return None

    def format_date(dt):
        return dt.strftime("%m/%d/%Y") if dt else ""

    soc = episode_info.get("startOfCare", "")
    soe = episode_info.get("startOfEpisode", "")
    eoe = episode_info.get("endOfEpisode", "")

    soc_dt = safe_parse_date(soc)
    soe_dt = safe_parse_date(soe)
    eoe_dt = safe_parse_date(eoe)

    print(f"\n\nParsed Dates - SOC: {soc_dt}, SOE: {soe_dt}, EOE: {eoe_dt}\n")

    # Fill missing SOC
    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    # Fill missing SOE
    if not soe_dt and eoe_dt:
        soe_dt = eoe_dt - timedelta(days=59)

    # Fill missing EOE
    if not eoe_dt and soe_dt:
        eoe_dt = soe_dt + timedelta(days=59)

    # If only SOC exists
    if not soe_dt and not eoe_dt and soc_dt:
        soe_dt = soc_dt
        eoe_dt = soe_dt + timedelta(days=59)

    # If still none of the dates are available
    if not soc_dt and not soe_dt and not eoe_dt:
        with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, "Missing SOC, SOE, EOE"])
        return None

    # Ensure all dates are formatted correctly at the end
    episode_info["startOfCare"] = format_date(soc_dt)
    episode_info["startOfEpisode"] = format_date(soe_dt)
    episode_info["endOfEpisode"] = format_date(eoe_dt)

    patient_data["episodeDiagnoses"][0] = episode_info
    return patient_data

from datetime import datetime, timedelta

def process_dates_for_order(order_data, patient_id):
    def parse_date(d):
        try:
            return datetime.strptime(d, "%m/%d/%Y") if d else None
        except ValueError:
            return None

    def format_date(d):
        return d.strftime("%m/%d/%Y") if d else ""

    # Extract existing order dates
    soc = order_data.get("startOfCare")
    soe = order_data.get("episodeStartDate")
    eoe = order_data.get("episodeEndDate")

    soc_dt = parse_date(soc)
    soe_dt = parse_date(soe)
    eoe_dt = parse_date(eoe)

    patient_details = get_patient_details_from_api(patient_id)
    agency_info = patient_details.get("agencyInfo", {}) if patient_details else {}

    if not soc_dt and not soe_dt and not eoe_dt:
        soc_dt = parse_date(agency_info.get("startOfCare"))
        soe_dt = parse_date(agency_info.get("startOfEpisode"))
        eoe_dt = parse_date(agency_info.get("endOfEpisode"))

    if not soe_dt and not eoe_dt:
        print("SOC:",agency_info.get("startOfCare") )
        soe_dt = parse_date(agency_info.get("startOfCare"))
        eoe_dt = soe_dt + timedelta(days=59)
        print("SOE:",soe_dt, "  EOE:",eoe_dt)
    
    
    
    if not soc_dt and soe_dt:
        soc_dt = soe_dt

    # Case 2: Compute missing SOE from EOE
    if not soe_dt and eoe_dt:
        soe_dt = eoe_dt - timedelta(days=59)

    # Case 3: Compute missing EOE from SOE
    if not eoe_dt and soe_dt:
        eoe_dt = soe_dt + timedelta(days=59)
        
    if not soc_dt and eoe_dt:
        soc_dt = soe_dt

    order_data["startOfCare"] = format_date(soc_dt)
    order_data["episodeStartDate"] = format_date(soe_dt)
    order_data["episodeEndDate"] = format_date(eoe_dt)

    return order_data


def check_if_patient_exists(fname, lname, dob):
    url = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/company/pg/4b51c8b7-c8c4-4779-808c-038c057f026b"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print("⚠ Failed to fetch patient list.")
        return None

    try:
        patients = response.json()
    except Exception as e:
        print("⚠ Error parsing patient list JSON:", e)
        return None

    dob = dob.strip()
    fname = fname.strip().upper()
    lname = lname.strip().upper()

    for p in patients:
        info = p.get("agencyInfo", {})
        if not info:
            continue
        existing_fname = (info.get("patientFName") or "").strip().upper()
        existing_lname = (info.get("patientLName") or "").strip().upper()
        existing_dob = (info.get("dob") or "").strip()
        if existing_fname == fname and existing_lname == lname and existing_dob == dob:
            patient_id = p.get("id") or p.get("patientId")
            print(f"\n✅ Patient exists: {fname} {lname}, DOB: {dob}, ID: {patient_id}")
            return patient_id
    return None


def get_or_create_patient(patient_data, daId, agency):
    print("Patient Data:", patient_data)
    dob = patient_data.get("dob", "").strip()
    fname = patient_data.get("patientFName", "").strip().upper()
    lname = patient_data.get("patientLName", "").strip().upper()

    key = f"{fname}_{lname}_{dob}"

    # Check locally created patients
    if key in created_patients:
        print(f"\n🔁 Patient already created earlier in this run: {key}")
        return created_patients[key]

    # Check if patient exists in API
    existing_id = check_if_patient_exists(fname, lname, dob)
    if existing_id:
        print(f"\n🔁 Patient exists on platform: {key}, ID: {existing_id}")
        created_patients[key] = existing_id  # Cache it to avoid refetching
        with open("hawthorn_internalmedicine.json", "w") as f:
            json.dump(created_patients, f, indent=2)
        return existing_id

    # Proceed to create new patient
    patient_data["daBackofficeID"] = str(daId)
    patient_data["pgCompanyId"] = PG_ID
    patient_data["companyId"] = company_map.get(agency.strip().lower())
    patient_data["physicianGroup"] = "Hawthorn Medical Associates - Internal Medicine"
    patient_data["physicianGroupNPI"] = "1659649853"
    print(f"\n\nPatient JSON for creating patient: {patient_data}\n\n")

    resp = requests.post(PATIENT_CREATE_URL, headers={"Content-Type": "application/json"}, json=patient_data)
    print("status code: ", resp.status_code)
    print(f"Patient details : {resp.text}")

    if resp.status_code == 201:
        new_id = resp.json().get("id") or resp.text
        created_patients[key] = new_id
        with open("hawthorn_internalmedicine.json", "w") as f:
            json.dump(created_patients, f, indent=2)
        return new_id

    return None

# def push_order(order_data, doc_id):
#     print(f"Req body: {order_data}\n\n")
#     if(order_data["orderNo"]==None):
#         order_data["orderNo"] = doc_id + "1"
#     resp = requests.post(ORDER_PUSH_URL, headers={"Content-Type": "application/json"}, json=order_data)
#     new_resp = resp.text
#     print(new_resp["orderID"])
#     if resp.status_code != 201:
#         with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
#             writer = csv.writer(file)
#             writer.writerow([doc_id, "Duplicate Order",new_resp["orderID"]])
#         print(resp.text)
#     return resp.text

def push_order(order_data, doc_id):
    print(f"Req body: {order_data}\n\n")
    
    if order_data["orderNo"] is None:
        order_data["orderNo"] = doc_id + "1"
    
    resp = requests.post(ORDER_PUSH_URL, headers={"Content-Type": "application/json"}, json=order_data)
    
    try:
        new_resp = resp.json()  # Parse the response to a dict
    except json.JSONDecodeError:
        print("Failed to decode JSON response:")
        print(resp.text)
        return resp.text
    
    order_id = new_resp.get("orderId")
    print(f"Order ID: {order_id}")
    
    if resp.status_code != 201:
        with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doc_id, "Duplicate Order", order_id])
        print(resp.text)

    return resp.text


def remove_none_fields(data):
    required_date_fields = {'orderDate', 'startOfCare', 'episodeStartDate', 'episodeEndDate'}
    if isinstance(data, dict):
        return {
            k: remove_none_fields(v)
            for k, v in data.items()
            if v is not None or k in required_date_fields
        }
    elif isinstance(data, list):
        return [remove_none_fields(item) for item in data]
    elif data is None:
        return ""

# def get_company_id_from_agency(agency, company_map):
#     agency_lower = agency.lower()
#     for key in company_map:
#         if agency_lower in key.lower():
#             return company_map[key]
#     return None  # Return None if no match is found

def process_csv(csv_path):
    
    def normalize_date_string(date_str):
        try:
            # Replace hyphens with slashes if needed
            date_str = date_str.replace("-", "/").strip()
            # Parse using flexible format
            dt = datetime.strptime(date_str, "%m/%d/%Y")
            # Format with leading zeroes
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return ""
    i = 0
    with open(csv_path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            # if(i==0):
            #     i+=1
            #     continue
            if i > 3:
                break
            i += 1
            # print(row)
            doc_id = row["ID"]
            agency = row["Facility"].strip()
            received = normalize_date_string(row["Received On"].strip())
            print(f"\n Processing Document ID: {doc_id} for Agency: {agency} date : {received}")

            try:
                res = get_pdf_text(doc_id)
                text = res[0]
                if not text.strip():
                    raise ValueError("Empty text extracted from PDF")
            except Exception as e:
                print(f"❌ Failed to extract text for Doc ID {doc_id}: {str(e)}")
                with open(AUDIT_PATIENTS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Could not extract text from PDF"])
                continue

            patient_data = extract_patient_data(text)
            print(f"\n\nResponse from gemini for patient: {patient_data}\n\n")
            # patient_data = remove_none_fields(p_data)
            patient_data = process_dates_for_patient(patient_data, doc_id)
            print(f"\n\nPatient data after setting dates : {patient_data}\n\n")

            if not patient_data:
                print(f"❌ Skipping patient creation for Doc ID {doc_id} due to insufficient date info.")
                continue

            patient_id = get_or_create_patient(patient_data, res[1], agency)
            print(f"Patient Created : \n {patient_id}\n\n")

            order_data = extract_order_data(text)
            # order_data = remove_none_fields(o_data)
            order_data["companyId"] = company_map.get(agency.lower())
            # order_data["companyId"] = "370583e3-01c8-477e-8b6d-d3537a10c767"
            
            if order_data["companyId"] == "" or order_data["companyId"] == None:
                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Missing episodeStartDate"])
                print(f"❌ Skipping order push for Doc ID {doc_id} due to missing companyID")
                continue
                
            order_data["pgCompanyId"] = PG_ID
            order_data["patientId"] = patient_id
            order_data["documentID"] = doc_id
            order_data["sentToPhysicianDate"] = received
            order_data["signedByPhysicianDate"] = fetch_signed_date(doc_id)
            
            
            
            # if not order_data.get("startOfCare") or not order_data.get("episodeStartDate") or not order_data.get("episodeEndDate"):
            #     patient_details = get_patient_details_from_api(patient_id)
            #     agency_info = patient_details.get("agencyInfo", {}) if patient_details else {}

            #     if not order_data.get("startOfCare") and agency_info.get("startOfCare"):
            #         order_data["startOfCare"] = agency_info["startOfCare"]

            #     if not order_data.get("episodeStartDate") and agency_info.get("startOfEpisode"):
            #         order_data["episodeStartDate"] = agency_info["startOfEpisode"]

            #     if not order_data.get("episodeEndDate") and agency_info.get("endOfEpisode"):
            #         order_data["episodeEndDate"] = agency_info["endOfEpisode"]

            order_data = process_dates_for_order(order_data,patient_id)
            if not order_data.get("orderDate"):
                order_data["orderDate"] = received
            elif not order_data.get("episodeStartDate"):
                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Missing episodeStartDate"])
                print(f"❌ Skipping order push for Doc ID {doc_id} due to missing episodeStartDate")
                continue
            elif not order_data.get("episodeEndDate"):
                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Missing episodeEndDate"])
                print(f"❌ Skipping order push for Doc ID {doc_id} due to missing episodeEndDate")
                continue
            elif not order_data.get("startOfCare"):
                with open(AUDIT_ORDERS_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([doc_id, "Missing startOfCare"])
                print(f"❌ Skipping order push for Doc ID {doc_id} due to missing startOfCare")
                continue

            print(f"📦 Pushing Order for Patient ID: {patient_id}")
            result = push_order(order_data, doc_id)
            print("✅ Order Push Response:", result)

# Call the main function
process_csv("orders_4.csv")