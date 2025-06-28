from dotenv import load_dotenv
load_dotenv()
import os
import requests

API_BASE = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net"
PATIENT_DELETE_ROUTE = "/api/Patient/{id}"

def delete_patient(patient_id, token):
    url = API_BASE + PATIENT_DELETE_ROUTE.format(id=patient_id)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(url, headers=headers)
    if response.status_code in (200, 204):
        print(f"Patient {patient_id} deleted successfully.")
    else:
        print(f"Failed to delete patient {patient_id}. Status: {response.status_code}, Response: {response.text}")

def main():
    # Hardcoded list of patient IDs to delete
    patient_ids = [
        "6c075ac2-2d7e-4aa5-827c-1dcf61ffe4c4",
        "44ad5335-1e57-48e1-8173-661de34ec866",
        "e00eabc9-4094-4738-9a87-93f8b51a6a5f",
        "c58aa2b1-3015-4eac-a6d1-755f9cbd1afb",
        "505232c7-3d7c-4e8e-8b4c-9688aca79e1c",
        "78e64c9b-e446-454d-86b6-df22b5d3cc42",
        "773fa9f7-72cb-40d6-84ca-80c3537cc552",
        "07aee368-6f1b-4d04-a4cb-f0dfaa439ffd",
        "4d8d8839-63d2-4128-aee9-74eb44d9f016",
        "97df30d4-58c5-4c20-a8e3-59237262facb",
        "4dce3c6d-fba8-4e6c-8746-c139f8a71b7b",
        "92262bed-3a1e-4814-9394-d6bdc3542f82",
        "c0f9735b-15ad-40e2-b325-778daa56e0ea",
        "ca072aac-74b0-4073-a7e9-e422b0b40695",
        "eae0f388-d8df-4388-b64e-4ff470fee456",
        "70273fbb-d724-4092-9734-d098d1dc804b",
        "b066cd8e-70eb-4f60-9973-603557c9fdce",
        "303f1331-d972-4272-87f3-c96b7ab4f5d1",
        "675ea0dd-7ee9-4bf2-b9b4-299e4bf331dc"
    ]

    token = os.getenv('AUTH_TOKEN')
    if not token:
        print("AUTH_TOKEN environment variable not set.")
        return

    for pid in patient_ids:
        delete_patient(pid, token)

if __name__ == "__main__":
    main() 