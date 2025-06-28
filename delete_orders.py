import os
import requests
import argparse

API_BASE = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net"
ORDER_DELETE_ROUTE = "/api/Order/{id}"

def delete_order(order_id, token):
    url = API_BASE + ORDER_DELETE_ROUTE.format(id=order_id)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(url, headers=headers)
    if response.status_code in (200, 204):
        print(f"Order {order_id} deleted successfully.")
    else:
        print(f"Failed to delete order {order_id}. Status: {response.status_code}, Response: {response.text}")

def parse_id_list(id_str):
    if not id_str:
        return []
    ids = [i.strip() for i in id_str.replace('\n', ',').replace(' ', ',').split(',') if i.strip()]
    return ids

def main():
    parser = argparse.ArgumentParser(description="Delete orders by ID (batch supported).")
    parser.add_argument('--order_ids', type=str, help='Comma or whitespace separated list of Order IDs to delete')
    args = parser.parse_args()

    token = os.getenv('AUTH_TOKEN')
    if not token:
        print("AUTH_TOKEN environment variable not set.")
        return

    order_ids = parse_id_list(args.order_ids)

    if order_ids:
        for oid in order_ids:
            delete_order(oid, token)
    else:
        print("No order_ids provided. Use --order_ids.")

if __name__ == "__main__":
    main() 