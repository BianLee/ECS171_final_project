import requests
import json
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def query_epochs_data(limit=1000):
    url = os.getenv('API_URL')
    
    headers = {
        "Content-Type": "application/json"
    }
    
    query_template = """
    {
      rounds(first: 1000, skip: SKIP_VALUE, orderBy: epoch, orderDirection: desc) {
        id
        epoch
        position
        failed
        lockPrice
        closePrice
        totalAmount
        bullAmount
        bearAmount
      }
    }
    """
    
    all_rounds = []
    skip = 0
    
    while len(all_rounds) < limit:
        query = query_template.replace("SKIP_VALUE", str(skip))
        payload = {
            "query": query,
            "variables": {}
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()
            
            if 'data' in data and 'rounds' in data['data']:
                rounds = data['data']['rounds']
                if not rounds:
                    break
                all_rounds.extend(rounds)
                skip += len(rounds)
            else:
                print("No more rounds data found.")
                break
            
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            break
    
    return all_rounds[:limit]

def save_to_csv(data, filename="epochs_data.csv"):
    if not data:
        print("No data to save.")
        return
    
    keys = data[0].keys()
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    limit = 1000  # Limit to the last 1000 epochs
    epochs_data = query_epochs_data(limit=limit)
    
    if epochs_data:
        save_to_csv(epochs_data)