import os
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate # pip install tabulate

# Load environment variables
load_dotenv()

def get_model_inventory():
    try:
        client = OpenAI()
        models = client.models.list()
        
        inventory = []
        
        # Define categories for sorting
        categories = {
            "REASONING (The Thinkers)": ["o1", "o3"],
            "FLAGSHIP (The Brains)": ["gpt-4o", "gpt-4-turbo", "gpt-4"],
            "ECONOMY (The Speedsters)": ["gpt-4o-mini", "gpt-3.5"],
            "SPECIALIZED (Audio/Image)": ["dall-e", "tts", "whisper", "audio"],
            "LEGACY/OTHER": []
        }

        print(f"Fetching models for API Key ending in ...{os.getenv('OPENAI_API_KEY')[-4:]}...\n")

        for m in models:
            # Convert timestamp to readable date
            created_date = datetime.datetime.fromtimestamp(m.created).strftime('%Y-%m-%d')
            
            # Categorize
            category = "LEGACY/OTHER"
            for cat, keywords in categories.items():
                if cat == "LEGACY/OTHER": continue
                if any(k in m.id for k in keywords):
                    category = cat
                    break
            
            inventory.append({
                "ID": m.id,
                "Created": created_date,
                "Owner": m.owned_by,
                "Category": category
            })

        # Sort by Category then Date (newest first)
        inventory.sort(key=lambda x: (x['Category'], x['Created']), reverse=True)
        
        return inventory

    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    data = get_model_inventory()
    
    # Print as a clean table
    print(tabulate(data, headers="keys", tablefmt="simple_grid"))
    
    print(f"\nTotal Models Available: {len(data)}")