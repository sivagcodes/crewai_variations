import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the client pointing to Perplexity's servers
client = OpenAI(
    api_key=os.getenv("PPLX_API_KEY"),  # Ensure your .env has PPLX_API_KEY=pplx-...
    base_url="https://api.perplexity.ai"
)

print("Listing available Perplexity models:\n")

try:
    # Perplexity supports the standard models.list() endpoint
    for model in client.models.list():
        # Note: Perplexity models don't have "supported_actions" or "display_name" attributes
        # like Google. They stick to the standard OpenAI 'id', 'created', 'owned_by' format.
        
        print(f"Model ID: {model.id}")
        print(f"Owner:    {model.owned_by}")
        print(f"Created:  {model.created}")
        print("-" * 30)

except Exception as e:
    print(f"Error fetching Perplexity models: {e}")