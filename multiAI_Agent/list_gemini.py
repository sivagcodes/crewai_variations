from google import genai
import os

# Initialize the client
# If you have GOOGLE_API_KEY set in your environment, you can just use Client()
#client = genai.Client(api_key="AIzaSyACBb6s6n7cAHjzS9Ne3i2g6WWytV0ON4Y") #Pro
client = genai.Client(api_key="") #Pro
print("Listing available Gemini models:\n")

# Use the models.list() method
for model in client.models.list():
    # Filter for models that support generating content
    if "generateContent" in model.supported_actions:
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print("-" * 30)