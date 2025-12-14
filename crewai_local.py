# test_ollama.py
import requests
import json

def query_ollama(prompt, model="tinyllama"):
    """
    Query a local Ollama model and return the response
    
    Requirements:
    - Ollama server must be running in background (see steps below)
    - The model specified must be pulled locally using `ollama pull`
    """
    url = "http://:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            url,
            headers=headers, 
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        return f"Error: {str(e)}"

# Test with a sample question
if __name__ == "__main__":
    print("Testing Ollama API...\n")
    result = query_ollama("Define CrewAI in one sentence.")
    print("Response:")
    print("=" * 40)
    print(result)
    print("=" * 40)