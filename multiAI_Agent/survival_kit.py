import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from colorama import Fore, Style, init

# --- 2025 SURVIVAL IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import ChatPerplexity
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()  # <--- This reads the .env file and sets the variables

import os
# ... rest of your imports
# Initialize colorama for nice tables
init(autoreset=True)

# --- CONFIGURATION: THE MODEL ZOO ---
# Add or remove models here as your access allows
MODEL_ZOO = {
    "OPENAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "o1-mini",  # Reasoning model
        # "gpt-3.5-turbo" # Legacy fallback
    ],
    "ANTHROPIC": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229"
    ],
    "GOOGLE": [
        "gemini-2.0-flash-exp", # Experimental
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "GROQ": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ],
    "NVIDIA_NIM": [
        "meta/llama-3.1-405b-instruct", # The Heavy Hitter
        "meta/llama-3.1-70b-instruct",
        "mistralai/mistral-large-2407",
        "nvidia/nemotron-4-340b-instruct"
    ],
    "QWEN (ALI)": [
        "qwen-max",   # Top tier
        "qwen-plus",  # Balanced
        "qwen-turbo", # Fast/Cheap
        "qwen-vl-max" # Vision capable
    ],
    "PERPLEXITY": [
        "sonar-reasoning-pro",
        "sonar-pro",
        "sonar"
    ],
    "HUGGINGFACE": [
        "HuggingFaceH4/zephyr-7b-beta",           # Very reliable, fast
        "mistralai/Mistral-7B-Instruct-v0.3",     # The standard workhorse
        "microsoft/Phi-3-mini-4k-instruct",       # Tiny, fast, smart
        "deepseek-ai/DeepSeek-V3",                # 2025 High-end open model
        "meta-llama/Meta-Llama-3-8B-Instruct"     # Standard Llama 3 on HF
    ],
    "OLLAMA (LOCAL)": [
        "llama3.2",
        "mistral",
        "qwen2.5",
        "phi4"
    ]
}

def get_llm_instance(provider: str, model: str):
    """Factory to create the specific LangChain object for the model."""
    try:
        if provider == "OPENAI":
            return ChatOpenAI(model=model, max_retries=1, request_timeout=10)
        elif provider == "ANTHROPIC":
            return ChatAnthropic(model=model, max_retries=1, default_request_timeout=10)
        elif provider == "GOOGLE":
            return ChatGoogleGenerativeAI(model=model, max_retries=1, request_timeout=10)
        elif provider == "GROQ":
            return ChatGroq(model=model, max_retries=1, request_timeout=10)
        elif provider == "NVIDIA_NIM":
            return ChatNVIDIA(model=model, max_retries=1, timeout=10)
        elif provider == "QWEN (ALI)":
            return ChatTongyi(model=model, max_retries=1) # Timeout handling varies
        elif provider == "PERPLEXITY":
            return ChatPerplexity(model=model, max_retries=1, request_timeout=10)
        elif provider == "OLLAMA (LOCAL)":
            return ChatOllama(model=model, max_retries=1, timeout=10)
        if provider == "HUGGINGFACE":
        # Specific handling for HF Serverless
            return HuggingFaceEndpoint(
                repo_id=model, 
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                timeout=10
        )
    except Exception as e:
        return None

def test_model_connectivity(provider: str, model: str) -> Dict:
    """Pings a specific model and returns metrics."""
    llm = get_llm_instance(provider, model)
    
    if not llm:
        return {
            "provider": provider, "model": model, "status": "CONFIG_ERR", 
            "latency": 0, "msg": "Client init failed"
        }

    start = time.time()
    try:
        # Use a tiny prompt to save tokens/latency
        msg = llm.invoke("Hi")
        latency = round(time.time() - start, 2)
        
        # Extract content safely
        content = str(msg.content)[:20].replace("\n", " ") if msg.content else "Empty"
        
        return {
            "provider": provider,
            "model": model,
            "status": "ONLINE",
            "latency": latency,
            "msg": content
        }
    except Exception as e:
        err = str(e)
        status = "ERROR"
        if "401" in err: status = "AUTH_FAIL"
        if "429" in err or "quota" in err.lower(): status = "RATE_LIMIT"
        if "not found" in err.lower(): status = "NOT_FOUND"
        if "connection" in err.lower(): status = "CONN_ERR"
        
        return {
            "provider": provider,
            "model": model,
            "status": status,
            "latency": 0,
            "msg": err[:40] + "..."
        }

def run_zoo_check():
    print(f"\n{Fore.CYAN}╔════════════════════════════════════════════════════════════════════════╗")
    print(f"║             ULTIMATE AI SURVIVAL KIT: MODEL ZOO DIAGNOSTICS            ║")
    print(f"╚════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

    tasks = []
    
    # Flatten the zoo into a list of tasks
    for provider, models in MODEL_ZOO.items():
        # Quick check if API key exists for the provider (optimization)
        # Note: Ollama doesn't need a key, others do.
        skip = False
        if provider == "OPENAI" and not os.getenv("OPENAI_API_KEY"): skip = True
        if provider == "ANTHROPIC" and not os.getenv("ANTHROPIC_API_KEY"): skip = True
        if provider == "GOOGLE" and not os.getenv("GOOGLE_API_KEY"): skip = True
        if provider == "GROQ" and not os.getenv("GROQ_API_KEY"): skip = True
        if provider == "NVIDIA_NIM" and not os.getenv("NVIDIA_API_KEY"): skip = True
        if provider == "QWEN (ALI)" and not os.getenv("DASHSCOPE_API_KEY"): skip = True
        if provider == "PERPLEXITY" and not os.getenv("PPLX_API_KEY"): skip = True
        
        if skip:
            print(f"{Fore.YELLOW}⚠ Skipped {provider}: Missing API Key{Style.RESET_ALL}")
            continue

        for model in models:
            tasks.append((provider, model))

    results = []
    print(f"Testing {len(tasks)} distinct models across providers...\n")
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model = {executor.submit(test_model_connectivity, p, m): (p, m) for p, m in tasks}
        
        for future in as_completed(future_to_model):
            results.append(future.result())

    # --- THE DASHBOARD ---
    # Sort by Provider then Status
    results.sort(key=lambda x: (x['provider'], x['status'] != 'ONLINE'))

    print(f"{'PROVIDER':<15} | {'MODEL':<35} | {'STATUS':<10} | {'LATENCY':<8} | {'MSG'}")
    print("-" * 95)

    online_count = 0
    
    for r in results:
        status = r['status']
        if status == "ONLINE":
            color = Fore.GREEN
            online_count += 1
        elif status == "RATE_LIMIT":
            color = Fore.YELLOW
        else:
            color = Fore.RED
            
        print(f"{r['provider']:<15} | {r['model']:<35} | {color}{status:<10}{Style.RESET_ALL} | {r['latency']:<5}s   | {r['msg']}")

    print(f"\n{Fore.CYAN}SUMMARY: {online_count}/{len(results)} models are operational.{Style.RESET_ALL}")

if __name__ == "__main__":
    run_zoo_check()