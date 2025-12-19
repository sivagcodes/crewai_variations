import os
import time
import sys
from typing import Dict, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- SDK IMPORTS ---
from google import genai
from groq import Groq
from huggingface_hub import HfApi
from openai import OpenAI

# --- LANGCHAIN INTEGRATION ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

# --- UTILS ---
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ==============================================================================
# 1. INTELLIGENCE ENGINE (New Feature)
# ==============================================================================

def get_model_intel(model_id: str) -> Dict[str, str]:
    """Analyzes model ID to determine Architecture and Best Use Case."""
    m = model_id.lower()
    
    # Defaults
    arch = "Transformer"
    desc = "General Tasks"

    # --- ARCHITECTURE GUESSING ---
    if "moe" in m or "mixtral" in m or "qwen-max" in m:
        arch = "MoE (Sparse)"
    elif "o1" in m or "o3" in m or "reasoning" in m or "r1" in m:
        arch = "CoT / Reasoning"
    elif "flash" in m or "haiku" in m or "nano" in m:
        arch = "Distilled Dense"
    elif "specdec" in m:
        arch = "Speculative"
    elif "vision" in m or "gemini" in m or "gpt-4o" in m or "sonar" in m:
        arch = "Multimodal"

    # --- USE CASE MAPPING ---
    if "coder" in m or "codex" in m:
        desc = "Coding Specialist"
    elif "reasoning" in m or "o1" in m or "o3" in m or "deep-research" in m:
        desc = "Deep Research/Math"
    elif "flash" in m or "instant" in m or "nano" in m:
        desc = "High Speed / Cost"
    elif "search" in m or "sonar" in m:
        desc = "Live Web RAG"
    elif "pro" in m or "max" in m or "plus" in m:
        desc = "Complex Production"

    return {"arch": arch, "desc": desc}

# ==============================================================================
# 2. THE FETCHERS (Unchanged)
# ==============================================================================

def fetch_google_expert() -> List[str]:
    print(f"{Fore.CYAN}Contacting Google HQ...{Style.RESET_ALL}")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return []
    try:
        client = genai.Client(api_key=api_key)
        found = []
        for m in client.models.list():
            if "generateContent" in m.supported_actions:
                name = m.name.replace("models/", "")
                if any(x in name for x in ["gemini-2", "gemini-3", "gemma-3", "nano", "pro", "flash"]):
                    found.append(name)
        return sorted(list(set(found)), reverse=True)
    except: return []

def fetch_openai_expert() -> List[str]:
    print(f"{Fore.CYAN}Contacting OpenAI HQ...{Style.RESET_ALL}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return []
    try:
        client = OpenAI(api_key=api_key)
        all_ids = [m.id for m in client.models.list()]
        survivable = []
        blacklist = ["dall-e", "tts-1", "whisper", "embedding", "moderation", "davinci", "babbage"]
        for m in all_ids:
            if any(b in m for b in blacklist): continue
            if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "sora")):
                survivable.append(m)
        return sorted(survivable, reverse=True)
    except: return []

def fetch_groq_expert() -> List[str]:
    print(f"{Fore.CYAN}Contacting Groq Fast Lane...{Style.RESET_ALL}")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return []
    try:
        client = Groq(api_key=api_key)
        return sorted([m.id for m in client.models.list().data if "whisper" not in m.id])
    except: return []

def fetch_hf_expert() -> List[str]:
    print(f"{Fore.CYAN}Scanning Hugging Face Hub...{Style.RESET_ALL}")
    try:
        api = HfApi()
        models = api.list_models(filter="text-generation", sort="likes", direction=-1, limit=10)
        return [m.modelId for m in models]
    except: return []

def fetch_perplexity_expert() -> List[str]:
    print(f"{Fore.CYAN}Loading Perplexity Sonar Grid...{Style.RESET_ALL}")
    if not os.getenv("PPLX_API_KEY"): return []
    return ["sonar-reasoning-pro", "sonar-reasoning", "sonar-pro", "sonar", "r1-1776"]

# ==============================================================================
# 3. THE INTERACTIVE ENGINE (Clean Labels & New Columns)
# ==============================================================================

class InteractiveSurvivalKit:
    def __init__(self):
        # CLEAN KEYS (No brackets, as requested)
        self.registry: Dict[str, Callable] = {
            "OPENAI": fetch_openai_expert,
            "GOOGLE": fetch_google_expert,
            "GROQ": fetch_groq_expert,
            "PERPLEXITY": fetch_perplexity_expert,
            "HUGGINGFACE": fetch_hf_expert
        }

    def _get_user_selection(self) -> Dict[str, List[str]]:
        print(f"\n{Fore.GREEN}=== SELECT PROVIDER ==={Style.RESET_ALL}")
        options = list(self.registry.keys())
        
        for i, provider in enumerate(options):
            print(f"{Fore.YELLOW}[{i+1}]{Style.RESET_ALL} {provider}")
        print(f"{Fore.YELLOW}[A]{Style.RESET_ALL} TEST ALL")
        print(f"{Fore.YELLOW}[Q]{Style.RESET_ALL} QUIT")

        choice = input(f"\n{Fore.CYAN}Selection > {Style.RESET_ALL}").strip().upper()
        active_zoo = {}

        if choice == 'Q': sys.exit(0)
        elif choice == 'A':
            for name, func in self.registry.items():
                active_zoo[name] = func()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    key = options[idx]
                    active_zoo[key] = self.registry[key]()
            except: pass
        return active_zoo

    def _ping(self, provider, model):
        # 1. Get Intel First (Architecture/Desc)
        intel = get_model_intel(model)
        
        try:
            llm = None
            if "OPENAI" in provider:
                llm = ChatOpenAI(model=model, max_retries=1, request_timeout=8)
            elif "GOOGLE" in provider:
                llm = ChatGoogleGenerativeAI(model=model, max_retries=1, request_timeout=8)
            elif "GROQ" in provider:
                llm = ChatGroq(model=model, max_retries=1)
            elif "HUGGING" in provider:
                llm = HuggingFaceEndpoint(repo_id=model, task="text-generation", timeout=10)
            elif "PERPLEXITY" in provider:
                llm = ChatOpenAI(model=model, api_key=os.getenv("PPLX_API_KEY"), base_url="https://api.perplexity.ai", max_retries=1)
            
            if not llm: 
                return {**intel, "p": provider, "m": model, "s": "SKIPPED", "l": 999}

            start = time.time()
            llm.invoke("Hi")
            lat = round(time.time() - start, 2)
            
            # Return combined data
            return {**intel, "p": provider, "m": model, "s": "ONLINE", "l": lat}
            
        except Exception as e:
            err = str(e).lower()
            status = "ERROR"
            if "quota" in err or "429" in err: status = "RATE LIMIT"
            elif "vision" in err: status = "VISION ONLY"
            elif "not found" in err: status = "NOT FOUND"
            elif "401" in err: status = "AUTH FAIL"
            
            return {**intel, "p": provider, "m": model, "s": status, "l": 999}

    def run(self):
        while True:
            zoo = self._get_user_selection()
            if not zoo: continue

            print(f"\n{Fore.CYAN}--- ANALYZING & PINGING ---{Style.RESET_ALL}")
            tasks = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                for p, models in zoo.items():
                    print(f"Scanning {len(models)} models for {p}...")
                    for m in models:
                        tasks.append(executor.submit(self._ping, p, m))
            
            results = []
            for f in as_completed(tasks):
                results.append(f.result())
            
            # Sort: Provider -> Status -> Architecture
            results.sort(key=lambda x: (x['p'], x['s'] != "ONLINE", x['arch']))
            
            # --- NEW TABLE LAYOUT ---
            # Model Name Truncation Logic
            print(f"\n{'PROVIDER':<12} | {'MODEL ID':<22} | {'STATUS':<10} | {'LATENCY':<7} | {'ARCHITECTURE':<18} | {'BEST FOR'}")
            print("-" * 105)
            
            for r in results:
                color = Fore.GREEN if r['s'] == "ONLINE" else Fore.RED
                if r['s'] == "RATE LIMIT": color = Fore.YELLOW
                if r['s'] == "VISION ONLY": color = Fore.BLUE
                
                # Truncate model name if > 20 chars
                mid = (r['m'][:19] + '..') if len(r['m']) > 21 else r['m']
                
                lat = f"{r['l']}s" if r['l'] != 999 else "-"
                
                # Print merged row
                print(f"{r['p']:<12} | {mid:<22} | {color}{r['s']:<10}{Style.RESET_ALL} | {lat:<7} | {Style.DIM}{r['arch']:<18}{Style.RESET_ALL} | {r['desc']}")
            
            print("\n")

if __name__ == "__main__":
    kit = InteractiveSurvivalKit()
    kit.run()