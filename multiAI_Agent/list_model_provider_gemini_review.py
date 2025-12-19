import os
import time
import sys
import json
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
from langchain_core.prompts import PromptTemplate

# --- UTILS ---
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ==============================================================================
# 1. THE INTELLIGENCE ENGINE (POWERED BY GEMINI 2.5 FLASH)
# ==============================================================================

def get_gemini_intel(target_model_id: str) -> Dict[str, str]:
    """
    Calls Gemini 2.5 Flash to generate technical specs for the target model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"arch": "N/A", "ctx": "N/A", "best": "No Google Key"}

    try:
        # Initialize the specific "Intel" Agent
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
            max_retries=1
        )

        # Strict JSON prompt for consistency
        template = """
        You are a technical database. I will give you a Model ID.
        Return ONLY a JSON object with these exact keys describing the model. 
        Keep values extremely short (abbreviated).
        
        Model ID: {model_id}
        
        Required JSON Format:
        {{
            "arch": "Architecture (e.g., 'MoE Trans.', 'Dense', 'Diffusion') - Max 12 chars",
            "ctx": "Context Window (e.g., '128k', '1M', '200k') - Max 6 chars",
            "best": "Best For (e.g., 'Coding', 'Reasoning', 'Fast Chat') - Max 15 chars"
        }}
        """
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        
        # Invoke Gemini 2.5 Flash
        response = chain.invoke({"model_id": target_model_id})
        
        # Clean and Parse JSON
        content = response.content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        return data

    except Exception as e:
        # Fallback if Gemini 2.5 Flash is busy or fails
        return {"arch": "Unknown", "ctx": "?", "best": "Analysis Failed"}

# ==============================================================================
# 2. THE FETCHERS (Standard Scanners)
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
                # Prioritize keeping only relevant, modern models
                if any(x in name for x in ["gemini-2", "gemini-3", "flash", "pro", "exp"]):
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
        blacklist = ["dall-e", "tts", "whisper", "embedding", "moderation"]
        for m in all_ids:
            if any(b in m for b in blacklist): continue
            if m.startswith(("gpt", "o1", "o3")):
                survivable.append(m)
        return sorted(survivable, reverse=True)[:10] # Limit to top 10 to save API calls
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
        models = api.list_models(filter="text-generation", sort="downloads", direction=-1, limit=5)
        return [m.modelId for m in models]
    except: return []

def fetch_perplexity_expert() -> List[str]:
    print(f"{Fore.CYAN}Loading Perplexity Sonar Grid...{Style.RESET_ALL}")
    if not os.getenv("PPLX_API_KEY"): return []
    return ["sonar-reasoning-pro", "sonar-pro", "sonar", "r1-1776"]

# ==============================================================================
# 3. THE INTERACTIVE ENGINE
# ==============================================================================

class InteractiveSurvivalKit:
    def __init__(self):
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
        # 1. CALL GEMINI 2.5 FLASH FOR INTEL FIRST
        # Note: This makes the scan slower but smarter.
        intel = get_gemini_intel(model)
        
        # 2. PING TEST
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
            
            return {**intel, "p": provider, "m": model, "s": "ONLINE", "l": lat}
            
        except Exception as e:
            err = str(e).lower()
            status = "ERROR"
            if "quota" in err or "429" in err: status = "RATE LIMIT"
            elif "vision" in err: status = "VISION ONLY"
            elif "not found" in err: status = "NOT FOUND"
            
            return {**intel, "p": provider, "m": model, "s": status, "l": 999}

    def run(self):
        while True:
            zoo = self._get_user_selection()
            if not zoo: continue

            total_models = sum(len(v) for v in zoo.values())
            print(f"\n{Fore.CYAN}--- ANALYZING {total_models} MODELS WITH GEMINI 2.5 FLASH ---{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}(This may take a moment as Gemini analyzes each model...){Style.RESET_ALL}")

            tasks = []
            # Reduced workers to prevent hitting Gemini Flash rate limits while asking for intel
            with ThreadPoolExecutor(max_workers=5) as executor:
                for p, models in zoo.items():
                    for m in models:
                        tasks.append(executor.submit(self._ping, p, m))
            
            results = []
            for f in as_completed(tasks):
                results.append(f.result())
            
            # Sort: Provider -> Status -> Architecture
            results.sort(key=lambda x: (x['p'], x['s'] != "ONLINE", x['m']))
            
            # --- NEW TABLE LAYOUT ---
            # Added: CONTEXT (CTX) and BEST FOR columns
            print(f"\n{'PROVIDER':<12} | {'MODEL ID':<24} | {'STATUS':<10} | {'LATENCY':<7} | {'ARCH':<14} | {'CTX':<8} | {'BEST FOR'}")
            print("-" * 115)
            
            for r in results:
                color = Fore.GREEN if r['s'] == "ONLINE" else Fore.RED
                if r['s'] == "RATE LIMIT": color = Fore.YELLOW
                if r['s'] == "VISION ONLY": color = Fore.BLUE
                
                mid = (r['m'][:22] + '..') if len(r['m']) > 24 else r['m']
                lat = f"{r['l']}s" if r['l'] != 999 else "-"
                
                print(f"{r['p']:<12} | {mid:<24} | {color}{r['s']:<10}{Style.RESET_ALL} | {lat:<7} | {Style.DIM}{r['arch']:<14}{Style.RESET_ALL} | {r['ctx']:<8} | {r['best']}")
            
            print("\n")

if __name__ == "__main__":
    kit = InteractiveSurvivalKit()
    kit.run()