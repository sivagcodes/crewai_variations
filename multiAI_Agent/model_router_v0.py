import os
import time
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- SDK IMPORTS ---
from google import genai
from groq import Groq
from huggingface_hub import HfApi
from openai import OpenAI

# --- LANGCHAIN IMPORTS ---
# Note: Ensure you have installed: langchain-google-genai, langchain-groq, langchain-openai, langchain-huggingface
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ==============================================================================
# 1. INTELLIGENCE ENGINE (Rule-Based for Speed)
# ==============================================================================

def get_model_intel(model_id: str) -> Dict[str, str]:
    """Analyzes model ID to determine Architecture and Best Use Case."""
    m = model_id.lower()
    arch = "Transformer"
    desc = "General Tasks"
    
    # Architecture Guessing
    if "moe" in m or "mixtral" in m or "qwen-max" in m: arch = "MoE (Sparse)"
    elif "o1" in m or "o3" in m or "reasoning" in m or "r1" in m: arch = "CoT / Reasoning"
    elif "flash" in m or "haiku" in m or "nano" in m: arch = "Distilled Dense"
    elif "specdec" in m: arch = "Speculative"
    elif "vision" in m or "gemini" in m or "gpt-4o" in m or "sonar" in m: arch = "Multimodal"
    
    # Best For Guessing
    if "coder" in m or "codex" in m: desc = "Coding Specialist"
    elif "reasoning" in m or "o1" in m or "o3" in m or "deep-research" in m: desc = "Deep Research/Math"
    elif "flash" in m or "instant" in m or "nano" in m: desc = "High Speed / Cost"
    elif "search" in m or "sonar" in m: desc = "Live Web RAG"
    elif "pro" in m or "max" in m or "plus" in m: desc = "Complex Production"
    
    return {"arch": arch, "desc": desc}

def fetch_google_expert() -> List[str]:
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return []
    try:
        client = OpenAI(api_key=api_key)
        all_ids = [m.id for m in client.models.list()]
        survivable = []
        blacklist = ["dall-e", "tts", "whisper", "embedding", "moderation", "davinci", "babbage"]
        for m in all_ids:
            if any(b in m for b in blacklist): continue
            if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "sora")):
                survivable.append(m)
        return sorted(survivable, reverse=True)
    except: return []

def fetch_groq_expert() -> List[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return []
    try:
        client = Groq(api_key=api_key)
        # Filter out Whisper and other non-chat models
        return sorted([m.id for m in client.models.list().data if "whisper" not in m.id])
    except: return []

def fetch_hf_expert() -> List[str]:
    try:
        api = HfApi()
        # Filter strictly for text-generation
        models = api.list_models(filter="text-generation", sort="likes", direction=-1, limit=5)
        return [m.modelId for m in models]
    except: return []

def fetch_perplexity_expert() -> List[str]:
    if not os.getenv("PPLX_API_KEY"): return []
    return ["sonar-reasoning-pro", "sonar-reasoning", "sonar-pro", "sonar", "r1-1776"]

# ==============================================================================
# 2. THE DYNAMIC ROUTER
# ==============================================================================

class DynamicRouter:
    def __init__(self):
        self.registry: Dict[str, Callable] = {
            "OPENAI": fetch_openai_expert,
            "GOOGLE": fetch_google_expert,
            "GROQ": fetch_groq_expert,
            "PERPLEXITY": fetch_perplexity_expert,
            "HUGGINGFACE": fetch_hf_expert
        }
        # Maps Internal Keys -> LangChain Provider Strings
        self.provider_map = {
            "GOOGLE": "google_genai",
            "OPENAI": "openai",
            "GROQ": "groq",
            "HUGGINGFACE": "huggingface",
            "PERPLEXITY": "openai" # Perplexity is OpenAI-compatible
        }

    def create_llm(self, provider_str: str, model: str):
        """Factory to create a lightweight pinging instance."""
        try:
            if provider_str == "google_genai":
                return ChatGoogleGenerativeAI(model=model, max_retries=1, request_timeout=5)
            
            elif provider_str == "groq":
                return ChatGroq(model=model, max_retries=1)
            
            elif provider_str == "openai":
                # Special Handling for Perplexity (Sonar/R1) routing
                if "sonar" in model.lower() or "r1-" in model.lower():
                    api_key = os.getenv("PPLX_API_KEY")
                    if not api_key: return None
                    return ChatOpenAI(
                        model=model, 
                        api_key=api_key, 
                        base_url="https://api.perplexity.ai", 
                        max_retries=1,
                        request_timeout=5
                    )
                return ChatOpenAI(model=model, max_retries=1, request_timeout=5)
            
            elif provider_str == "huggingface":
                return HuggingFaceEndpoint(repo_id=model, task="text-generation", timeout=5)
                
            return None
        except Exception:
            return None

    def _ping(self, provider, model):
        # 1. Filter out known non-chat models that crash ping tests
        model_lower = model.lower()
        if any(x in model_lower for x in ["prompt-guard", "moderation", "embedding", "vision"]):
            return None
            
        try:
            start = time.time()
            # Convert internal provider key (e.g., "GOOGLE") to LangChain string (e.g., "google_genai")
            lc_provider_str = self.provider_map.get(provider)
            if not lc_provider_str: return None

            llm = self.create_llm(lc_provider_str, model)
            
            if llm:
                # Simple ping to check latency and auth
                llm.invoke("Hi")
                latency = time.time() - start
                
                return {
                    "provider": lc_provider_str, # Return the formatted string for LangChain
                    "original_provider": provider,
                    "model": model,
                    "lat": latency,
                    "intel": get_model_intel(model)
                }
        except Exception:
            # Silently fail for models that are offline/erroring
            pass
        return None

    def get_best_model(self, strategy="SMART") -> Dict:
        print(f"{Fore.CYAN}--- ROUTER: Finding Best '{strategy}' Model... ---{Style.RESET_ALL}")
        
        candidates = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = []
            for p, fetcher in self.registry.items():
                # Check for API Key existence before even trying to fetch
                has_key = False
                if p == "HUGGINGFACE": has_key = True # HF often works without key for public endpoints
                elif p == "PERPLEXITY" and os.getenv("PPLX_API_KEY"): has_key = True
                elif os.getenv(f"{p}_API_KEY"): has_key = True
                
                if has_key:
                    models = fetcher()
                    for m in models: 
                        futures.append(ex.submit(self._ping, p, m))
            
            for f in as_completed(futures):
                if res := f.result(): 
                    candidates.append(res)

        if not candidates: 
            print(f"{Fore.RED}No online models found. Check your .env file.{Style.RESET_ALL}")
            # Fallback to a safe default if everything fails
            return {"provider": "openai", "model": "gpt-4o-mini"}

        # Selection Logic
        winner = None
        if strategy == "SMART":
            # Filter for "Production" or "Research" grade models
            smart = [c for c in candidates if c['intel']['desc'] in ["Complex Production", "Deep Research/Math"]]
            if not smart:
                # Soft Fallback: Anything that isn't purely "Speed" focused
                smart = [c for c in candidates if c['intel']['desc'] != "High Speed / Cost"]
            
            # Pick fastest of the smart ones
            winner = sorted(smart, key=lambda x: x['lat'])[0] if smart else sorted(candidates, key=lambda x: x['lat'])[0]
        else:
            # Filter for "Speed" models
            fast = [c for c in candidates if c['intel']['desc'] == "High Speed / Cost"]
            winner = sorted(fast, key=lambda x: x['lat'])[0] if fast else sorted(candidates, key=lambda x: x['lat'])[0]

        print(f"{Fore.YELLOW} > Routed to: {winner['provider']}/{winner['model']} ({winner['lat']:.2f}s){Style.RESET_ALL}")
        return {"provider": winner['provider'], "model": winner['model']}

if __name__ == "__main__":
    router = DynamicRouter()
    
    # Test Smart Routing
    smart_choice = router.get_best_model("SMART")
    
    # Test Fast Routing
    fast_choice = router.get_best_model("FAST")

    print(smart_choice, fast_choice)