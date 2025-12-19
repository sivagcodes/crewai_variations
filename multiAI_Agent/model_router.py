import os
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from colorama import Fore, Style, init

# --- LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
init(autoreset=True)

class DynamicRouter:
    # 1. STATIC CHAMPIONS LIST (The "Fast Track")
    # We try these specific models first because we know they are good.
    CHAMPIONS = {
        "SMART": [
            {"p": "OPENAI", "m": "gpt-4o"},
            {"p": "GOOGLE", "m": "gemini-1.5-pro"},
            {"p": "PERPLEXITY", "m": "sonar-reasoning-pro"},
            {"p": "GROQ", "m": "llama-3.3-70b-versatile"}, # Surprisingly smart
        ],
        "FAST": [
            {"p": "GROQ", "m": "llama-3.1-8b-instant"},
            {"p": "GOOGLE", "m": "gemini-2.5-flash"},
            {"p": "OPENAI", "m": "gpt-4o-mini"},
        ]
    }

    # 2. CACHE STORAGE
    _cache = {
        "SMART": {"data": None, "time": 0},
        "FAST": {"data": None, "time": 0}
    }
    CACHE_DURATION = 3600  # 1 Hour

    def __init__(self):
        self.provider_map = {
            "GOOGLE": "google_genai",
            "OPENAI": "openai",
            "GROQ": "groq",
            "PERPLEXITY": "openai",
            "HUGGINGFACE": "huggingface"
        }

    def _create_pinger(self, provider, model):
        """Creates a lightweight instance just for pinging."""
        try:
            if provider == "GOOGLE":
                return ChatGoogleGenerativeAI(model=model, request_timeout=3, max_retries=1)
            elif provider == "GROQ":
                return ChatGroq(model=model, max_retries=1)
            elif provider == "OPENAI":
                return ChatOpenAI(model=model, request_timeout=3, max_retries=1)
            elif provider == "PERPLEXITY":
                return ChatOpenAI(
                    model=model, 
                    api_key=os.getenv("PPLX_API_KEY"), 
                    base_url="https://api.perplexity.ai",
                    request_timeout=3, max_retries=1
                )
            elif provider == "HUGGINGFACE":
                return HuggingFaceEndpoint(repo_id=model, timeout=3)
        except:
            return None

    def _ping_candidate(self, candidate: Dict):
        """Pings a single candidate model."""
        p, m = candidate["p"], candidate["m"]
        
        # Skip if key is missing (Don't waste time trying)
        key_name = f"{p}_API_KEY" if p != "PERPLEXITY" else "PPLX_API_KEY"
        if p != "HUGGINGFACE" and not os.getenv(key_name):
            return None

        try:
            start = time.time()
            llm = self._create_pinger(p, m)
            if llm:
                llm.invoke("Hi") # The Ping
                lat = time.time() - start
                return {
                    "provider": self.provider_map[p],
                    "model": m,
                    "lat": lat,
                    "base_url": "https://api.perplexity.ai" if p == "PERPLEXITY" else None,
                    "api_key": os.getenv(key_name)
                }
        except:
            pass
        return None

    def get_best_model(self, strategy="SMART", force_refresh=False) -> Dict:
        # STEP 1: CHECK CACHE
        now = time.time()
        cached = self._cache[strategy]
        if not force_refresh and cached["data"] and (now - cached["time"] < self.CACHE_DURATION):
            # Return cached result instantly
            # print(f"{Fore.GREEN}>> Cache Hit ({strategy}){Style.RESET_ALL}") 
            return cached["data"]

        print(f"{Fore.CYAN}--- ROUTER: Racing '{strategy}' Champions... ---{Style.RESET_ALL}")
        
        # STEP 2: LOAD CANDIDATES (The Fast Track)
        candidates = self.CHAMPIONS[strategy]
        
        results = []
        # STEP 3: PARALLEL PING (Max 5 threads is enough for champions)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(self._ping_candidate, c) for c in candidates]
            for f in as_completed(futures):
                if res := f.result():
                    results.append(res)

        # STEP 4: DECIDE WINNER
        if not results:
            print(f"{Fore.RED} All champions failed. Defaulting to OpenAI.{Style.RESET_ALL}")
            return {"provider": "openai", "model": "gpt-4o"}

        # Sort by latency (lowest first)
        winner = sorted(results, key=lambda x: x['lat'])[0]
        
        print(f"{Fore.YELLOW} > Winner: {winner['provider']}/{winner['model']} ({winner['lat']:.2f}s){Style.RESET_ALL}")

        # STEP 5: UPDATE CACHE
        self._cache[strategy] = {"data": winner, "time": now}
        
        return winner

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    router = DynamicRouter()
    
    # 1. First Call: Takes ~1 second (Networking)
    start = time.time()
    model_config = router.get_best_model("SMART")
    print(f"Total Logic Time: {time.time() - start:.2f}s\n")

    # 2. Second Call: Takes 0.00 seconds (Cache)
    start = time.time()
    model_config = router.get_best_model("SMART")
    print(f"Total Logic Time: {time.time() - start:.2f}s")