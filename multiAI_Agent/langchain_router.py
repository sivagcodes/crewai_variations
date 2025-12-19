import os
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# --- PURE LANGCHAIN IMPORTS ---
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

# --- SPECIFIC PROVIDERS (For Intel/Scanning only) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# --- UTILS ---
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)
logging.basicConfig(level=logging.ERROR)

# ==============================================================================
# 1. INTELLIGENCE KERNEL (Gemini 2.5 Flash)
# ==============================================================================

def get_gemini_intel(target_model_id: str) -> Dict[str, str]:
    """Uses Gemini 2.5 Flash to classify a model's capabilities."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return {"type": "BALANCED"}

    try:
        # Lightweight client just for Intelligence
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
            max_retries=1
        )
        
        template = """
        Classify Model ID: '{model_id}'
        Return valid JSON only:
        {{
            "type": "FAST" (if flash, haiku, llama-7b), "SMART" (if pro, o1, claude-3.5, llama-70b), or "BALANCED"
        }}
        """
        response = llm.invoke(template.format(model_id=target_model_id))
        clean_json = response.content.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except:
        return {"type": "BALANCED"}

# ==============================================================================
# 2. THE DYNAMIC ROUTER (LangChain Compatible)
# ==============================================================================

class LangChainRouter:
    def __init__(self):
        # Maps internal IDs to LangChain 'init_chat_model' provider strings
        self.provider_map = {
            "GOOGLE": "google_genai",
            "OPENAI": "openai",
            "GROQ": "groq",
            "ANTHROPIC": "anthropic"
        }
        
        self.routes = {
            "GOOGLE": self._fetch_google,
            "OPENAI": self._fetch_openai,
            "GROQ": self._fetch_groq,
        }

    def _fetch_google(self):
        if not os.getenv("GOOGLE_API_KEY"): return []
        return ["gemini-2.0-flash-exp", "gemini-1.5-pro"]

    def _fetch_openai(self):
        if not os.getenv("OPENAI_API_KEY"): return []
        return ["gpt-4o", "gpt-4o-mini"]

    def _fetch_groq(self):
        if not os.getenv("GROQ_API_KEY"): return []
        return ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

    def ping_model(self, provider, model):
        try:
            # Latency Check
            start = time.time()
            # We use specific classes here just for the ping
            if provider == "GOOGLE":
                ChatGoogleGenerativeAI(model=model).invoke("Hi")
            elif provider == "GROQ":
                ChatGroq(model=model).invoke("Hi")
            elif provider == "OPENAI":
                ChatOpenAI(model=model).invoke("Hi")
            latency = time.time() - start
            
            # Get Intel
            intel = get_gemini_intel(model)
            
            return {
                "lc_provider": self.provider_map[provider], # <--- Crucial for init_chat_model
                "model": model,
                "latency": latency,
                "type": intel.get("type", "BALANCED")
            }
        except Exception:
            return None

    def find_best_model(self, strategy="SMART") -> Dict:
        print(f"{Fore.CYAN}--- ROUTER: Scanning Neural Pathways (Strategy: {strategy}) ---{Style.RESET_ALL}")
        
        candidates = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = []
            for prov, fetcher in self.routes.items():
                models = fetcher()
                for m in models:
                    futures.append(ex.submit(self.ping_model, prov, m))
            
            for f in as_completed(futures):
                res = f.result()
                if res:
                    candidates.append(res)
                    print(f" > Found: {res['lc_provider']:<12} | {res['model']:<25} | {res['type']:<5} | {res['latency']:.2f}s")

        if not candidates:
            print(f"{Fore.RED}No models found. Defaulting to OpenAI.{Style.RESET_ALL}")
            return {"lc_provider": "openai", "model": "gpt-4o"}

        # Selection Logic
        if strategy == "SMART":
            smart_ones = [c for c in candidates if c['type'] == "SMART"]
            winner = sorted(smart_ones, key=lambda x: x['latency'])[0] if smart_ones else candidates[0]
        else: # FAST
            winner = sorted(candidates, key=lambda x: x['latency'])[0]

        print(f"{Fore.GREEN}>>> WINNER: {winner['lc_provider']} / {winner['model']}{Style.RESET_ALL}\n")
        return winner

# ==============================================================================
# 3. PURE LANGCHAIN EXECUTION
# ==============================================================================

def run_dynamic_chain():
    # 1. ROUTING
    router = LangChainRouter()
    # Let's say we want a "SMART" model for complex logic
    best_config = router.find_best_model(strategy="SMART")
    
    # 2. DYNAMIC INSTANTIATION (Pure LangChain)
    # init_chat_model automatically picks ChatGroq, ChatOpenAI, or ChatGoogleGenerativeAI
    model = init_chat_model(
        model=best_config["model"],
        model_provider=best_config["lc_provider"],
        temperature=0.7
    )
    
    print(f"{Fore.YELLOW}Initialized Runnable: {type(model).__name__}{Style.RESET_ALL}")

    # 3. STANDARD LANGCHAIN PIPELINE
    prompt = ChatPromptTemplate.from_template(
        "You are {model_name} running on {provider}. Explain your architecture briefly."
    )
    
    chain = prompt | model | StrOutputParser()

    # 4. EXECUTION
    print(f"\n{Fore.MAGENTA}--- INVOKING CHAIN ---{Style.RESET_ALL}")
    
    # We pass the dynamic metadata into the prompt just for demonstration
    response = chain.invoke({
        "model_name": best_config["model"],
        "provider": best_config["lc_provider"]
    })
    
    print(response)

if __name__ == "__main__":
    run_dynamic_chain()