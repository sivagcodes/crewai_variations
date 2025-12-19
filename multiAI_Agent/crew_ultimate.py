## This code will not work as crewai requires liteLLM and LLM to call modesl

import os
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# --- CREWAI IMPORTS ---
from crewai import Agent, Task, Crew, Process, LLM  # <--- The Smart Wrapper

# --- LANGCHAIN (Only needed for the Scanner/Intel phase) ---
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
# 1. INTELLIGENCE KERNEL (Using Gemini 2.5 Flash for Decisions)
# ==============================================================================

def get_gemini_intel(target_model_id: str) -> Dict[str, str]:
    """Uses Gemini 2.5 Flash to classify a model's capabilities."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return {"type": "BALANCED"}

    try:
        # We still use LangChain here just for the lightweight 'Intel' check
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
# 2. THE DYNAMIC ROUTER
# ==============================================================================

class ModelRouter:
    def __init__(self):
        # Maps internal provider ID to the Prefix required by CrewAI/LiteLLM
        self.prefixes = {
            "GOOGLE": "gemini",
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
            # Quick Latency Check
            start = time.time()
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
                "provider": provider,
                "prefix": self.prefixes[provider], # <--- The Magic String
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
                    print(f" > Found: {res['provider']:<7} | {res['model']:<25} | {res['type']:<5} | {res['latency']:.2f}s")

        if not candidates:
            # Fallback to OpenAI if scan fails (Default behavior you noted)
            print(f"{Fore.RED}No models found active. Defaulting to OpenAI.{Style.RESET_ALL}")
            return {"provider": "OPENAI", "prefix": "openai", "model": "gpt-4o"}

        # Selection Logic
        if strategy == "SMART":
            smart_ones = [c for c in candidates if c['type'] == "SMART"]
            winner = sorted(smart_ones, key=lambda x: x['latency'])[0] if smart_ones else candidates[0]
        else: # FAST
            winner = sorted(candidates, key=lambda x: x['latency'])[0]

        print(f"{Fore.GREEN}>>> WINNER: {winner['prefix']}/{winner['model']}{Style.RESET_ALL}\n")
        return winner

# ==============================================================================
# 3. THE SMART FACTORY (CREWAI LLM CLASS)
# ==============================================================================

def get_crew_llm(strategy="SMART"):
    """
    1. Runs the Router.
    2. Returns the 'smart' CrewAI LLM object with the correct prefix.
    """
    router = ModelRouter()
    best = router.find_best_model(strategy=strategy)
    
    # Construct the mandatory string: "provider/model_name"
    full_model_string = f"{best['prefix']}/{best['model']}"
    
    print(f"{Fore.YELLOW}Initializing CrewAI LLM: {full_model_string}{Style.RESET_ALL}")
    
    return LLM(
        model=full_model_string,
        temperature=0.7
    )

# ==============================================================================
# 4. CREWAI IMPLEMENTATION
# ==============================================================================

def run_dynamic_crew():
    # 1. Get the Dynamic LLM (Auto-routed)
    # This might return LLM(model="gemini/gemini-2.0-flash-exp") or LLM(model="groq/llama-3.3-70b-versatile")
    smart_llm = get_crew_llm(strategy="SMART")

    # 2. Define Agents 
    # Notice we pass the LLM object directly. 
    analyst = Agent(
        role='Tech Strategist',
        goal='Verify the model being used.',
        backstory='You are an AI expert.',
        verbose=True,
        llm=smart_llm  # <--- The Magic happens here
    )

    # 3. Define Task
    task = Task(
        description="Tell me exactly what Large Language Model you are. Mention your name and version.",
        expected_output="The model name.",
        agent=analyst
    )

    # 4. Kickoff
    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential
    )

    result = crew.kickoff()
    print(f"\n\n{Fore.MAGENTA}################## RESULT ##################{Style.RESET_ALL}")
    print(result)

if __name__ == "__main__":
    run_dynamic_crew()