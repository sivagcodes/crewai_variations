import os
import json
import time
from typing import Dict, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- LANGCHAIN IMPORTS ---
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- INTEL & ROUTER UTILS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ==============================================================================
# 1. THE INTELLIGENCE ENGINE (Gemini 2.5 Flash)
#    - Analyzes the "Vibe" and "Logistics" before planning starts.
# ==============================================================================

def get_destination_intel(city: str) -> Dict[str, str]:
    """Uses Gemini 2.5 Flash to profile the destination."""
    print(f"{Fore.CYAN}--- INTEL: Scanning Destination '{city}' with Gemini 2.5 Flash ---{Style.RESET_ALL}")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return {"vibe": "General", "currency": "USD", "walkability": "Moderate"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
        
        # We ask for a "Vibe Check" to adjust the Planner's personality later
        template = """
        Analyze this destination: '{city}'.
        Return ONLY valid JSON:
        {{
            "vibe": "One adjective (e.g., Ancient, Futuristic, Chaotic, Romantic)",
            "currency": "Currency Code (e.g., JPY, EUR)",
            "walkability": "High/Medium/Low",
            "best_for": "History/Food/Adventure/Relaxation"
        }}
        """
        response = llm.invoke(template.format(city=city))
        clean_json = response.content.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        print(f"{Fore.GREEN} > Intel Acquired: {data}{Style.RESET_ALL}")
        return data
    except:
        return {"vibe": "Standard", "currency": "USD", "walkability": "Unknown"}

# ==============================================================================
# 2. THE DYNAMIC ROUTER (Finds "Smart" vs "Fast" Routes)
# ==============================================================================

class DynamicRouter:
    def __init__(self):
        self.provider_map = {"GOOGLE": "google_genai", "OPENAI": "openai", "GROQ": "groq"}
        self.routes = {
            "GOOGLE": lambda: ["gemini-2.0-flash-exp", "gemini-1.5-pro"],
            "OPENAI": lambda: ["gpt-4o", "gpt-4o-mini"],
            "GROQ": lambda: ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
        }

    def _ping(self, provider, model):
        try:
            start = time.time()
            if provider == "GOOGLE": ChatGoogleGenerativeAI(model=model).invoke("Hi")
            elif provider == "GROQ": ChatGroq(model=model).invoke("Hi")
            elif provider == "OPENAI": ChatOpenAI(model=model).invoke("Hi")
            return {"lc_p": self.provider_map[provider], "m": model, "lat": time.time() - start}
        except: return None

    def get_best_model(self, strategy="SMART") -> Dict:
        """Finds the best model based on Strategy (SMART=Capability, FAST=Speed)"""
        print(f"{Fore.CYAN}--- ROUTER: Finding Best '{strategy}' Model... ---{Style.RESET_ALL}")
        candidates = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = []
            for p, fetcher in self.routes.items():
                if os.getenv(f"{p}_API_KEY"):
                    for m in fetcher(): futures.append(ex.submit(self._ping, p, m))
            
            for f in as_completed(futures):
                if res := f.result(): candidates.append(res)

        if not candidates: raise Exception("No Models Online!")

        # Simple Logic: "Smart" prefers GPT-4/Gemini Pro. "Fast" prefers Flash/Groq.
        if strategy == "SMART":
            # Prioritize heavy models
            smart_keywords = ["pro", "4o", "70b"]
            smart = [c for c in candidates if any(k in c['m'] for k in smart_keywords)]
            winner = sorted(smart, key=lambda x: x['lat'])[0] if smart else candidates[0]
        else:
            # Prioritize latency
            winner = sorted(candidates, key=lambda x: x['lat'])[0]

        print(f"{Fore.YELLOW} > Routed to: {winner['lc_p']}/{winner['m']} ({winner['lat']:.2f}s){Style.RESET_ALL}")
        return {"provider": winner['lc_p'], "model": winner['m']}

# ==============================================================================
# 3. THE TRAVEL AGENT SWARM (Complex LangChain)
# ==============================================================================

def run_travel_planner(destination: str, days: int = 3):
    # STEP 1: GATHER INTEL
    intel = get_destination_intel(destination)

    # STEP 2: ROUTING (We need two brains)
    router = DynamicRouter()
    # Brain A: The Architect (Smart model for the itinerary)
    planner_cfg = router.get_best_model("SMART")
    # Brain B: The Logistics Manager (Fast model for packing/money)
    logistics_cfg = router.get_best_model("FAST")

    # STEP 3: INITIALIZE MODELS
    planner_model = init_chat_model(model=planner_cfg['model'], model_provider=planner_cfg['provider'])
    logistics_model = init_chat_model(model=logistics_cfg['model'], model_provider=logistics_cfg['provider'])

    # STEP 4: DEFINE CHAINS
    
    # --- CHAIN A: The Deep Itinerary ---
    itinerary_prompt = ChatPromptTemplate.from_template("""
    You are a luxury travel architect. The destination is {city}.
    
    INTEL REPORT:
    - Vibe: {vibe}
    - Walkability: {walkability}
    - Best For: {best_for}

    Create a complex, non-generic {days}-day itinerary.
    For each day, provide:
    1. Morning (Specific activity with timing)
    2. Lunch (Specific food recommendation based on {best_for})
    3. Afternoon (Hidden gem or culture spot)
    4. Evening (Nightlife or relaxation matching the '{vibe}' vibe)
    
    Format as clean Markdown.
    """)
    itinerary_chain = itinerary_prompt | planner_model | StrOutputParser()

    # --- CHAIN B: Logistics (Running Parallel) ---
    logistics_prompt = ChatPromptTemplate.from_template("""
    You are a pragmatic travel assistant. 
    Destination: {city}. Currency: {currency}.
    
    Provide a bulleted list of:
    1. 3 Essential items to pack for a "{vibe}" location.
    2. One specific scam to avoid in {city}.
    3. Estimated daily budget in {currency} (Low/Mid/High).
    
    Keep it very brief.
    """)
    logistics_chain = logistics_prompt | logistics_model | StrOutputParser()

    # STEP 5: EXECUTE SWARM (Parallel Execution)
    print(f"\n{Fore.MAGENTA}--- SWARM: Generating Plan for {destination}... ---{Style.RESET_ALL}")
    
    swarm = RunnableParallel({
        "itinerary": itinerary_chain,
        "logistics": logistics_chain
    })

    # We inject the Intel data + User Input into the parallel chains
    inputs = {
        "city": destination,
        "days": days,
        "vibe": intel['vibe'],
        "currency": intel['currency'],
        "walkability": intel['walkability'],
        "best_for": intel['best_for']
    }
    
    result = swarm.invoke(inputs)

    # STEP 6: RENDER OUTPUT
    print("\n" + "="*60)
    print(f"{Fore.GREEN}✈️  TRIP ARCHITECTURE: {destination.upper()} ({days} Days){Style.RESET_ALL}")
    print("="*60)
    print(result['itinerary'])
    
    print("\n" + "-"*60)
    print(f"{Fore.YELLOW}📦 LOGISTICS & INTEL{Style.RESET_ALL}")
    print("-"*(60))
    print(result['logistics'])

if __name__ == "__main__":
    # Example: Planning a complex trip to Tokyo
    run_travel_planner("Paris", days=3)