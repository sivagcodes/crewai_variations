import os
import json
import time
from typing import Dict
from dotenv import load_dotenv
from colorama import Fore, Style, init

# --- LANGCHAIN IMPORTS ---
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# --- LOCAL IMPORTS ---
from model_router import DynamicRouter

load_dotenv()
init(autoreset=True)

# ==============================================================================
# 1. THE INTELLIGENCE ENGINE
# ==============================================================================

def get_destination_intel(city: str) -> Dict[str, str]:
    """Uses a dynamic smart model to profile the destination."""
    print(f"{Fore.CYAN}--- INTEL: Scanning Destination '{city}' ---{Style.RESET_ALL}")
    
    router = DynamicRouter()
    try:
        cfg = router.get_best_model("SMART")
        llm = init_chat_model(model=cfg['model'], model_provider=cfg['provider'])
        
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
    except Exception as e:
        print(f"{Fore.RED}Error getting intel: {e}{Style.RESET_ALL}")
        # Return default values in case of error
        return {"vibe": "Standard", "currency": "USD", "walkability": "Unknown", "best_for": "General"}

# ==============================================================================
# 2. THE TRAVEL AGENT SWARM (Complex LangChain)
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