import os
from crewai import Crew, Agent, Task, Process, LLM

# --- Key fix: satisfy CrewAI's OpenAI check OR switch provider explicitly ---

# Option A: Set a dummy OPENAI_API_KEY (quick and works for local testing)
os.environ["GROQ_API_KEY"] = ""
os.environ["LITELLM_LOG"] = "ERROR"  # Only critical errors
os.environ["LITELLM_NO_COLD_STORAGE"] = "1"  # Skip advanced logging
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Option B (recommended with newer CrewAI): explicitly select Ollama as provider
os.environ["CREWAI_LLM_PROVIDER"] = "ollama"

# Initialize Ollama LLM
llm = LLM(
    model="ollama/tinyllama",          # note the slash form used in docs
    base_url="http://:11434", # Ollama default
    api_key="ollama",                  # any non-empty string; not actually used by Ollama
    temperature=0.2
)

researcher = Agent(
    role="Researcher",
    goal="Find a concise definition of CrewAI and its main purpose",
    backstory="You are an AI researcher familiar with emerging multi-agent frameworks.",
    verbose=True,
    llm=llm,
)

task = Task(
    description="Give a short (1-2 sentence) definition of CrewAI.",
    agent=researcher,
    expected_output="A clear definition of CrewAI in plain English.",
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
print("\n=== CrewAI Definition ===")
print(result)
