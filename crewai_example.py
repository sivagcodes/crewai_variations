# Minimal CrewAI example – run with `python crewai_example.py`

from crewai import Crew, Agent, Task, Process, LLM
 # or any other LLM wrapper you have installed

llm = LLM(model="gpt-4o-mini",api_key=""
)   # tiny, inexpensive model for quick demossk

# 2️⃣  Define a single agent
researcher = Agent(
    role="Researcher",
    goal="Find a concise definition of CrewAI and its main purpose",
    backstory="You are an AI researcher familiar with emerging multi‑agent frameworks.",
    verbose=True,
    llm=llm,
)

# 3️⃣  Define the task this agent will perform
task = Task(
    description="Give a short (1‑2 sentence) definition of CrewAI.",
    agent=researcher,
    expected_output="A clear definition of CrewAI in plain English."
)

# 4️⃣  Create the crew (using the default Sequential process)
crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,   # or Process.parallel if you have many agents
    verbose=True,
)

# 5️⃣  Run the crew and print the output
result = crew.kickoff()
print("\n=== CrewAI Definition ===")
print(result)