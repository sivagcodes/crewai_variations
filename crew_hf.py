import os
from crewai import Agent, Task, Crew, LLM


# ⚠️ NEVER commit API keys to code! Use environment variables instead
# setx GROQ_API_KEY "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" (Windows)
# Then remove this line:
os.environ["GROQ_API_KEY"] = ""
os.environ["LITELLM_LOG"] = "ERROR"  # Only critical errors
os.environ["LITELLM_NO_COLD_STORAGE"] = "1"  # Skip advanced logging
# Validate API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Set GROQ_API_KEY environment variable first")

# ✅ CORRECTED: Use valid Groq model name
llm = LLM(
    model="groq/openai/gpt-oss-20b",  # Valid Qwen model on Groq
    temperature=0.1
    
)

# Test LLM before agent creation
#print("Testing LLM:", llm.invoke("Say hello").content)

# Rest of your code remains identical...
researcher = Agent(
    role="Market Researcher",
    goal="Analyze market trends and identify opportunities",
    backstory="Expert analyst with 10+ years in competitive intelligence.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content based on research findings",
    backstory="Skilled writer who transforms data into compelling narratives.",
    llm=llm,
    verbose=True
)

research_task = Task(
    description="Research latest trends in AI agent frameworks for 2025.",
    expected_output="Comprehensive report with 3 key trends and opportunities.",
    agent=researcher
)

write_task = Task(
    description="Write a 300-word blog post based on the research findings.",
    expected_output="Well-structured blog post ready for publication.",
    agent=writer,
    context=[research_task]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

result = crew.kickoff()
print(result)
