import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage

# ── Configuration ────────────────────────────────────────────────────────────

MCP_CONFIG = {
    "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],   # ✅ correct modern package
        "env": {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": "0"},
        "transport": "stdio",
    }
}

SYSTEM_PROMPT = """
You are a deep research agent with full browser access via Playwright.

Guidelines:
- Navigate websites carefully and wait for pages to fully load.
- Extract only relevant, factual information.
- If a page fails, retry once before giving up.
- Always summarize findings clearly and concisely.
""".strip()

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_skills(path: str = "skills.md") -> str:
    """Load custom instructions from skills.md, falling back to default prompt."""
    try:
        with open(path) as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    return SYSTEM_PROMPT


def print_step(step: dict) -> None:
    """Pretty-print each agent step."""
    for node_name, output in step.items():
        messages = output.get("messages", [])
        if not messages:
            continue

        last = messages[-1]
        print(f"\n{'─' * 60}")
        print(f"  Node : {node_name}")

        if isinstance(last, AIMessage):
            # Show tool calls if present, otherwise show text content
            if last.tool_calls:
                for tc in last.tool_calls:
                    print(f"  Tool : {tc['name']}")
                    args_preview = str(tc["args"])[:120]
                    print(f"  Args : {args_preview}")
            elif last.content:
                print(f"  AI   : {last.content}")

        elif isinstance(last, ToolMessage):
            preview = str(last.content)[:300]
            print(f"  Tool result preview: {preview}")

        print(f"{'─' * 60}")


# ── Main Agent ───────────────────────────────────────────────────────────────

async def run_agent(query: str) -> str:
    """Connect to MCP, build the agent, run the query, return final answer."""

    async with MultiServerMCPClient(MCP_CONFIG) as client:
        tools = client.get_tools()

        if not tools:
            raise RuntimeError("No tools loaded from MCP server. Is Playwright installed?")

        print(f"✅ Loaded {len(tools)} Playwright tool(s): {[t.name for t in tools]}")

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY"),  # explicit — avoids silent failures
        )

        # ✅ In LangGraph ≥ 0.2, pass prompt= (not state_modifier=) for system instructions
        agent = create_react_agent(
            llm,
            tools,
            prompt=load_skills(),      # <-- correct kwarg for modern LangGraph
        )

        print("\n--- Agent starting ---\n")

        final_answer = ""
        async for step in agent.astream(
            {"messages": [("user", query)]},
            stream_mode="updates",
        ):
            print_step(step)

            # Capture the last AI text as the final answer
            for output in step.values():
                for msg in output.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        final_answer = msg.content

        return final_answer


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    QUERY = (
        "Go to https://techcrunch.com and find the latest article about OpenAI. "
        "Return the headline and a 2-sentence summary."
    )

    result = asyncio.run(run_agent(QUERY))

    print("\n══════════════════════════════════════════════")
    print("FINAL ANSWER")
    print("══════════════════════════════════════════════")
    print(result)
