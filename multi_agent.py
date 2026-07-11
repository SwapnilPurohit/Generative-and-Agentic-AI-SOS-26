import os
from crewai import Agent, Task, Crew, Process, LLM
import litellm

# Monkey patch LiteLLM to strip 'cache_breakpoint' which CrewAI forces but Groq rejects
original_completion = litellm.completion
def patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            if "cache_breakpoint" in msg:
                del msg["cache_breakpoint"]
    return original_completion(*args, **kwargs)
litellm.completion = patched_completion

# ---------------------------------------------------------
# 1. Load Environment Variables (Groq API Key)
# ---------------------------------------------------------
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

# Configure the LLM to use Groq's fast inference endpoint natively via litellm
groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.7
)

# ---------------------------------------------------------
# 2. Define Agents (The "Who")
# ---------------------------------------------------------

# The Researcher Agent
researcher = Agent(
    role='Senior AI Research Analyst',
    goal='Analyze complex AI architectures and break them down into simple concepts.',
    backstory="""You are an expert AI researcher at a top university. You excel at 
    understanding complex papers like 'Attention Is All You Need' and extracting the 
    core mechanisms in a way that is highly accurate but easy to digest.""",
    verbose=True,
    allow_delegation=False, # This agent will just do research, not delegate
    cache=False, # Disable caching to prevent LiteLLM/Groq parameter errors
    llm=groq_llm
)

# The Writer Agent
writer = Agent(
    role='Technical Content Strategist',
    goal='Craft compelling and easy-to-understand educational content based on research.',
    backstory="""You are a renowned technical writer known for your ability to explain 
    highly complex AI topics to absolute beginners. You take raw research notes and 
    turn them into engaging, well-structured blog posts or explanations.""",
    verbose=True,
    allow_delegation=False,
    cache=False, # Disable caching to prevent LiteLLM/Groq parameter errors
    llm=groq_llm
)

# ---------------------------------------------------------
# 3. Define Tasks (The "What")
# ---------------------------------------------------------

# The Research Task
research_task = Task(
    description="""Provide a comprehensive overview of the 'Self-Attention' mechanism 
    found in Transformer models. Explain what it is, why it is important, and how it 
    differs from older RNN approaches. Output bulleted notes.""",
    expected_output="A bulleted list of research notes explaining the Self-Attention mechanism.",
    agent=researcher
)

# The Writing Task
write_task = Task(
    description="""Using the research notes provided by the Senior AI Research Analyst, 
    write a short, engaging 2-paragraph explanation of Self-Attention intended for a 
    high school student. Use a fun analogy.""",
    expected_output="A 2-paragraph explanation of Self-Attention with a fun analogy.",
    agent=writer
)

# ---------------------------------------------------------
# 4. Create the Crew and Execute (The Orchestration)
# ---------------------------------------------------------
def main():
    print("Welcome to the Multi-Agent CrewAI Demo! (Powered by Groq)")
    print("Assembling the crew (Researcher + Writer)...\n")
    
    # Form the crew with a sequential process
    # This means research_task runs first, and its output is passed to write_task
    ai_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True
    )

    # Kickoff the process!
    print("Kicking off tasks! You will see the agents' thoughts printed below:\n")
    print("-" * 50)
    result = ai_crew.kickoff()
    
    print("-" * 50)
    print("\n[System] Final Handoff Result (Output from Writer Agent):")
    print("\n" + str(result))

if __name__ == "__main__":
    main()
