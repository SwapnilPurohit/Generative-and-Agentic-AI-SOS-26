import os
import json
import re
import datetime
from openai import OpenAI

# Load environment variables from .env file manually
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

# Initialize the OpenAI client pointing to the Groq API
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.1-8b-instant"

# ==========================================
# 1. Define Custom Tools
# ==========================================

def calculate_math(expression: str) -> str:
    """Evaluates a mathematical expression safely."""
    try:
        # Use a safe subset of globals for eval
        allowed_names = {"__builtins__": None}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating math expression: {e}"

def read_file(filepath: str) -> str:
    """Reads the content of a local file."""
    try:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' not found."
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Truncate if too long to prevent context overflow
            if len(content) > 1000:
                return content[:1000] + "\n...[truncated]"
            return content
    except Exception as e:
        return f"Error reading file: {e}"

def get_time(dummy: str = "") -> str:
    """Returns the current local date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Map tool names to functions
TOOLS = {
    "calculate_math": calculate_math,
    "read_file": read_file,
    "get_time": get_time
}

# ==========================================
# 2. System Prompt Definition
# ==========================================

SYSTEM_PROMPT = """You are a ReAct (Reasoning and Acting) agent.
You have access to the following tools:

- calculate_math: Evaluates a mathematical expression (e.g., "5 * 3 + 2"). Input should be the math string.
- read_file: Reads the contents of a local file. Input should be the file path.
- get_time: Returns the current date and time. Input can be empty.

You must solve the user's request by utilizing these tools. Use the following strict format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [calculate_math, read_file, get_time]
Action Input: the input to the action
Observation: the result of the action (provided by the system)
... (this Thought/Action/Action Input/Observation loop can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

# ==========================================
# 3. The Orchestration Loop
# ==========================================

def run_agent(question: str, max_iterations: int = 5):
    print(f"\nUser Question: {question}\n")
    print("-" * 50)
    
    # Initialize conversation history with system prompt and the user's question format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    for i in range(max_iterations):
        # Call the LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stop=["Observation:"], # Stop generating when it's time for the system to provide an observation
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        print(f"\033[94m{reply}\033[0m") # Print the LLM's thought process in blue
        
        # Append the LLM's response to memory
        messages.append({"role": "assistant", "content": reply})
        
        # Check if the LLM arrived at the final answer
        if "Final Answer:" in reply:
            print("-" * 50)
            print("\nAgent finished successfully.")
            return
            
        # Parse Action and Action Input using regex
        action_match = re.search(r"Action: ([\w_]+)", reply)
        action_input_match = re.search(r"Action Input: (.*)", reply)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # Execute the tool
            if action in TOOLS:
                observation = TOOLS[action](action_input)
            else:
                observation = f"Error: '{action}' is not a valid tool."
                
            obs_text = f"Observation: {observation}"
            print(f"\033[92m{obs_text}\033[0m\n") # Print observation in green
            
            # Append observation to memory as if the user provided it
            messages.append({"role": "user", "content": obs_text})
        else:
            print("\033[91mError: Could not parse Action or Action Input from the response.\033[0m")
            messages.append({"role": "user", "content": "Observation: Formatting error. Please use the exact format: 'Action: <tool>' and 'Action Input: <input>'."})

    print("-" * 50)
    print("\nAgent reached maximum iterations without finding a final answer.")

if __name__ == "__main__":
    # Test the ReAct loop
    sample_question = "What is the current time? Then take the current hour (in 24-hour format) and multiply it by 17."
    run_agent(sample_question)
