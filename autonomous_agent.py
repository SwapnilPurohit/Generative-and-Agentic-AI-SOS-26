import os
import json
from openai import OpenAI

# Load environment variables from .env file manually
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

# Initialize the OpenAI client pointing to the Groq API
# Make sure to set the GROQ_API_KEY environment variable!
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# You can change the model to llama-3.1-70b-versatile or mixtral-8x7b-32768
MODEL = "llama-3.1-8b-instant"

# 1. Define our tool (a python function)
def get_weather(location: str) -> str:
    """A dummy function that returns mock weather data."""
    print(f"\n[System] Tool 'get_weather' executed for location: {location}")
    # In a real app, this would make an API call to a weather service.
    mock_weather = {
        "New York": "75°F and sunny",
        "London": "60°F and rainy",
        "Tokyo": "80°F and cloudy"
    }
    return mock_weather.get(location, f"Weather data not available for {location}. Assuming 70°F and clear.")

# 2. Define the JSON schema for the tool so the LLM understands how to call it
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., San Francisco",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

def main():
    print("Welcome to the Autonomous Agent Demo! (Powered by Groq & LLaMA 3)")
    user_prompt = "What's the weather like in Tokyo right now?"
    print(f"\nUser: {user_prompt}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_prompt}
    ]
    
    print("\n[System] Sending prompt to the LLM...")
    
    try:
        # 3. Call the LLM, providing our tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto", # The model decides whether to call a tool or not
        )
        
        response_message = response.choices[0].message
        
        # 4. Check if the LLM decided to call a tool
        tool_calls = response_message.tool_calls
        if tool_calls:
            print("\n[System] The LLM decided to call a tool!")
            
            # Step 5: Append the assistant's tool-call message to memory
            messages.append(response_message)
            
            # Step 6: Execute the tool locally and provide the result back to the LLM
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "get_weather":
                    # Execute the python function
                    function_response = get_weather(location=function_args.get("location"))
                    
                    # Append the tool's output to memory
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
            
            # Step 7: Send the tool results back to the LLM so it can formulate a final answer
            print("\n[System] Sending tool results back to the LLM...")
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            print(f"\nAssistant: {second_response.choices[0].message.content}")
            
        else:
            # If the LLM didn't call a tool, just print its response
            print(f"\nAssistant: {response_message.content}")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Did you set your GROQ_API_KEY environment variable?")

if __name__ == "__main__":
    main()
