import os
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

def main():
    print("Welcome to the Terminal Chatbot! (Powered by Groq & LLaMA 3)")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # This list acts as our persistent conversation memory for this session
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant. You remember context from previous messages in the conversation."}
    ]
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
            
            # 1. Append user's message to memory
            messages.append({"role": "user", "content": user_input})
            
            # 2. Call the API with the full conversation history
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            # 3. Extract and print the assistant's reply
            assistant_reply = response.choices[0].message.content
            print(f"\nAssistant: {assistant_reply}\n")
            
            # 4. Append assistant's reply to memory so it is remembered for the next turn
            messages.append({"role": "assistant", "content": assistant_reply})
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Did you set your GROQ_API_KEY environment variable?")
            break

if __name__ == "__main__":
    main()
