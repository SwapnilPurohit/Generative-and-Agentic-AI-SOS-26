import os
import re
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from openai import OpenAI

# Import components from our previous modules
from rag_pipeline import rag_query
from unet import SimpleUNet
from diffusion_forward import linear_beta_schedule

# Load environment variables
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.1-8b-instant"

# ==========================================
# 1. Define Custom Tools
# ==========================================

def search_knowledge_base(query: str) -> str:
    """Searches the local RAG ChromaDB for answers."""
    try:
        print(f"\n[Tool: search_knowledge_base] Querying for: {query}")
        result = rag_query(query)
        return result
    except Exception as e:
        return f"Error querying RAG: {e}"

def generate_digits(num_images_str: str) -> str:
    """Generates novel MNIST digits using the PyTorch DDPM."""
    try:
        num_images = int(num_images_str)
        print(f"\n[Tool: generate_digits] Generating {num_images} digits using PyTorch DDPM...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleUNet().to(device)
        model.load_state_dict(torch.load('models/ddpm_mnist.pth', map_location=device, weights_only=True))
        model.eval()
        
        timesteps = 200
        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas
        
        x = torch.randn((num_images, 1, 28, 28)).to(device)
        
        with torch.no_grad():
            for t in reversed(range(timesteps)):
                t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
                predicted_noise = model(x, t_tensor)
                
                alpha = alphas[t]
                alpha_cumprod = alphas_cumprod[t]
                mean = sqrt_recip_alphas[t] * (x - ((1.0 - alpha) / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise)
                
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(posterior_variance[t])
                    x = mean + sigma * noise
                else:
                    x = mean
        
        final_images = x
        final_images = (final_images + 1.0) / 2.0
        final_images = torch.clamp(final_images, 0, 1)
        
        # Determine nrow for a neat grid
        nrow = int(torch.sqrt(torch.tensor(num_images).float()).ceil().item())
        grid = make_grid(final_images.cpu(), nrow=nrow, padding=2, normalize=False)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Generated {num_images} Digits")
        
        output_path = os.path.join("outputs", "capstone_generated.png")
        plt.savefig(output_path)
        plt.close()
        
        return f"Successfully generated {num_images} digits and saved them to {output_path}"
    except Exception as e:
        return f"Error generating images: {e}"

TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "generate_digits": generate_digits
}

# ==========================================
# 2. System Prompt Definition
# ==========================================

SYSTEM_PROMPT = """You are an advanced Capstone Agent capable of researching AI concepts and generating images.
You have access to the following tools:

- search_knowledge_base: Searches a vector database to answer questions about Transformers, DDPMs, or VAEs. Input should be the search query string.
- generate_digits: Generates novel MNIST digits using a Diffusion Model. Input should be an integer representing the number of images to generate (e.g., '4').

You must solve the user's request by utilizing these tools. Use the following strict format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [search_knowledge_base, generate_digits]
Action Input: the input to the action
Observation: the result of the action (provided by the system)
... (this Thought/Action/Action Input/Observation loop can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

# ==========================================
# 3. The Orchestration Loop
# ==========================================

def run_agent(question: str, max_iterations: int = 7):
    print(f"\n=======================================================")
    print(f"Capstone Agent Initiated")
    print(f"User Request: {question}")
    print(f"=======================================================\n")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stop=["Observation:"],
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        print(f"\033[94m{reply}\033[0m")
        
        messages.append({"role": "assistant", "content": reply})
        
        if "Final Answer:" in reply:
            print("\n=======================================================")
            print("Agent finished successfully.")
            print("=======================================================\n")
            return
            
        action_match = re.search(r"Action: ([\w_]+)", reply)
        action_input_match = re.search(r"Action Input: (.*)", reply)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            if action in TOOLS:
                observation = TOOLS[action](action_input)
            else:
                observation = f"Error: '{action}' is not a valid tool."
                
            obs_text = f"Observation: {observation}"
            print(f"\033[92m{obs_text}\033[0m\n")
            
            messages.append({"role": "user", "content": obs_text})
        else:
            print("\033[91mError: Could not parse Action or Action Input from the response.\033[0m")
            messages.append({"role": "user", "content": "Observation: Formatting error. Please use the exact format: 'Action: <tool>' and 'Action Input: <input>'."})

    print("\nAgent reached maximum iterations without finding a final answer.")

if __name__ == "__main__":
    sample_request = "Explain the core mechanism of the Transformer architecture, and then generate 4 digits for me."
    run_agent(sample_request)
