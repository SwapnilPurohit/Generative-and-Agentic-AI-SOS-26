# Generative and Agentic AI 🚀

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF7F50.svg)](https://crewai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive exploration of modern Artificial Intelligence, spanning from the mathematical foundations of generative models to the orchestration of autonomous LLM agents. This repository contains raw, from-scratch implementations of Variational Autoencoders (VAEs), Denoising Diffusion Probabilistic Models (DDPMs), and various Agentic architectures including ReAct and Retrieval-Augmented Generation (RAG).

## 📑 Table of Contents
- [Part 1: Generative AI](#part-1-generative-ai)
  - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
  - [Denoising Diffusion Probabilistic Models (DDPMs)](#denoising-diffusion-probabilistic-models-ddpms)
- [Part 2: Agentic AI](#part-2-agentic-ai)
  - [Transformers & Self-Attention](#transformers--self-attention)
  - [Conversational Memory & Tool Calling](#conversational-memory--tool-calling)
  - [ReAct Architecture](#react-architecture)
  - [RAG & Multi-Agent Systems](#rag--multi-agent-systems)
- [Installation & Setup](#installation--setup)

---

## 🧠 Part 1: Generative AI

The first half of this project focuses on understanding and implementing generative image models from scratch using PyTorch.

### Variational Autoencoders (VAEs)
- **`train_vae.py`**: A complete pipeline to compress and reconstruct MNIST handwritten digits. Implements the Reparameterization Trick and optimizes the Evidence Lower Bound (ELBO) to create a continuous, interpolatable 2D latent space.

### Denoising Diffusion Probabilistic Models (DDPMs)
- **`diffusion_forward.py`**: Visualizes the forward diffusion process, demonstrating how linear and cosine variance schedules inject Gaussian noise into a clean image over $T$ timesteps.
- **`unet.py`**: A from-scratch implementation of a symmetric down-and-up convolutional U-Net architecture with skip connections and sinusoidal timestep embeddings.
- **`train_ddpm.py`**: The training loop that optimizes the U-Net to predict added noise using the simplified DDPM objective function.
- **`sample_ddpm.py`**: The reverse sampling script that starts from pure Gaussian noise and iteratively denoises it to generate novel, unconditional digits.

---

## 🤖 Part 2: Agentic AI

The second half transitions from generating pixels to orchestrating intelligence. We explore how Large Language Models (LLMs) can be structured to think, use tools, and collaborate.

### Transformers & Self-Attention
- **`self_attention.py`**: A pure NumPy mathematical implementation of `ScaledDotProductAttention` and `MultiHeadAttention`. This serves as the foundational building block for modern LLMs, mapping theoretical understanding into raw tensors.

### Conversational Memory & Tool Calling
- **`chatbot_memory.py`**: A terminal-based chatbot that maintains conversational context by managing a rolling history of `user` and `assistant` messages via the Groq API.
- **`autonomous_agent.py`**: Demonstrates primitive agentic behavior by instructing an LLM with a strictly typed JSON schema. The model autonomously identifies when it needs external data, requests tool execution, and formulates an answer based on the tool's output.

### ReAct Architecture
- **`react_agent.py`**: A robust, raw-Python implementation of the **Reasoning and Acting (ReAct)** loop. Without relying on heavy frameworks, this script enforces a rigorous `Thought -> Action -> Action Input -> Observation` cycle, allowing the model to dynamically solve math problems, read local files, and check the system time.

### RAG & Multi-Agent Systems
- **`rag_pipeline.py`**: A Retrieval-Augmented Generation (RAG) system using `ChromaDB` and HuggingFace `sentence-transformers`. It embeds local text documents into a vector database, performs similarity searches based on user queries, and feeds the relevant context directly to the LLM.
- **`multi_agent.py`**: Orchestrates a sequential collaboration between two autonomous agents using **CrewAI**. A *Senior AI Research Analyst* extracts complex technical concepts, and a *Technical Content Strategist* transforms those notes into engaging, easy-to-understand educational content.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies:**
   Make sure you have Python 3.11+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` should include `torch`, `torchvision`, `numpy`, `openai`, `langchain`, `langchain-groq`, `crewai`, `chromadb`, and `sentence-transformers`.)*

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key (required for the Agentic AI scripts):
   ```bash
   GROQ_API_KEY="your_api_key_here"
   ```

4. **Run the scripts!**
   - For generative models, start with `python train_vae.py`.
   - For agentic systems, try `python react_agent.py` or `python multi_agent.py`.

---
*Developed as part of the Summer of Science (SOS) 2026 Exploration.*
