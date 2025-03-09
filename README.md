# Reproducing nanoGPT from Scratch

This repository contains a from-scratch reproduction of nanoGPT, a minimal and efficient implementation of GPT-2 (124M). The project follows a structured and incremental approach, building a fully functional transformer-based language model step by step. By following the commit history, you can track the model's development and refinement over time.

## Project Overview
- Implements a GPT-2 (124M) model from scratch
- Uses structured git commits for easy progress tracking
- Can be extended to reproduce larger models, including GPT-3
- Training performed on a local GPU
- Outputs generated text based on input prompts

## Features
- **Transformer Architecture**: Implements attention layers, positional embeddings, and feedforward networks
- **Optimized Training**: Supports flash attention and mixed precision training
- **Scalable Implementation**: Can scale up from GPT-2 (124M) to larger models
- **Step-by-Step Development**: The commit history allows incremental learning and debugging
- **Pretraining Only**: Focuses on model pretraining; fine-tuning for chat applications is not included

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JaySh21/GPT
   cd GPT
   ```

   ```

## Training Details
- The model was trained on a local GPU
- Training time and performance may vary depending on hardware
- Model checkpoints are saved periodically for resumption and evaluation

## Sample Outputs
After training on 10B tokens, the model generates:
```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone...
```
After training on 40B tokens, the model generates:
```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs...
```

## Future Work
- Extend the model to larger versions like GPT-3
- Fine-tune the model for interactive chatbot applications
- Optimize training for cost efficiency and performance improvements

## Credits
This project is a reproduction of an open-source implementation inspired by nanoGPT. Special thanks to the original creator for their detailed breakdown and structured approach.

Watch the original YouTube lecture here: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=2407s)

