from huggingface_hub import InferenceClient
import gradio as gr
import random

client = InferenceClient("google/gemma-2b-it")

def format_prompt(message, history):
    prompt = ""
    if history:
        for user_prompt, bot_response in history:
            prompt += f"<start_of_turn>user{user_prompt}<end_of_turn>"
            prompt += f"<start_of_turn>model{bot_response}"
    prompt += f"<start_of_turn>user{message}<end_of_turn><start_of_turn>model"
    return prompt

def generate(prompt, history, temperature=0.7, max_new_tokens=1024, top_p=0.90, repetition_penalty=0.9):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
   
    if not history:
        history = []

    rand_seed = random.randint(1, 1111111111111111)
    
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=rand_seed,
    )
