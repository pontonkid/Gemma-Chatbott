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
