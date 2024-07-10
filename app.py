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


formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    history.append((prompt, output))
    return output



mychatbot = gr.Chatbot(
    avatar_images=["./user.png", "./botgm.png"], bubble_full_width=False, show_label=False, show_copy_button=True, likeable=True,)

additional_inputs=[
    gr.Slider(
        label="Temperature",
        value=0.7,
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        interactive=True,
        info="Higher values generate more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=6400,
        minimum=0,
        maximum=8000,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.01,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
