from src.prompts.prompt import Prompt
from typing import Any

def get_prompt(name: str, **kwargs) -> Prompt:
    prompt = {
        'prompt': Prompt,
    }[name]
    
    return prompt(**kwargs)
    