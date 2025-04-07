from typing import Any

class Prompt:
    def __init__(self, **kwargs):
        self.prompt = None
        
        if 'template' in kwargs:
            self.from_template(kwargs['template'])

    def get_prompt(self, **kwargs) -> dict:
        return {
            'prompt': self.format(**kwargs)
        }
    
    def from_template(self, template: str, **kwargs):
        self.template = template
        
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def __call__(self, **kwargs) -> Any:
        return self.get_prompt(
            **kwargs
        )
