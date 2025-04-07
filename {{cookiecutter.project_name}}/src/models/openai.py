from typing import List, Union
from src.models.model import Model
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    A class to represent an OpenAI model.
    
    Attributes:
        name (str): The name of the model.
        max_tokens (int): The maximum number of tokens to generate.
        system_prompt (str): The system prompt to use.
    """
    def __init__(
        self,
        name: str = "http://localhost:1234/v1",
        max_tokens: int = 128,
        api_key='lm-studio',
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(name='OpenAI', max_tokens=max_tokens)
        self.model_name = name
        self.client = None
        self.system_prompt = kwargs['system_prompt']
        self.api_key = api_key
        self.temperature = temperature
        
        if 'azure_endpoint' in kwargs:
            self.azure_endpoint = kwargs['azure_endpoint']
        else:
            self.azure_endpoint = None
            
        print(f'Azure: {self.azure_endpoint}, {self.api_key}')
        self.load()

    def load(self) -> 'OpenAIModel':
        """
        Load the OpenAI model.
        """
        if self.azure_endpoint is not None:
            print(self.api_key, self.azure_endpoint)
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version='2024-02-01',
                azure_endpoint=self.azure_endpoint
            )
        elif 'http' in self.model_name:
            self.client = OpenAI(base_url=self.model_name, api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key)

        logging.log(
            logging.INFO, f'Created OpenAI client with base URL {self.model_name}')

    def generate(self, prompt: Union[str, List[str]], max_tokens: int = None) -> Union[str, List[str]]:
        """
        Generate text based on the prompt.
        
        Args:
            prompt (Union[str, List[str]]): The prompt to generate text from.
            
        Returns:
            str: The generated text.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        is_list = isinstance(prompt, list)
        if not is_list:
            prompt = [prompt]
        
        answers = []
        for p in tqdm(prompt, desc="Generating text", total=len(prompt)):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            else:
                messages.append({"role": "system", "content": "You are a helpful assistant."})
                
            messages.append({"role": "user", "content": p})
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )

            output = completion.choices[0].message.content
            if output:
                answers.append(output.strip())
            else:
                answers.append('')

        if not is_list:
            answers = answers[0]
        
        return answers
