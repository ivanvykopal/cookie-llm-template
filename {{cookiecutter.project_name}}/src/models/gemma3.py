from typing import List, Union
from src.models.model import Model
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Gemma3(Model):
    """
    A class to represent a Gemma3 model.
    
    Attributes:
        name (str): The name of the model.
        max_tokens (int): The maximum number of tokens to generate.
        do_sample (bool): Whether to use sampling.
        device_map (str): The device map.
        load_in_4bit (bool): Whether to load the model in 4-bit.
        load_in_8bit (bool): Whether to load the model in 8-bit.
        offload_folder (str): The offload folder.
        offload_state_dict (bool): Whether to offload the state dict.
        max_memory (Any): The maximum memory to use.
        system_prompt (str): The system prompt to use.    
    """
    def __init__(
        self, 
        name: str = 'google/gemma-3-27b-it',
        max_tokens: int = 128, 
        do_sample: bool = False, 
        device_map: str = 'auto', 
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(name='HFModel', max_tokens=max_tokens)
        self.model_name = name
        self.tokenizer = None
        self.model = None
        self.device_map = eval(device_map) if '{' in device_map else device_map
        self.do_sample = do_sample
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.temperature = temperature
            
        if 'max_memory' not in kwargs:
            self.max_memory = None
        else:
            self.max_memory = eval(kwargs['max_memory']) if kwargs['max_memory'] else None
        self.system_prompt = kwargs['system_prompt'] if kwargs['system_prompt'] != 'None' else None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load()
        
    def _load_model(self) -> None:
        """
        Load the model.
        """
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map = self.device_map,
            max_memory = self.max_memory,
            torch_dtype=torch.bfloat16,
        )
        
    def load_quantized_model(self) -> None:
        """
        Load the quantized model.
        """
        print(f'Loading quantized model - {self.model_name}')
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_8bit=self.load_in_8bit,
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name, 
            device_map=self.device_map, 
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

    def load(self) -> 'Gemma3':
        """
        Load the Hugging Face model.
        """
        self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
        if self.load_in_8bit or self.load_in_4bit:
            self.load_quantized_model()
        else:
            self._load_model()

        logging.log(
            logging.INFO, f'Loaded model and tokenizer from {self.model_name}')
        
    def _is_chat(self) -> bool:
        """
        Check if the model is a chat model.
        """
        return hasattr(self.tokenizer, 'chat_template')
    
    def _get_system_role(self) -> str:
        return 'system'

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
        for p in tqdm(prompt, desc="Generating text", unit="prompt"):
            if self.system_prompt and self._get_system_role() == 'system':
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": p}]}
                ]
            elif self.system_prompt:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": f'{self.system_prompt}\n\n{p}'}]}
                ]
            else:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": p}]}
                ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors='pt'
            ).to(self.device, dtype=torch.bfloat16)
        
            
            with torch.inference_mode():
                generated_output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                )
                generated_ids = generated_output.sequences
            
            input_len = inputs["input_ids"].shape[-1]
            
            decoded = self.tokenizer.batch_decode(
                generated_ids[:, input_len:],
                skip_special_tokens=True
            )[0]
            
            answers.append(decoded.strip())
        
        if not is_list:
            answers = answers[0]
        
        return answers
