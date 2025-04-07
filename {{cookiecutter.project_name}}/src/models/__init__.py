from src.models.model import Model
from src.models.hf_model import HFModel
from src.models.openai import OpenAIModel
from src.models.anthropic import AnthropicModel
from src.models.gemma3 import Gemma3
from src.models.llama4 import Llama4
from src.models.vllm_model import VLLMModel


def get_model(**kwargs) -> Model:
    name = kwargs.get('name')
    if 'vllm/' in name:
        name = name.replace('vllm/', '')
        kwargs['name'] = name
        return VLLMModel(**kwargs)
    elif 'anthropic/' in name:
        name = name.replace('anthropic/', '')
        kwargs['name'] = name
        return AnthropicModel(**kwargs)
    elif 'openai/' in name:
        name = name.replace('openai/', '')
        kwargs['name'] = name
        return OpenAIModel(**kwargs)
    elif 'azure/' in name:
        name = name.replace('azure/', '')
        kwargs['name'] = name
        return OpenAIModel(**kwargs)
    elif 'gemma-3' in name:
        return Gemma3(**kwargs)
    elif 'Llama-4' in name:
        return Llama4(**kwargs)
    else:
        return HFModel(**kwargs)
