import os
import json
from abc import ABC, abstractmethod

from google import genai
from google.genai.types import GenerateContentConfig, HarmBlockThreshold, HarmCategory, SafetySetting
from openai import OpenAI

# Improve modularity by imposing an abstract base class
class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients to ensure consistent interface."""
    @abstractmethod
    def generate_content(self, text_body: str) -> str:
        pass



class GeminiClient(BaseLLMClient):
    """
    A client to configure and interact with the Google Gemini API.
    """
    def __init__(self, config, response_schema, system_prompt):
        # Define Safety Settings
        safety_settings = [
            SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_NONE)
            for c in [
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                HarmCategory.HARM_CATEGORY_HARASSMENT,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
            ]
        ]
        
        self.model_config = GenerateContentConfig(
            temperature=config['temperature'],
            safety_settings=safety_settings,
            response_schema=response_schema,
            response_mime_type="application/json",
            system_instruction=system_prompt
        )

        # Initialize Client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = config['model_id']

    def generate_content(self, sutta_body):
        """
        Generates content using the configured Gemini model.
        """
        response_text = self.client.models.generate_content(
            model=self.model_id,
            config=self.model_config,
            contents=sutta_body
        ).text
        return  response_text
    

class OpenAIClient:
    """
    A client to configure and interact with the Google Gemini API.
    """
    def __init__(self, config, system_prompt):        
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.model_id = config['model_id']
        self.system_prompt = system_prompt


    def generate_content(self, sutta_body):
        """
        Generates content using the configured Gemini model.
        """
        # First define messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": sutta_body}
        ]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        response_text = json.loads(response.choices[0].message.content)
        return response_text

# Define factory to move between clients
def get_llm_client(extraction_config, system_prompt, response_schema_class) -> BaseLLMClient:
    """
    Factory function to get the appropriate LLM client based on config.
    """
    model_id = extraction_config['model_id']

    if 'gemini' in model_id:
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in environment variables for Gemini client.")
        return GeminiClient(extraction_config, response_schema_class, system_prompt)
    
    elif 'deepseek' in model_id:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables for DeepSeek client.")
        return OpenAIClient(extraction_config, system_prompt)
    
    else:
        raise ValueError(f"Unsupported model provider for model_id: {model_id}")