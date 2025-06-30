import os
from google import genai
from google.genai.types import GenerateContentConfig, HarmBlockThreshold, HarmCategory, SafetySetting

class GeminiClient:
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