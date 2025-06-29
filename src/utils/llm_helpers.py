import os

from google import genai
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting

from dotenv import load_dotenv

load_dotenv()

SAFETY_SETTINGS = [
    SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_NONE)
    for c in [HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
              HarmCategory.HARM_CATEGORY_HARASSMENT,
              HarmCategory.HARM_CATEGORY_HATE_SPEECH,
              HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT]
]

def initialize_gemini_client(config):
    """
    Configures the Gemini API client and returns it.

    Args:
        config (dict): A dictionary containing 'api_key_path' and 'model_id'.

    Returns:
        A Google genAI client.
    """
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print(f"Google GenAI client configured.")
        return client
    except FileNotFoundError:
        print(f"Error: API key file not found at {config['api_key_path']}")
        return None
    except Exception as e:
        print(f"An error occurred during LLM initialization: {e}")
        return None