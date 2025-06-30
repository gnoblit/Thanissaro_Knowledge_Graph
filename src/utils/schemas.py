from pydantic import BaseModel, Field
from typing import List

class Concept(BaseModel):
    """
    Represents a single conceptual term extracted from a Sutta.
    This class validates the structure of individual concept objects.
    """
    concept_name: str = Field(..., description="A concise, normalized name for the concept.")
    concept_type: str = Field(..., description="The generated category for the concept (e.g., Person, Place).")
    evidence_quote: str = Field(..., description="The specific sentence or phrase from the text as evidence.")

class SuttaConcepts(BaseModel):
    """
    Represents the top-level JSON object containing a list of concepts.
    This is the main model for validating the LLM's output for a single Sutta.
    """
    concepts: List[Concept]

    