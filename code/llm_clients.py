"""LLM client wrappers for Gemini and OpenAI with structured outputs."""

import os
import json
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from pydantic import BaseModel

from schemas import TransitionDetection

load_dotenv()


class GeminiClient:
    """Wrapper for Google Gemini API with structured output support."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: type[BaseModel] = TransitionDetection
    ) -> BaseModel:
        """Generate a structured response from Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            response_schema: Pydantic model for response structure
            
        Returns:
            Instance of response_schema with parsed response
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
        )
        
        # Parse JSON and validate with Pydantic
        response_json = json.loads(response.text)
        return response_schema(**response_json)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a raw text response from Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            
        Returns:
            Response text
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        return response.text


class OpenAIClient:
    """Wrapper for OpenAI API with structured output support."""
    
    def __init__(self, model_name: str = "gpt-5"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: type[BaseModel] = TransitionDetection
    ) -> BaseModel:
        """Generate a structured response from OpenAI using structured outputs.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            response_schema: Pydantic model for response structure
            
        Returns:
            Instance of response_schema with parsed response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Use beta parse method for structured outputs
        # Note: GPT-5 doesn't support temperature parameter
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_schema
        )
        
        return completion.choices[0].message.parsed
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a raw text response from OpenAI.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            
        Returns:
            Response text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content


def parse_llm_response(response_text: str, schema: type[BaseModel] = TransitionDetection) -> BaseModel:
    """Parse LLM JSON response into a Pydantic model.
    
    Args:
        response_text: JSON string from LLM
        schema: Pydantic model class
        
    Returns:
        Instance of schema with parsed data
    """
    # Clean up potential markdown code blocks
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Parse JSON and validate with Pydantic
    data = json.loads(text)
    return schema(**data)
