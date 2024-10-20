import os
from typing import Optional, Dict, Any
from flask import jsonify
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI
from groq import Groq

class HybridResponse(BaseModel):
    """
    Modelo para representar una respuesta híbrida que puede incluir información de ubicación. No uses caracteres especiales.
    
    Attributes:
        ubicacion (Optional[str]): Nombre de la ubicación, si es aplicable.
        lat (Optional[float]): Latitud de la ubicación, si es aplicable.
        lon (Optional[float]): Longitud de la ubicación, si es aplicable.
        respuesta (str): La respuesta principal del asistente.
    """
    respuesta: str = Field(alias='res')


Response = HybridResponse

MODELS = {
    "ollama": "llama3.1:8b",
    "groq": "llama3-70b-8192"
}

def create_completion(prompt: str, system_prompt: str , lang: str = "Spanish", model_choice: str = "groq"):
    """
    Crea una completion utilizando el modelo especificado.
    
    Args:
        prompt (str): El prompt del usuario.
        system_prompt (str): El prompt del sistema. Por defecto, DEFAULT_SYSTEM_PROMPT.
        lang (str): El idioma de la respuesta. Por defecto, "Spanish".
        model_choice (str): El modelo a utilizar. Por defecto, "groq".
    
    Returns:
        tuple: Una tupla conteniendo la respuesta JSON y el código de estado HTTP.
    
    Raises:
        ValueError: Si los argumentos son inválidos.
    """
    print(f"DEBUG: Entering create_completion with model_choice: {model_choice}")
    if not isinstance(prompt, str) or not isinstance(system_prompt, str):
        raise ValueError("Both prompt and system_prompt must be strings.")

    if model_choice not in MODELS:
        raise ValueError(f"Invalid model choice. Available models are: {', '.join(MODELS.keys())}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    return _create_completion_ollama(messages) if model_choice == "ollama" else _create_completion_groq(messages)

def _create_completion_ollama(messages: list) -> tuple:
    """
    Crea una completion utilizando el modelo Ollama.
    
    Args:
        messages (list): Lista de mensajes para la conversación.
    
    Returns:
        tuple: Una tupla conteniendo la respuesta JSON y el código de estado HTTP.
    """
    print("DEBUG: Entering _create_completion_ollama")
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="llama3")

    try:
        response = client.chat.completions.create(
            model=MODELS["ollama"],
            messages=messages,
            max_tokens=1000,
            stream=False,
            # response_format={"type": "json_object"}
        )
        print(f"DEBUG: Ollama raw response: {response}")
    except Exception as e:
        print(f"An error occurred with Ollama: {e}")
        return jsonify({"error": str(e)}), 500

    return _process_response(response)

def _create_completion_groq(messages: list) -> tuple:
    """
    Crea una completion utilizando el modelo Groq.
    
    Args:
        messages (list): Lista de mensajes para la conversación.
    
    Returns:
        tuple: Una tupla conteniendo la respuesta JSON y el código de estado HTTP.
    """
    print("DEBUG: Entering _create_completion_groq")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    try:
        response = client.chat.completions.create(
            model=MODELS["groq"],
            #  max_length = 200,
            messages=messages,
            max_tokens=500,
            stream=False,
            # response_format={"type": "json_object"}
        )
        print(f"DEBUG: Groq raw response: {response}")
    except Exception as e:
        print(f"An error occurred with Groq: {e}")
        return jsonify({"error": str(e)}), 500

    return _process_response(response)

def _process_response(response) -> tuple:
    """
    Procesa la respuesta del modelo y la convierte en un objeto JSON válido.
    
    Args:
        response: La respuesta del modelo.
    
    Returns:
        tuple: Una tupla conteniendo la respuesta JSON y el código de estado HTTP.
    """
    print("DEBUG: Entering _process_response")
    try:
        content = response.choices[0].message.content
        print(f"DEBUG: Raw content from model: {content}")
        return content, 200
    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500