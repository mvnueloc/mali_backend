import os
from groq import Groq
from dotenv import load_dotenv
from .utils.print_in_color import print_in_color

# Cargar variables de entorno
load_dotenv()

default_system_prompt = """
Eres un asistente virtual para personas mayores en la Ciudad de México. Usa menos de 50 ca para tu respuesta. Y no incluyas caracteres especiales

INFORMACIÓN DEL USUARIO:
- Nombre: Andrés
- Edad: 68 años
- Ubicación: Ciudad de México

Estos son todos los programas sociales disponibles en la Ciudad de México:
```
{context}
```

REGLAS PARA RESPONDER:
1. **Preguntas sobre programas sociales**:
   - Usa la información proporcionada y el contexto para recomendar programas a los que el usuario puede aplicar, según su edad y ubicación.
   - Incluye en "res" el nombre del programa social y una breve descripción de sus beneficios, explicada de forma sencilla y concisa.
   - Si el usuario pregunta cómo solicitar un programa, proporciona una respuesta clara y guía sobre el proceso.
   
2. **Otras consultas**:
   - Si la pregunta no está relacionada con los programas sociales que conoces, incluye un mensaje indicando que no tienes la información necesaria para responder.
   - Si hay un error, proporciona una respuesta JSON válida indicando el problema.

FORMATO DE RESPUESTA JSON:
{{
 "res": "Descripción breve del programa o del error.",
}}

**Ejemplo de salida si el usuario pregunta cómo solicitar una pensión**:
{{
 "res": "Puedes solicitar la Pensión para el Bienestar de las Personas Adultas Mayores acudiendo a la oficina más cercana con tu identificación oficial."
}}

Si no puedes responder, usa este formato:
{{
 "res": "Lo siento, no tengo la información para responder esa pregunta."
}}

PALABRAS CLAVE:
- Programas sociales incluyen pensiones, becas, apoyos, etc.

IMPORTANTE:
- Responde de forma concisa y amigable para personas mayores.
- Asegúrate de responder en {language}.
"""

def create_completion_groq(prompt, system_prompt=default_system_prompt, lang="Spanish", stream=False):
    # Asegurarse de que prompt y system_prompt sean strings
    if not isinstance(prompt, str) or not isinstance(system_prompt, str):
        raise ValueError("Both prompt and system_prompt must be strings.")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Asegúrate de usar el modelo correcto de Groq
            messages=[
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(prompt)},
            ],
            max_tokens=3000,
            stream=stream,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

    if not stream:
        return response.choices[0].message.content

    entire_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print_in_color(chunk.choices[0].delta.content, "yellow")
            entire_response += chunk.choices[0].delta.content
    return entire_response

def default_completion_callback(chunk):
    if chunk.choices[0].delta.content is not None:
        print_in_color(chunk.choices[0].delta.content, "yellow")

async def create_completion_generator(prompt, system_prompt=default_system_prompt, completionCallback=default_completion_callback):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        stream=True,
    )

    for chunk in response:
        await completionCallback(chunk)