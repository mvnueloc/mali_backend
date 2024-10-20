import os
import sounddevice as sd
import numpy as np
from src.completions import create_completion
from src.vectors import ChromaManager
from src.utils.print_in_color import print_in_color
from flask import Flask
from flask import request
from flask import jsonify
from typing import List
from pydantic import BaseModel
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

def get_system_prompt(context: str, language: str = "Spanish"):
   return """
Solo responde la pregunta específica con una o dos frases cortas, sin más detalles a menos que se te pidan explícitamente.

INFORMACIÓN DEL USUARIO:
- Nombre: Andrés
- Sexo: Masculino
- Edad: 68 años
- Ubicación: Ciudad de México

Estos son todos los programas sociales disponibles en la Ciudad de México:

{
  "solicitud_usuario": {
    "edad": 68,
    "genero": "hombre",
    "tipo_solicitud": "pension" 
  },
  "programas": [
    {
      "nombre": "Pensión Mujeres Bienestar",
      "tipo": "pension",
      "descripcion": "Pensión bimestral de 3,000 pesos para mujeres de 60-64 años",
      "inicio": "01/2025",
      "entidad": "Secretaría de Bienestar",
      "poblacion": {
        "genero": "mujer",
        "edad_min": 60,
        "edad_max": 64
      },
      "requisitos": ["INE", "Acta nacimiento", "CURP", "Comprobante domicilio", "Teléfono", "Formato bienestar"],
      "inscripcion": {
        "fechas": "07/10-30/11",
        "modulos": "Web oficial Secretaría de Bienestar"
      },
      "pago": {
        "monto": "3,000 bimestral",
        "metodo": "Tarjeta Banco Bienestar"
      },
      "contacto": {
        "web": "https://programasparaelbienestar.gob.mx/pension-mujeres-bienestar/",
        "social": "https://facebook.com/apoyosbienestar"
      }
    },
    {
      "nombre": "Pensión Bienestar Personas Adultas Mayores",
      "tipo": "pension",
      "descripcion": "Pensión bimestral de 6,000 pesos para mayores de 65 años",
      "inicio": "01/2025",
      "entidad": "Secretaría de Bienestar",
      "poblacion": {
        "genero": "indistinto",
        "edad_min": 65
      },
      "requisitos": ["INE", "Acta nacimiento", "CURP", "Comprobante domicilio", "Teléfono", "Formato bienestar"],
      "inscripcion": {
        "fechas": "07/10-30/11",
      },
      "pago": {
        "monto": "6,000 bimestral",
        "metodo": "Tarjeta Banco Bienestar"
      },
      "contacto": {
        "web": "https://programasparaelbienestar.gob.mx/pension-bienestar-adultos-mayores/",
        "social": "https://facebook.com/apoyosbienestar"
      }
    },
    {
      "nombre": "Salud Casa por Casa",
      "tipo": "salud",
      "descripcion": "Atención médica domiciliaria para mayores de 65 y discapacitados",
      "inicio": "02/2025",
      "entidad": "Secretaría de Bienestar",
      "poblacion": {
        "genero": "indistinto",
        "edad_min": 65
      },
      "requisitos": ["INE", "CURP", "Comprobante domicilio", "Cuestionario salud"],
      "inscripcion": {
        "fechas": "07/10-30/12/2024",
        "proceso": "Censo de salud y bienestar"
      },
      "servicios": ["Atención primaria", "Signos vitales", "Curaciones", "Estudios básicos", "Prescripción medicamentos", "Enlace atención superior"],
      "contacto": {
        "web": "https://programasparaelbienestar.gob.mx/salud-casa-por-casa/",
        "social": "https://facebook.com/apoyosbienestar"
      }
    }
  ]
}

```

REGLAS PARA RESPONDER:
1. **Preguntas sobre programas sociales**:
   - En "res" Usa la información proporcionada y el contexto para ayudar a resolver su duda, según su edad, sexo o nivel educativo que esta cursando. Se muy breve y consiso. Si no te pido requisitos para realizar el proceso no los proporciones.
    - Si te pido requisitos, hazlo en un formato claro y simple, sin agregar detalles innecesarios.
   - Dale formato a la respuesta usando markdown como titulos, saltos de linea, negritas y si te pido requisitos, inclúyelos en bulletpoints.
   - Si te pido informacion de como hacer el proceso o donde puedo registrarme, no me digas que en la pagina web oficial, dame el link con formato markdown de la pagina oficial.
   
2. **Otras consultas**:
   - Si la pregunta no está relacionada con los programas sociales que conoces, incluye un mensaje indicando que no tienes la información necesaria para responder.
   - Si hay un error, proporciona una respuesta JSON válida indicando el problema.

FORMATO DE RESPUESTA JSON:
"Descripción breve del programa o del error."

**Ejemplo de salida si el usuario pregunta cómo solicitar una pensión**:
"**Pensión Bienestar Personas Adultas Mayores**: Esta es la beca que se adecua a tu perfil. Si necesitas ayuda para solicitarla, por favor, házmelo saber."

Si no puedes responder, usa este formato:
"Lo siento, no tengo la información para responder esa pregunta."

PALABRAS CLAVE:
- Programas sociales incluyen pensiones, becas, apoyos, etc.

IMPORTANTE:
- Cuando uses salto de línea, usa dos saltos de línea para que se vea correctamente en la respuesta final.

"""


chroma = ChromaManager()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    if not client.api_key:
        return jsonify({"error": "GROQ_API_KEY no está configurada"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]

    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        return jsonify({"error": "Tipo de archivo no soportado"}), 400

    if request.content_length > 10 * 1024 * 1024:
        return jsonify({"error": "El archivo es demasiado grande"}), 400

    try:
        file_content = audio_file.read()
        
        from io import BytesIO
        file_obj = BytesIO(file_content)
        
        transcription = client.audio.transcriptions.create(
            file=(audio_file.filename, file_obj, 'audio/wav'),
            model="whisper-large-v3-turbo"
        )
        return jsonify({"text": transcription.text}), 200
    except Exception as e:
        print(f"Error detallado: {str(e)}")  
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500

@app.route("/conversation_text", methods=["POST"])
def conversation_text():
    try:
        data = request.json
        user_message = data["user_message"]
        model_choice = data["model_choice"]
        messages = data["messages"]
        print(f"messages: {messages}")  
        print(user_message, model_choice)
        context = chroma.get_context_from_query(query_text=user_message, n_results=5)
        completion = create_completion(prompt=user_message,
                                              system_prompt=get_system_prompt(context, "autodetect"), model_choice=model_choice)
        return completion
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def root():
    return jsonify({"message": "mali run ..."})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=8080)