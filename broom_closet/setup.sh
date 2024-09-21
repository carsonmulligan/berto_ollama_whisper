#!/bin/bash

# Navigate to the desired directory
cd /Users/carsonmulligan/ei_berto_spanish_chat/

# Create the berto.py script with the full code implementation
cat > berto.py << 'EOF'
import sounddevice as sd
import numpy as np
import whisper
import requests
import asyncio
import edge_tts
import simpleaudio as sa
import threading
import os
import sys
import json
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load Whisper model
print(Fore.CYAN + "Cargando el modelo Whisper...")
model = whisper.load_model('base')

# Conversation history
conversation = []

def record_audio(fs=16000, duration=5):
    print(Fore.GREEN + "Berto está escuchando... (Presiona Ctrl+C para detener)")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(recording)
    except Exception as e:
        print(Fore.RED + f"Error al grabar audio: {e}")
        return None

def transcribe_audio(audio_data):
    print(Fore.CYAN + "Transcribiendo...")
    try:
        audio_data = audio_data.astype(np.float32)
        result = model.transcribe(audio_data, language='es')
        text = result['text'].strip()
        print(Fore.YELLOW + f"Tú: {text}")
        return text
    except Exception as e:
        print(Fore.RED + f"Error al transcribir audio: {e}")
        return ""

def get_ai_response(user_input):
    conversation.append(f"Usuario: {user_input}")
    recent_conversation = conversation[-10:]
    system_prompt = (
        "Eres Berto, un taxista amigable de la Ciudad de México. "
        "Hablas en jerga mexicana y mantienes tus respuestas cortas y conversacionales. "
        "Alientas a la conversación a avanzar haciendo preguntas o comentarios."
    )
    conversation_text = "\n".join(recent_conversation)
    prompt = f"{system_prompt}\n{conversation_text}\nBerto:"

    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'llama2',
        'prompt': prompt,
        'temperature': 0.7,
        'max_tokens': 150,
        'stop': ['Usuario:', 'Berto:']
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        # Process the streamed response
        response_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_data = json.loads(line)
                if 'token' in json_data:
                    token = json_data['token']
                    response_text += token['content']
        response_text = response_text.strip()
        conversation.append(f"Berto: {response_text}")
        print(Fore.CYAN + f"Berto: {response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error al comunicarse con Ollama: {e}")
        return "Lo siento, tengo problemas para responder en este momento."
    except ValueError as e:
        print(Fore.RED + f"Error al analizar la respuesta JSON: {e}")
        return "Lo siento, tengo problemas para responder en este momento."

def speak_response(response_text):
    try:
        voice = "es-MX-JorgeNeural"  # Mexican Spanish male voice
        communicate = edge_tts.Communicate(text=response_text, voice=voice)
        # Save audio to a file
        asyncio.run(communicate.save("berto_response.mp3"))
        # Play the audio
        wave_obj = sa.WaveObject.from_wave_file("berto_response.mp3")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(Fore.RED + f"Error al reproducir audio: {e}")

def main():
    print(Fore.CYAN + "¡Iniciando conversación con Berto!")
    initial_prompt = (
        "¡Qué onda, mano! Soy Berto, tu compa taxista de la CDMX. "
        "¿Cómo te va hoy?"
    )
    print(Fore.CYAN + f"Berto: {initial_prompt}")
    conversation.append(f"Berto: {initial_prompt}")
    speak_response(initial_prompt)

    while True:
        try:
            audio_data = record_audio()
            if audio_data is None:
                continue
            user_text = transcribe_audio(audio_data)
            if user_text == "":
                continue
            berto_response = get_ai_response(user_text)
            speak_response(berto_response)
        except KeyboardInterrupt:
            print(Fore.RED + "\nConversación terminada.")
            break
        except Exception as e:
            print(Fore.RED + f"Ha ocurrido un error: {e}")
            break

if __name__ == "__main__":
    main()
EOF

# Create and activate the conda environment
echo "Creando y activando el entorno conda..."
conda create -n berto_env python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate berto_env

# Install required Python packages
echo "Instalando los paquetes de Python necesarios..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper
pip install edge-tts
pip install simpleaudio
pip install sounddevice soundfile numpy requests colorama

# Install pyobjc for pyttsx3 on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Instalando pyobjc para macOS..."
    pip install pyobjc
fi

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Iniciando Ollama..."
    ollama serve &
    sleep 5  # Wait for Ollama to start
else
    echo "Ollama ya está en ejecución."
fi

# Pull the llama2 model if not already available
if ! ollama list | grep -q "llama2"; then
    echo "Descargando el modelo llama2..."
    ollama pull llama2
else
    echo "El modelo llama2 ya está disponible."
fi

# Run berto.py
echo "Ejecutando berto.py..."
python berto.py
