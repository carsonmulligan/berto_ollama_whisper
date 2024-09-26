#!/bin/bash

# Navigate to the desired directory
cd /Users/carsonmulligan/ei_berto_spanish_chat/

# Create the berto.py script with the full code implementation
cat > berto.py << 'EOF'
import sounddevice as sd
import numpy as np
import whisper
import requests
import pyttsx3
import threading
import os
import sys

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model('base')

# Initialize pyttsx3
print("Initializing text-to-speech engine...")
engine = pyttsx3.init()

# Set Spanish voice
voices = engine.getProperty('voices')
for voice in voices:
    if 'es_' in voice.id or 'Spanish' in voice.name:
        engine.setProperty('voice', voice.id)
        break
else:
    print("Spanish voice not found. Using default voice.")
    # Optionally, you can exit the script if a Spanish voice is essential
    # sys.exit(1)

# Adjust speech rate if necessary
engine.setProperty('rate', 150)  # You can adjust the rate as needed

# Conversation history
conversation = []

def record_audio(fs=16000, duration=5):
    print("Grabando...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(recording)
    except Exception as e:
        print(f"Error al grabar audio: {e}")
        return None

def transcribe_audio(audio_data):
    print("Transcribiendo...")
    try:
        audio_data = audio_data.astype(np.float32)
        result = model.transcribe(audio_data, language='es')
        text = result['text'].strip()
        print(f"Tú: {text}")
        return text
    except Exception as e:
        print(f"Error al transcribir audio: {e}")
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
        'model': 'llama3.1',
        'prompt': prompt
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        response_text = response.json()['choices'][0]['text'].strip()
        conversation.append(f"Berto: {response_text}")
        print(f"Berto: {response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error al comunicarse con Ollama: {e}")
        return "Lo siento, tengo problemas para responder en este momento."

def speak_response(response_text):
    try:
        engine.say(response_text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error al reproducir audio: {e}")

def main():
    print("¡Iniciando conversación con Berto!")
    initial_prompt = (
        "¡Qué onda, mano! Soy Berto, tu compa taxista de la CDMX. "
        "¿Cómo te va hoy?"
    )
    print(f"Berto: {initial_prompt}")
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
            print("\nConversación terminada.")
            break
        except Exception as e:
            print(f"Ha ocurrido un error: {e}")
            break

if __name__ == "__main__":
    main()
EOF

# Create and activate the conda environment
echo "Creating and activating conda environment..."
conda create -n berto_env python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate berto_env

# Install required Python packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper
pip install pyttsx3
pip install sounddevice soundfile numpy requests

# Install pyobjc for pyttsx3 on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing pyobjc for macOS..."
    pip install pyobjc
fi

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5  # Wait for Ollama to start
else
    echo "Ollama is already running."
fi

# Pull the llama3.1 model if not already available
if ! ollama list | grep -q "llama3.1"; then
    echo "Pulling llama3.1 model..."
    ollama pull llama3.1
else
    echo "llama3.1 model is already available."
fi

# Run berto.py
echo "Running berto.py..."
python berto.py
