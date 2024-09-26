import os
import sys
import shutil
import tempfile
import json
import requests
import threading
import whisper
import sounddevice as sd
import soundfile as sf
import pyttsx3

# Constants
SAMPLE_RATE = 16000  # Hertz
CHANNELS = 1
BLOCK_DURATION = 10  # Seconds
EXIT_COMMANDS = ['exit', 'salir', 'adiós', 'bye']

def setup_tts_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Select a Spanish voice if available
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'es_' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break
    return engine

def text_to_speech(text, engine):
    engine.say(text)
    engine.runAndWait()

def record_audio(filename, duration):
    print("Berto está escuchando... (Presiona Ctrl+C para detener)")
    try:
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
        sd.wait()  # Wait until recording is finished
        sf.write(filename, recording, SAMPLE_RATE)
    except Exception as e:
        print(f"Error durante la grabación: {e}")
        raise

def transcribe_audio(filename):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(filename, language='es')
        return result['text']
    except Exception as e:
        print(f"Error durante la transcripción: {e}")
        return ""

def generate_response(prompt):
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 150
    }
    try:
        response = requests.post('http://localhost:11434/generate', headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error al generar la respuesta: {e}")
        return "Lo siento, no puedo responder en este momento."

def greet_user(engine):
    greeting = "¡Órale! ¿Cómo te va, amigo?"
    print(f"Berto: {greeting}")
    text_to_speech(greeting, engine)

def main():
    engine = setup_tts_engine()
    greet_user(engine)
    
    while True:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_filename = temp_audio.name
            
            record_audio(audio_filename, BLOCK_DURATION)
            user_input = transcribe_audio(audio_filename)
            os.unlink(audio_filename)  # Remove the temp audio file
            
            if not user_input.strip():
                print("No te escuché bien, ¿puedes repetirlo?")
                continue

            print(f"Tú: {user_input}")
            
            if any(exit_command.lower() in user_input.lower() for exit_command in EXIT_COMMANDS):
                farewell = "¡Hasta luego! Cuídate mucho."
                print(f"Berto: {farewell}")
                text_to_speech(farewell, engine)
                break
            
            response = generate_response(user_input)
            print(f"Berto: {response}")
            text_to_speech(response, engine)
        
        except KeyboardInterrupt:
            print("\nInteracción terminada por el usuario.")
            break
        except Exception as e:
            print(f"Se produjo un error: {e}")
            continue

if __name__ == "__main__":
    main()
