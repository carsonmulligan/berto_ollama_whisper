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

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model('base')

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
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("Raw response:", response.text)
        # Handle potential extra data
        json_text = response.text.strip().split('\n')[0]
        response_json = json.loads(json_text)
        response_text = response_json['choices'][0]['text'].strip()
        conversation.append(f"Berto: {response_text}")
        print(f"Berto: {response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error al comunicarse con Ollama: {e}")
        return "Lo siento, tengo problemas para responder en este momento."
    except ValueError as e:
        print(f"Error al analizar la respuesta JSON: {e}")
        print("Respuesta recibida:", response.text)
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
