import sounddevice as sd
import numpy as np
import mlx_whisper
import requests
import asyncio
import edge_tts
import simpleaudio as sa
import os
import json
from colorama import init, Fore, Style
from pydub import AudioSegment

# Initialize colorama for colored text in terminal
init(autoreset=True)

# Load the MLX Whisper model for Spanish transcription
print(Fore.CYAN + "Cargando el modelo Whisper...")
model_path = '/Users/carsonmulligan/Desktop/guest_house/code/mlx-examples/whisper/mlx_models/whisper-small-spanish'

# Historial de conversación
conversation = []

def record_audio(fs=16000, duration=5):
    """Record audio from the microphone."""
    print(Fore.GREEN + "Berto está escuchando... (Presiona Ctrl+C para detener)")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(recording)
    except Exception as e:
        print(Fore.RED + f"Error al grabar audio: {e}")
        return None

def transcribe_audio(audio_data):
    """Transcribe the recorded audio using MLX Whisper."""
    print(Fore.CYAN + "Transcribiendo...")
    try:
        audio_data = audio_data.astype(np.float32)
        result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=model_path)["text"]
        text = result.strip()
        print(Fore.YELLOW + f"Tú: {text}")
        return text
    except Exception as e:
        print(Fore.RED + f"Error al transcribir audio: {e}")
        return ""

def get_ai_response(user_input):
    """Get a response from the AI (Berto) using Ollama."""
    conversation.append(f"Usuario: {user_input}")
    recent_conversation = conversation[-10:]
    system_prompt = ( """
        <Berto>
    <Descripcion>
        Berto es: Un taxista amigable de la Ciudad de México. 
        Puede mantener conversaciones sobre historia de México, astrofísica, relatividad general y cuestiones no resueltas en matemáticas.
        Le gusta mantener las cosas casuales y conversacionales, pero puede profundizar en temas más complejos si se le solicita.
    </Descripcion>
    <Preguntas>
        - ¿Qué opinas sobre moverse más rápido que la velocidad de la luz?
        - ¿Qué piensas de la relatividad del tiempo al acercarse a los agujeros negros?
        - ¿Te imaginas cómo sería mapear todas las transformaciones de la materia y la energía desde el Big Bang?
    </Preguntas>
</Berto>
"""
    )
    conversation_text = "\n".join(recent_conversation)
    prompt = f"{system_prompt}\n{conversation_text}\nBerto:"

    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'llama3.1',
        'prompt': prompt,
        'temperature': 0.7,
        'max_tokens': 150,
        'stop': ['Usuario:', 'Berto:']
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        response_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_data = json.loads(line)
                if 'response' in json_data:
                    response_text += json_data['response']
        response_text = response_text.strip()
        conversation.append(f"Berto: {response_text}")
        print(Fore.CYAN + f"Berto: {response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error al comunicarse con Ollama: {e}")
        return "Lo siento, tengo problemas para responder en este momento. El Ollama."
    except json.JSONDecodeError as e:
        print(Fore.RED + f"Error al analizar la respuesta JSON: {e}")
        return "Lo siento, tengo problemas para responder en este momento. El JSON."

async def save_audio(response_text):
    """Convert AI response to speech using Edge TTS."""
    voice = "es-MX-JorgeNeural"
    communicate = edge_tts.Communicate(text=response_text, voice=voice)
    await communicate.save("berto_response.mp3")
    print(Fore.GREEN + "Audio guardado exitosamente")

def speak_response(response_text):
    """Play the AI's spoken response."""
    try:
        asyncio.run(save_audio(response_text))
        
        if os.path.exists("berto_response.mp3") and os.path.getsize("berto_response.mp3") > 0:
            print(Fore.GREEN + "El archivo MP3 existe y tiene contenido")
            sound = AudioSegment.from_mp3("berto_response.mp3")
            sound.export("berto_response.wav", format="wav")
            
            if os.path.exists("berto_response.wav") and os.path.getsize("berto_response.wav") > 0:
                print(Fore.GREEN + "El archivo WAV existe y tiene contenido")
                wave_obj = sa.WaveObject.from_wave_file("berto_response.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
            else:
                print(Fore.RED + "Error: El archivo WAV está vacío o no existe")
        else:
            print(Fore.RED + "Error: El archivo MP3 está vacío o no existe")
    except Exception as e:
        print(Fore.RED + f"Error al reproducir audio: {e}")

def test_ollama_connection():
    """Test the connection to Ollama."""
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'llama3.1',
        'prompt': 'Di "Hola, estoy funcionando" en español',
        'temperature': 0.7,
        'max_tokens': 50
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.text
        print(Fore.GREEN + f"Prueba de Ollama exitosa. Respuesta: {result}")
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error conectando a Ollama: {e}")

def main():
    """Main loop for the CLI interaction."""
    print(Fore.CYAN + "¡Iniciando conversación con Berto!")
    
    # Test Ollama connection
    test_ollama_connection()
    
    initial_prompt = (
        "Hola Chamo, soy Don Berto de la Isla de Pascua, Cómo va la vida, mi compa?"
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
