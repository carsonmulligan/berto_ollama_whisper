import sounddevice as sd
import numpy as np
import whisper
import requests
import asyncio
import edge_tts
import simpleaudio as sa
import os
import json
from colorama import init, Fore
from pydub import AudioSegment
from scipy.io.wavfile import write

# Initialize colorama for colored text in terminal
init(autoreset=True)

# Load the Whisper model
print(Fore.CYAN + "Cargando el modelo Whisper...")
model = whisper.load_model('base')

# Conversation history
conversation = []

def record_audio(fs=16000):
    """Record audio from the microphone until silence is detected."""
    print(Fore.GREEN + "Berto está escuchando... (Presiona Ctrl+C para detener)")
    try:
        duration = 0.5  # Chunk duration in seconds
        silence_threshold = 0.01  # Silence threshold
        max_silence_duration = 1.5  # Max duration of silence before stopping

        recording = []
        silence_duration = 0

        while True:
            audio_chunk = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            rms = np.sqrt(np.mean(audio_chunk**2))
            recording.append(audio_chunk)
            if rms < silence_threshold:
                silence_duration += duration
            else:
                silence_duration = 0
            if silence_duration > max_silence_duration:
                break
        full_recording = np.concatenate(recording, axis=0)
        return np.squeeze(full_recording)
    except Exception as e:
        print(Fore.RED + f"Error al grabar audio: {e}")
        return None

def transcribe_audio(audio_data):
    """Transcribe the recorded audio using Whisper."""
    print(Fore.CYAN + "Transcribiendo...")
    try:
        # Save audio_data to a temporary WAV file
        write("temp.wav", 16000, audio_data)
        # Transcribe using Whisper
        result = model.transcribe("temp.wav", language='es')
        text = result['text'].strip()
        print(Fore.YELLOW + f"Tú: {text}")
        return text
    except Exception as e:
        print(Fore.RED + f"Error al transcribir audio: {e}")
        return ""

def get_ai_response(user_input):
    """Get a response from Berto using Ollama."""
    conversation.append(f"Usuario: {user_input}")
    recent_conversation = conversation[-10:]

    # Check for 'berto stop' command
    if 'berto stop' in user_input.lower():
        print(Fore.CYAN + "Berto ha dejado de hablar y está escuchando...")
        conversation.clear()
        return ""

    system_prompt = """
<Berto>
<Descripcion>
Berto es un tutor amigable que ayuda a los usuarios a mejorar sus habilidades para hacer preguntas en español.
Corrige amablemente las preguntas si es necesario y ofrece sugerencias para mejorarlas.
Luego, responde de manera informativa y clara.
</Descripcion>
</Berto>
"""

    conversation_text = "\n".join(recent_conversation)
    prompt = f"{system_prompt}\n{conversation_text}\nBerto:"

    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'llama3.1',
        'prompt': prompt,
        'temperature': 0.7,
        'max_tokens': 200,
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
        return "Lo siento, tengo problemas para responder en este momento."
    except json.JSONDecodeError as e:
        print(Fore.RED + f"Error al analizar la respuesta JSON: {e}")
        return "Lo siento, tengo problemas para responder en este momento."

async def save_audio(response_text):
    """Convert Berto's response to speech using Edge TTS."""
    voice = "es-MX-JorgeNeural"
    communicate = edge_tts.Communicate(text=response_text, voice=voice)
    await communicate.save("berto_response.mp3")
    print(Fore.GREEN + "Audio guardado exitosamente")

def speak_response(response_text):
    """Play Berto's spoken response."""
    if not response_text:
        return
    try:
        asyncio.run(save_audio(response_text))

        if os.path.exists("berto_response.mp3") and os.path.getsize("berto_response.mp3") > 0:
            sound = AudioSegment.from_mp3("berto_response.mp3")
            sound.export("berto_response.wav", format="wav")

            if os.path.exists("berto_response.wav") and os.path.getsize("berto_response.wav") > 0:
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
        print(Fore.GREEN + "Prueba de Ollama exitosa.")
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error conectando a Ollama: {e}")

def main():
    """Main loop for the CLI interaction."""
    print(Fore.CYAN + "¡Iniciando conversación con Berto!")

    # Test Ollama connection
    test_ollama_connection()

    initial_prompt = (
        "Hola, soy Berto. ¿De dónde eres? ¿Qué te trae por Perú?"
    )
    print(Fore.CYAN + f"Berto: {initial_prompt}")
    conversation.append(f"Berto: {initial_prompt}")
    speak_response(initial_prompt)

    while True:
        try:
            audio_data = record_audio()
            if audio_data is None or len(audio_data) == 0:
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
