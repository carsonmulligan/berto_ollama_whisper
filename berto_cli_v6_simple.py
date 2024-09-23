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
model = whisper.load_model('medium')  # Use 'large' for even better accuracy

# Set the desired model (replace with your model)
selected_model = 'llama2-uncensored:latest'  # Replace with the exact model name from Ollama

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
        options = dict(language='es', fp16=False)  # Disable FP16
        result = model.transcribe("temp.wav", **options)
        text = result['text'].strip()
        print(Fore.YELLOW + f"Tú: {text}")
        os.remove("temp.wav")  # Clean up the temp file
        return text
    except Exception as e:
        print(Fore.RED + f"Error al transcribir audio: {e}")
        return ""

def get_ai_response(user_input, selected_model):
    """Get a response from Berto using Ollama."""
    # Check for 'berto stop' command
    if 'berto stop' in user_input.lower():
        print(Fore.CYAN + "Berto ha dejado de hablar y está escuchando...")
        conversation.clear()
        return ""

    # Append the latest interaction
    conversation.append(f"Usuario: {user_input}")

    # Limit the conversation history
    recent_conversation = conversation[-4:]

    # Exclude previous refusals
    filtered_conversation = [
        line for line in recent_conversation
        if not any(
            phrase in line.lower()
            for phrase in ["lo siento", "no puedo ayudar", "no puedo generar", "no puedo interactuar"]
        )
    ]

    conversation_text = "\n".join(filtered_conversation)
    prompt = f"{system_prompt}\n{conversation_text}\nBerto:"

    print(Fore.BLUE + f"Prompt enviado al modelo:\n{prompt}\n")

    url = 'http://localhost:11434/api/generate'
    data = {
        'model': selected_model,  # Use the selected model
        'prompt': prompt,
        'temperature': 0.5,
        'max_tokens': 300,
        'stop': ['Usuario:']
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

        # Check for refusal
        if "lo siento" in response_text.lower() and "no puedo" in response_text.lower():
            print(Fore.RED + "El modelo ha generado una respuesta de rechazo.")
            # Optionally, modify the prompt or notify the user
        else:
            conversation.append(f"Berto: {response_text}")

        print(Fore.BLUE + f"Respuesta del modelo:\n{response_text}\n")
        print(Fore.CYAN + f"Berto: {response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error al comunicarse con Ollama: {e}")
        return "Lo siento, tengo problemas para responder en este momento."
    except json.JSONDecodeError as e:
        print(Fore.RED + f"Error al analizar la respuesta JSON: {e}")
        return "Lo siento, tengo problemas para responder en este momento."

async def save_audio(text, filename, voice):
    """Convert text to speech using Edge TTS."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(filename)
    print(Fore.GREEN + f"Audio guardado exitosamente en {filename}")

def speak_response(response_text, voice='es-MX-JorgeNeural'):
    """Play the spoken response."""
    if not response_text:
        return
    try:
        filename_mp3 = "response.mp3"
        filename_wav = "response.wav"
        asyncio.run(save_audio(response_text, filename_mp3, voice))

        if os.path.exists(filename_mp3) and os.path.getsize(filename_mp3) > 0:
            sound = AudioSegment.from_mp3(filename_mp3)
            sound.export(filename_wav, format="wav")

            if os.path.exists(filename_wav) and os.path.getsize(filename_wav) > 0:
                wave_obj = sa.WaveObject.from_wave_file(filename_wav)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                os.remove(filename_mp3)
                os.remove(filename_wav)
            else:
                print(Fore.RED + "Error: El archivo WAV está vacío o no existe")
        else:
            print(Fore.RED + "Error: El archivo MP3 está vacío o no existe")
    except Exception as e:
        print(Fore.RED + f"Error al reproducir audio: {e}")

def generate_question(topic, selected_model):
    """Generate a question about the given topic using Ollama."""
    system_prompt = f"""
Eres un asistente que genera preguntas interesantes sobre {topic}.solamenta habla en español.
Por favor, proporciona una pregunta breve y clara sobre {topic}.solamenta habla en español.
"""
    data = {
        'model': selected_model,  # Use the selected model
        'prompt': system_prompt,
        'temperature': 0.7,
        'max_tokens': 50,
        'stop': ['\n']
    }
    headers = {'Content-Type': 'application/json'}
    url = 'http://localhost:11434/api/generate'

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        response_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_data = json.loads(line)
                if 'response' in json_data:
                    response_text += json_data['response']
        question = response_text.strip()
        return question
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error al generar pregunta con Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        print(Fore.RED + f"Error al analizar la respuesta JSON: {e}")
        return None

def present_cli_options(selected_model):
    """Present CLI options to the user."""
    options = {
        '1': "Responder a Berto con tus propias palabras.",
        '2': "Escuchar una pregunta sobre ciencia.",
        '3': "Escuchar una pregunta sobre la historia de un país latinoamericano."
    }

    print(Fore.MAGENTA + "\nPor favor, elige una opción:")
    for key, value in options.items():
        print(f"{key}. {value}")

    choice = input(Fore.MAGENTA + "Ingresa el número de tu elección: ").strip()

    if choice not in options:
        print(Fore.RED + "Opción no válida. Intenta de nuevo.")
        return None

    if choice == '1':
        # Option 1: Let the user respond in their own words, then record
        print(Fore.GREEN + "Por favor, responde a Berto con tus propias palabras.")
        audio_data = record_audio()
        if audio_data is None or len(audio_data) == 0:
            return None
        user_text = transcribe_audio(audio_data)
        return user_text
    else:
        # Option 2 or 3: Generate a question about science or history
        if choice == '2':
            # Generate a question about science
            topic = 'ciencia'
        elif choice == '3':
            # Generate a question about la historia de un país latinoamericano
            topic = 'la historia de un país latinoamericano'

        # Generate the question using AI
        generated_question = generate_question(topic, selected_model)
        if generated_question:
            # Read the question aloud in a different voice
            print(Fore.YELLOW + f"Tú (opción {choice}): {generated_question}")
            speak_response(generated_question, voice='es-MX-DaliaNeural')  # Different voice for the user
            return generated_question
        else:
            print(Fore.RED + "No se pudo generar una pregunta.")
            return None

def test_ollama_connection(selected_model):
    """Test the connection to Ollama."""
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': selected_model,
        'prompt': 'Di "Hola, estoy funcionando" en español',
        'temperature': 0.7,
        'max_tokens': 50
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print(Fore.GREEN + f"Prueba de Ollama exitosa con el modelo '{selected_model}'.")
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error conectando a Ollama con el modelo '{selected_model}': {e}")
        exit(1)

def main():
    """Main loop for the CLI interaction."""
    print(Fore.CYAN + "¡Iniciando conversación con Berto!")

    # Test Ollama connection
    test_ollama_connection(selected_model)

    # Define the system prompt globally
    global system_prompt
    system_prompt = """
solamenta habla en español. Eres Berto, un tutor amigable y experto en ciencias y en la historia de América Latina. profesor de la lingua española. solamenta habla en español.
"""

    initial_prompt = (
        "Puuuuuuuta madre, Que pedo wei? bienvenido a este chat - soy berto wei y yo solo habla en español"
    )
    print(Fore.CYAN + f"Berto: {initial_prompt}")
    conversation.append(f"Berto: {initial_prompt}")
    speak_response(initial_prompt)

    while True:
        try:
            # Present CLI options
            user_text = present_cli_options(selected_model)
            if user_text:
                berto_response = get_ai_response(user_text, selected_model)
                speak_response(berto_response)
            else:
                continue

        except KeyboardInterrupt:
            print(Fore.RED + "\nConversación terminada.")
            break
        except Exception as e:
            print(Fore.RED + f"Ha ocurrido un error: {e}")
            break

if __name__ == "__main__":
    main()
