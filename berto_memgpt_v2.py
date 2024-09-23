import sounddevice as sd
import numpy as np
import whisper
import asyncio
import edge_tts
import simpleaudio as sa
import os
from colorama import init, Fore
from pydub import AudioSegment
from scipy.io.wavfile import write
from memgpt.agent import Agent

# Initialize colorama for colored text in terminal
init(autoreset=True)

# Load the Whisper model
print(Fore.CYAN + "Cargando el modelo Whisper...")
model = whisper.load_model('medium', weights_only=True)  # Set weights_only=True to avoid security issues

# Initialize MemGPT Agent
agent = Agent(
    model_endpoint='http://localhost:11434',  # Ollama endpoint for MemGPT
    model_backend='ollama',
    context_window=2048,  # Adjust based on your model's context window
    system_prompt="""
    Solamente habla en español. Eres Berto, un tutor amigable y experto en ciencias y en la historia de América Latina. Profesor de la lengua española.
    """
)

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

def get_ai_response(user_input):
    """Get a response from Berto using MemGPT."""
    # Check for 'berto stop' command
    if 'berto stop' in user_input.lower():
        print(Fore.CYAN + "Berto ha dejado de hablar y está escuchando...")
        agent.reset()  # Reset MemGPT agent's memory
        return ""

    try:
        # Send user input to MemGPT agent
        agent.user_message(user_input)

        # Generate the response
        berto_response = agent.generate_reply()

        print(Fore.BLUE + f"Respuesta del modelo:\n{berto_response}\n")
        print(Fore.CYAN + f"Berto: {berto_response}")
        return berto_response
    except Exception as e:
        print(Fore.RED + f"Error al comunicarse con MemGPT: {e}")
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

def generate_question(topic):
    """Generate a question about the given topic using MemGPT."""
    # Create a prompt to generate a question
    question_prompt = f"Por favor, proporciona una pregunta breve y clara sobre {topic}."

    try:
        # Send the prompt to the agent
        agent.user_message(question_prompt)

        # Generate the question
        generated_question = agent.generate_reply()

        return generated_question.strip()
    except Exception as e:
        print(Fore.RED + f"Error al generar pregunta con MemGPT: {e}")
        return None

def generate_follow_up_question(previous_response):
    """Generate a follow-up question about the given response using MemGPT."""
    # Create a prompt to generate a follow-up question
    follow_up_prompt = f"Por favor, proporciona una pregunta de seguimiento sobre: \"{previous_response}\"."

    try:
        # Send the prompt to the agent
        agent.user_message(follow_up_prompt)

        # Generate the follow-up question
        follow_up_question = agent.generate_reply()

        return follow_up_question.strip()
    except Exception as e:
        print(Fore.RED + f"Error al generar pregunta de seguimiento con MemGPT: {e}")
        return None

def present_cli_options():
    """Present CLI options to the user."""
    options = {
        '1': "Responder a Berto con tus propias palabras.",
        '2': "Hacer una pregunta de seguimiento sobre la respuesta anterior de Berto.",
        '3': "Escuchar una pregunta sobre ciencia.",
        '4': "Escuchar una pregunta sobre la historia de un país latinoamericano."
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
    elif choice == '2':
        # Option 2: Ask a follow-up question about Berto's previous response
        previous_responses = agent.get_conversation_history()
        if previous_responses:
            # Get the last response from Berto
            last_response = None
            for message in reversed(previous_responses):
                if message['role'] == 'assistant':
                    last_response = message['content']
                    break

            if last_response:
                print(Fore.GREEN + f"Generando una pregunta de seguimiento sobre la respuesta anterior de Berto: {last_response}")
                follow_up_question = generate_follow_up_question(last_response)
                if follow_up_question:
                    print(Fore.YELLOW + f"Tú (pregunta de seguimiento): {follow_up_question}")
                    speak_response(follow_up_question, voice='es-MX-DaliaNeural')  # Different voice for the user
                    return follow_up_question
                else:
                    print(Fore.RED + "No se pudo generar una pregunta de seguimiento.")
                    return None
            else:
                print(Fore.RED + "No hay respuesta anterior de Berto para hacer una pregunta de seguimiento.")
                return None
        else:
            print(Fore.RED + "No hay conversación previa.")
            return None
    else:
        # Option 3 or 4: Generate a question about science or history
        if choice == '3':
            topic = 'ciencia'
        elif choice == '4':
            topic = 'la historia de un país latinoamericano'

        generated_question = generate_question(topic)
        if generated_question:
            print(Fore.YELLOW + f"Tú (opción {choice}): {generated_question}")
            speak_response(generated_question, voice='es-MX-DaliaNeural')  # Different voice for the user
            return generated_question
        else:
            print(Fore.RED + "No se pudo generar una pregunta.")
            return None

def main():
    """Main loop for the CLI interaction."""
    print(Fore.CYAN + "¡Iniciando conversación con Berto!")

    # Initial greeting from Berto
    initial_prompt = (
        "Hola, bienvenido a este chat. Soy Berto y hablo en español."
    )
    print(Fore.CYAN + f"Berto: {initial_prompt}")
    speak_response(initial_prompt)

    # Start the conversation in MemGPT
    agent.user_message(initial_prompt)

    while True:
        try:
            # Present CLI options
            user_text = present_cli_options()
            if user_text:
                print(Fore.GREEN + f"Texto del usuario: {user_text}")
                berto_response = get_ai_response(user_text)
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
