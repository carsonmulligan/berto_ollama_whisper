import whisper
import subprocess
import os
import sounddevice as sd
import numpy as np
import gtts
import tempfile
from scipy.io.wavfile import write

# Cargar el modelo Whisper con weights_only=True
model = whisper.load_model("small", weights_only=True)

# Función para transcribir audio usando Whisper
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']

# Función para mantener una conversación con EI Berto usando Ollama
def converse_with_ei_berto(prompt):
    # Llamar a Ollama para generar una respuesta
    command = f"ollama run llama2 '{prompt}'"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    response, error = process.communicate()
    
    if process.returncode != 0:
        print(f"Error al ejecutar Ollama: {error.decode('utf-8')}")
        return ""
    
    return response.decode('utf-8').strip()

# Función para generar un archivo de voz desde texto
def speak_text(text):
    if not text:
        print("No hay texto para hablar.")
        return
    tts = gtts.gTTS(text, lang='es')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        os.system(f"mpg321 {f.name} >/dev/null 2>&1")  # Suprimir logs de mpg321

# Función para manejar la transcripción y desencadenar la conversación
def respond_to_transcription():
    while True:
        print("Tú: ", end="", flush=True)
        user_input = record_audio()
        print(f"{user_input}")  # Mostrar el texto transcrito

        if "adios" in user_input.lower():
            print("EI Berto: Adiós!")
            speak_text("Adiós!")
            break

        ei_berto_response = converse_with_ei_berto(user_input)
        
        if not ei_berto_response:
            print("EI Berto: Lo siento, no pude procesar tu solicitud.")
            speak_text("Lo siento, no pude procesar tu solicitud.")
            continue

        print(f"EI Berto: {ei_berto_response}")  # Mostrar la respuesta de Berto
        speak_text(ei_berto_response)

# Función para grabar audio del usuario
def record_audio(duration=5, sample_rate=16000):
    print("\nEscuchando... (habla ahora)")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    # Guardar la grabación en un archivo WAV válido usando scipy
    audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    write(audio_file, sample_rate, recording)  # Escribir archivo WAV

    # Convertir audio grabado a texto usando Whisper
    transcription = transcribe_audio(audio_file)
    return transcription

# Función principal para la interfaz de línea de comandos
def cli():
    # Saludar al usuario con un discurso
    greet_text = "¡Bienvenido a EI Berto! Diga algo para comenzar a charlar en español."
    print(greet_text)
    speak_text(greet_text)

    # Iniciar la interacción
    respond_to_transcription()

if __name__ == "__main__":
    cli()
