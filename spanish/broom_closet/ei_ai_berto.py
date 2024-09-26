"""
# Setup Instructions for EI Berto

1. Ensure you're in the correct Conda environment:
    ```bash
    conda activate berto
    ```

2. Install the necessary packages:
    ```bash
    pip uninstall whisper
    pip install git+https://github.com/openai/whisper.git
    pip install sounddevice numpy gTTS
    brew install mpg321  # On macOS, install mpg321 to play the text-to-speech audio
    ```

3. Pull the LLaMA model for Ollama:
    ```bash
    ollama pull llama2
    ```

4. Once the setup is complete, run the script:
    ```bash
    python3 ei_berto.py
    ```

# How to Use

- The script will greet you in Spanish.
- You can respond with voice input, and Whisper will transcribe it in real time.
- EI Berto (powered by Ollama) will respond in Spanish.
- Say "adiós" to end the conversation.
"""

import whisper
import subprocess
import os
import sounddevice as sd
import numpy as np
import gtts
import tempfile

# Load Whisper model
model = whisper.load_model("small")

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']

# Function to have a conversation with EI Berto using Ollama
def converse_with_ei_berto(prompt):
    # Call Ollama to generate a response
    command = f"ollama chat llama2 --prompt '{prompt}'"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    response, _ = process.communicate()
    
    return response.decode('utf-8')

# Function to generate a speech file from text
def speak_text(text):
    tts = gtts.gTTS(text, lang='es')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        os.system(f"mpg321 {f.name} >/dev/null 2>&1")  # Suppress mpg321 logs

# Function to handle transcription and trigger the conversation
def respond_to_transcription():
    while True:
        print("Tú: ", end="", flush=True)
        user_input = record_audio()
        print(f"{user_input}")  # Show your transcribed text

        if "adios" in user_input.lower():
            print("EI Berto: Adiós!")
            speak_text("Adiós!")
            break

        ei_berto_response = converse_with_ei_berto(user_input)
        print(f"EI Berto: {ei_berto_response}")  # Show Berto's response
        speak_text(ei_berto_response)

# Function to record audio from the user
def record_audio(duration=5, sample_rate=16000):
    print("\nListening... (speak now)")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    np.save(audio_file, recording)
    
    # Convert recorded audio to text using Whisper
    transcription = transcribe_audio(audio_file)
    return transcription

# Main function for command-line interface
def cli():
    # Greet the user with a speech
    greet_text = "¡Bienvenido a EI Berto! Diga algo para comenzar a charlar en español."
    print(greet_text)
    speak_text(greet_text)

    # Start the interaction
    respond_to_transcription()

if __name__ == "__main__":
    cli()
