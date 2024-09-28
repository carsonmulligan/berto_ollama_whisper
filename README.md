# Fully Local Taxi-Driver Language Assistant

## Overview

This project provides a fully local language assistant called Berto. Berto can listen to your audio, transcribe it using Whisper, and then interact with you in Spanish using an AI model served locally via Ollama. It also has text-to-speech capabilities to provide audio responses.

## Features

- Local transcription of spoken language using Whisper.
- Interaction with an AI model (Llama 2 uncensored) served locally via Ollama.
- Conversation options, including following up on questions related to science, history, and politics.
- Text-to-speech responses using Edge TTS.
- Audio playback of Berto's responses.

## Setup

### 1. Install Dependencies

First, you need to install the required packages. You can do this by running the following command to install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Download and Install Ollama

This project requires Ollama to serve the AI model (`llama2-uncensored`). You can download Ollama from the [Ollama website](https://ollama.com/) and install it on your local machine.

Once installed, you need to download the model:

```bash
ollama run llama2-uncensored
```

Make sure Ollama is running on `localhost:11434` to handle the requests.

### 3. Run the Script

To run the assistant, execute the following script:

```bash
python bertosito_chat.py
```

This will start a conversation with Berto, who will transcribe your spoken audio and respond based on the conversation using the AI model hosted on Ollama.

### How It Works

- **Recording Audio:** Berto listens to your voice and transcribes it using Whisper.
- **Generating Responses:** It sends your transcribed text to the AI model and generates a response.
- **Text-to-Speech:** Berto will convert the generated response to speech and play it back.
- **Conversation Options:** The assistant presents multiple conversation options, including asking questions and following up on prior responses.

## File Structure

- `bertosito_chat.py`: The main script to run Berto.
- `response.mp3` and `response.wav`: Audio files generated during interaction.
- `requirements.txt`: File containing all the necessary dependencies.

### Notes:

- Ensure that you have Ollama running locally and the required models downloaded before starting the script.
- The assistant only responds in Spanish and expects interactions in the same language.
- Make sure to run Ollama using the exact model name `llama2-uncensored`.

---

Enjoy interacting with Berto!
