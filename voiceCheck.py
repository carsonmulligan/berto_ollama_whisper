import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

# for voice in voices:
#     print(f"Voice ID: {voice.id}, Name: {voice.name}, Languages: {voice.languages}")


# # Voice ID: com.apple.eloquence.es-MX.Grandma, Name: Grandma (Spanish (Mexico)), Languages: ['es_MX']
# # Voice ID: com.apple.eloquence.es-MX.Grandpa, Name: Grandpa (Spanish (Mexico)), Languages: ['es_MX']
# ... existing code ...

# Set the voice to Grandpa
for voice in voices:
    if voice.id == "com.apple.eloquence.es-MX.Shelley":
        engine.setProperty('voice', voice.id)
        break

# Test the voice
engine.say("Hola, soy el abuelo.")
engine.runAndWait()

# ... existing code ...