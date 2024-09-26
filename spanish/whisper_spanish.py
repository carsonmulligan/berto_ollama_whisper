# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="clu-ling/whisper-small-spanish")  
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("clu-ling/whisper-small-spanish")
model = AutoModelForSpeechSeq2Seq.from_pretrained("clu-ling/whisper-small-spanish")