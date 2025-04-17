import os
from vosk import Model, KaldiRecognizer
import wave

# Load the model
model_path = "/home/mona/voiceCommand/vosk-model-small-en-us-0.15"
model = Model(model_path)

# Open the audio file
audio_path = "/home/mona/voiceCommand/test.wav"
wf = wave.open(audio_path, "rb")

# Set up the recognizer
rec = KaldiRecognizer(model, wf.getframerate())

# Recognize speech from the audio file
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())

# Output final result
print(rec.FinalResult())
