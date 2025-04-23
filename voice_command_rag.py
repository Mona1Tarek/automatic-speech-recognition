import os
import pyaudio
import json
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# -------------------------
# Step 1: Load Vosk STT
# -------------------------

model_path = "/home/mona/voiceCommand/vosk-model-small-en-us-0.15"
vosk_model = Model(model_path)
samplerate = 16000
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=samplerate,
                input=True,
                input_device_index=None,  # Optional: specify if needed
                frames_per_buffer=1024)

stream.start_stream()
rec = KaldiRecognizer(vosk_model, samplerate)

print(" Say something...")
  # Add this line


# -------------------------
# Step 2: Define commands
# -------------------------

commands = [
    {"id": "1", "text": "lock the doors", "action": "lock_the_doors"},
    {"id": "2", "text": "unlock the doors", "action": "unlock_the_doors"},
    {"id": "3", "text": "stop the car", "action": "stop_the_car"},
    {"id": "4", "text": "turn on the headlights", "action": "turn_on_the_headlights"},
    {"id": "5", "text": "turn on the ac", "action": "turn_on_the_ac"},

]

# -------------------------
# Step 3: Load ChromaDB and embed commands
# -------------------------

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="voice_commands", embedding_function=embedding_function)

for command in commands:
    collection.add(
        documents=[command["text"]],
        ids=[command["id"]],
        metadatas=[{"action": command["action"]}]
    )

# -------------------------
# Step 4: Real-time Recognition + Retrieval
# -------------------------

while True:
    data = stream.read(4000)
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        user_text = result["text"]

        if user_text.strip() == "":
            continue

        print(f"\n Recognized: {user_text}")

        # Retrieve closest match from ChromaDB
        results = collection.query(query_texts=[user_text], n_results=1)
        matched_text = results['documents'][0][0]
        matched_action = results['metadatas'][0][0]['action']

        print(f" Matched Command: {matched_text}")
        print(f"Action to perform: {matched_action}")

    # Optional: show live partials
    partial_result = json.loads(rec.PartialResult())
    if "partial" in partial_result and partial_result["partial"]:
        print(f" Listening: {partial_result['partial']}", end="\r")
