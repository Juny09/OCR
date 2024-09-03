import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
import soundfile as sf
from pydub import AudioSegment
import os

# Specify the path to ffmpeg if not in PATH
AudioSegment.converter = r"ffmpeg/bin/ffmpeg.exe"  # Update this path as needed

st.title("Audio to Text with LLM")

# Load the Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "aiff", "ogg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert the audio file to WAV format if necessary
    audio_format = uploaded_file.name.split('.')[-1]
    if audio_format != 'wav':
        audio = AudioSegment.from_file("temp_audio_file", format=audio_format)
        audio.export("temp_audio_file.wav", format="wav")
        audio_file_path = "temp_audio_file.wav"
    else:
        audio_file_path = "temp_audio_file"

    # Resample the audio to 16 kHz
    audio = AudioSegment.from_file(audio_file_path)
    audio = audio.set_frame_rate(16000)
    audio.export("resampled_audio.wav", format="wav")
    audio_file_path = "resampled_audio.wav"

    # Load the resampled audio file
    speech, sample_rate = sf.read(audio_file_path)

    # Prepare the audio input for the model
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode the output
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0]

    st.write("Transcribed Text")
    st.write(f"{text}")

    if text:
        nlp_model = pipeline("text-generation", model="gpt2")
        processed_text = nlp_model(text, max_new_tokens=50)[0]['generated_text']

        st.write("Processed Text")
        st.write(f"{processed_text}")

    # Clean up temporary files
    os.remove(audio_file_path)
    os.remove("temp_audio_file")
