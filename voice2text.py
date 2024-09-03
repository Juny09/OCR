import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
import os

st.title("Audio to Text with Processing")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "aiff", "ogg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    temp_audio_path = "temp_audio_file"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Always convert the audio file to WAV format
    audio_format = uploaded_file.name.split('.')[-1].lower()
    wav_audio_path = "temp_audio_file.wav"
    audio = AudioSegment.from_file(temp_audio_path, format=audio_format)
    audio.export(wav_audio_path, format="wav")

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_audio_path) as source:
            audio_data = recognizer.record(source)
        
        text = recognizer.recognize_google(audio_data)
        
        st.subheader("Transcribed Text")
        container = st.container()
        container.write(f"{text}")

    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except ValueError:
        st.error("Audio file could not be read; please check if the file is corrupted or in an unsupported format.")

    if text:
        nlp_model = pipeline("text-generation", model="gpt2")
        processed_text = nlp_model(text, max_new_tokens=50)[0]['generated_text']
        
        st.subheader("Processed Text")
        container = st.container()
        container.write(f"{processed_text}")
        
    # Clean up temporary files
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    if os.path.exists(wav_audio_path):
        os.remove(wav_audio_path)
