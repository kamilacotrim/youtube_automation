import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if transcript:
            return '\n'.join([entry['text'] for entry in transcript])
        else:
            return None
    except Exception as e:
        st.error(f"Error getting transcript: {e}")
        return None

def translate_text(transcript):
    model_name = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    max_length = 512
    chunks = [transcript[i:i + max_length] for i in range(0, len(transcript), max_length)]

    translations = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt')
        translation_ids = model.generate(**inputs)
        translation = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
        translations.append(translation)

    final_translation = ' '.join(translations)
    return final_translation

def download_video(selected_video, save_path=".", get_transcript=False, create_audio=False, display_video=False):
    st.write(f"Downloading: {selected_video.title}...")
    
    stream = selected_video.streams.filter(file_extension="mp4").first()
    video_filename = f"{selected_video.title.replace(' ', '_')}.mp4"
    stream.download(output_path=save_path, filename=video_filename)

    st.success("Download complete!")

    if display_video:
        st.subheader("Video:")
        with open(os.path.join(save_path, video_filename), "rb") as f:
            video_content = f.read()
        st.video(video_content)

    transcript = ""

    if get_transcript:
        st.subheader("Transcript:")
        transcript = get_youtube_transcript(selected_video.video_id)
        st.write("\nOriginal Transcript:")
        st.write(transcript)

    if create_audio and transcript:
        punctuation_model = PunctuationModel()
        corrected_transcript = punctuation_model.restore_punctuation(transcript)
        st.write("\nCorrected Transcript:")
        st.write(corrected_transcript)

        translated_transcript = translate_text(corrected_transcript)
        st.write("\nTranslated Transcript:")
        st.write(translated_transcript)

        # Display in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original Transcript")
            st.write(transcript)
        with col2:
            st.subheader("Corrected Transcript")
            st.write(corrected_transcript)
        with col3:
            st.subheader("Translated Transcript")
            st.write(translated_transcript)

        audio_filename = f"{selected_video.title.replace(' ', '_')}_translated_audio.mp3"
        tts = gTTS(translated_transcript, lang='en', slow=False)
        tts.save(os.path.join(save_path, audio_filename))
        st.audio(audio_filename, format='audio/mp3')

if __name__ == "__main__":
    st.title("YouTube Video Downloader and Transcript App")

    option = st.radio("Choose an option", ["Search for videos", "Provide list of URLs"])

    if option == "Search for videos":
        query = st.text_input("Enter your search query:")
        save_path = st.text_input("Enter the path to save the video (default is current directory):", ".")
        
        get_transcript = st.checkbox("Get Transcript")
        create_audio = st.checkbox("Create audio from translated transcript?")
        display_video = st.checkbox("Display video?")

        if st.button("Search and Download") and query and save_path:
            results = YouTube(f"https://www.youtube.com/results?search_query={query}")

            st.header("Search Results:")
            
            for i, video in enumerate(results.streams.filter(type="video").all(), start=1):
                st.write(f"{i}. {video.title}")

            choice = st.number_input("Enter the number of the video you want to download:", min_value=1, max_value=len(results))
            selected_video = results.streams.filter(type="video").all()[int(choice) - 1]

            download_video(selected_video, save_path, get_transcript, create_audio, display_video)

    elif option == "Provide list of URLs":
        url_list = st.text_area("Enter a comma-separated list of video URLs:")
        save_path = st.text_input("Enter the path to save the videos (default is current directory):")
        
        get_transcript = st.checkbox("Get Transcript")
        create_audio = st.checkbox("Create audio from translated transcript")
        display_video = st.checkbox("Display video")

        if st.button("Download Videos") and url_list and save_path:
            url_list = url_list.split(",")
            for url in url_list:
                selected_video = YouTube(url)
                download_video(selected_video, save_path, get_transcript, create_audio, display_video)
