# Automação do youtube - Parte 1

Informação é um instrumento poderoso que precisa ser bem filtrado. 
Na pior das hipóteses, você afoga com tanta informação - tanto úteis quanto inúteis -,
ou você usa sem medir as consequências externas.  

Quando falamos de automação, a preocupação não deveria ser com robôs fazendo trabalho de humanos,
mas a responsabilidade de um humano em produzir uma grande quantidade de informação tendo um alcance global em pouco tempo.

Programadores são ensinados a quebrar o problema em partes menores sem perder o panorama total do quebra-cabeça.

primera parte:
- Autoconhecimento. Não faça nada sem ter objetivos legítimos e benéficos para você e as pessoas em sua volta.
é a parte mais demorada e você não pode negligenciá-la senão você vai desistir na primeira linha de código que não tiver "hello, world".

segunda parte:
- Identifique um problema na sociedade que você queira resolver


terceira parte:
- Pegue o café e vamos para o código.
<br />
<br />


Instale as bibliotecas necessárias. Se você não sabe inglês, dá um google aí.

Não precisa instalar o transformers do Hugging Face porque a biblioteca chamada deepmultilingualpunctuation
já tem o transformers e pytorch, assim você vai evitar conflitos de dependências.




```
import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os
```

Cada função(def) vai passar um argumento, portanto não se esqueça de pegar esse argumento antes de chamar a função 
que vai trabalhar com ele ou dentro da própria função só a ativando depois de passar o valor da variável.

Primeiro vamos pegar o transcript de um video do youtube e colocar um exception para mostrar uma mensagem de erro caso não retenha o transcript na variável.


```
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
```

Traduza o transcript usando modelos pré-treinados do Hugging Face (se você quiser pode criar o seu próprio modelo, mas isso vai te exigir um pouco mais).<br />
Nessa parte eu tive um problema com o tamanho do texto e tive que dividir em partes(chunks) e depois juntar tudo(append).

Antes de continuar, lembre-se da regra absoluta para programadores: se funciona, NÃO TOCA.

Somente quando o usuário acionar o download do video, o texto vai ser encaminhado para a função com as próximas etapas.
No caso, vai ser usado um model do Hugging Face para a correção da pontuação e a biblioteca gtts para a criação do audio a partir do transcript traduzido e corrigido.


```
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
```


```
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
```


Só depois vai ser preparada a interface do aplicativo usando streamlit depois de finalizar a definição das funções.
Leia a documentação para incluir checkbox, title, display e audio para ativar as respectivas funções que foram criadas no inicio. 
Cada if só vai funcionar se a pessoa selecionar o checkbox para cada ação.


```
if __name__ == "__main__":
    st.title("YouTube Video Downloader and Transcript App")

    option = st.radio("Choose an option", ["Search for videos", "Provide list of URLs"])

    if option == "Search for videos":
        query = st.text_input("Enter your search query:")
        save_path = st.text_input("Enter the path to save the video (default is current directory):")
        
        get_transcript = st.checkbox("Get Transcript")
        create_audio = st.checkbox("Create audio from translated transcript")
        display_video = st.checkbox("Display video")

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
        save_path = st.text_input("Enter the path to save the videos (default is current directory):", ".")
        
        get_transcript = st.checkbox("Get Transcript")
        create_audio = st.checkbox("Create audio from translated transcript")
        display_video = st.checkbox("Display video")

        if st.button("Download Videos") and url_list and save_path:
            url_list = url_list.split(",")
            for url in url_list:
                selected_video = YouTube(url)
                download_video(selected_video, save_path, get_transcript, create_audio, display_video)
```
<br />
Citations
<br />
<br />
@article{guhr-EtAl:2021:fullstop,
  title={FullStop: Multilingual Deep Models for Punctuation Prediction},
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  Böhme, Hans Joachim},
  booktitle      = {Proceedings of the Swiss Text Analytics Conference 2021},
  month          = {June},
  year           = {2021},
  address        = {Winterthur, Switzerland},
  publisher      = {CEUR Workshop Proceedings},  
  url       = {http://ceur-ws.org/Vol-2957/sepp_paper4.pdf}
}

