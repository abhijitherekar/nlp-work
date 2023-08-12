from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import whisper
import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from deep_translator import GoogleTranslator
from random_number_tool import random_number_tool
from youTube_helper import youtube_tool, youtube_search
from url_scraping_tool import url_scraping_tool
from current_time_tool import current_time_tool
from wiki_tool import wiki_tool
from weather_tool import weather_tool
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from tool_retrieval import get_tools

from text_to_image import text_to_image
from text_to_video import text_to_video

from gtts import gTTS

reader = PdfReader('./pdfs/embedding.pdf')
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text


text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools)]
model_name = "hkunlp/instructor-xl"
vector_store = Chroma.from_texts(texts, HuggingFaceInstructEmbeddings(model_name=model_name))


chain = load_qa_chain(OpenAI(), chain_type="stuff", verbose=False)

# from langchain.chains import VectorDBQAWithSourcesChain

# chain = VectorDBQAWithSourcesChain.from_chain_type(
#     llm=OpenAI(),
#     return_source_documents=True,
#     reduce_k_below_max_tokens=True,
#     chain_type="stuff"
# )

# to get input from speech use the following libs
model = whisper.load_model("large")

# define llm
llm = OpenAI(temperature=0.1)
# core function which will do all the work (POC level code)
def transcribe(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    detected_language = max(probs, key=probs.get)
    print("detected_language --> ", detected_language)
    if detected_language == "en":
        print("Detected English language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language='en')
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "ta":
        print("Detected Tamil language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='ta', target='en').translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    if result_text != "Unknown language" and len(result_text)!= 0:
        # Now add the lanfChain logic here to process the text and get the responses.
        # once we get the response, we can output it to the voice.
        #agent.
        tools = get_tools(result_text)
        agent = initialize_agent(tools=tools,  llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        print("agent tool --> ", agent.tools)
        agent_output = agent.run(result_text)
    else:
        agent_output = "I'm sorry I cannot understand the language you are speaking. Please speak in English or Tamil."

    # init some default image and video. Override based on agent output.
    detailed = ''
    image_path = 'supportvectors.png'
    video_path = 'welcome.mp4'

    if "tool" in agent_output:
        print("This is an article.")
        tldr = agent_output["tldr"]
        detailed = agent_output["article"]
        if (agent_output["tool"] == "youtube") and ("video" in agent_output):
            video_path = agent_output["video"]

    else:
        print("This is not an article. It is coming from agent.", agent_output)
        tldr = agent_output


    # generate image based on tldr
    try:
        image = text_to_image(tldr)
        if image:
            image_path = 'output.png'
    except BaseException as e:
        print('Some problem generating image.', str(e))

    # generate image based on tldr
    try:
        tldr_video_path = text_to_video(tldr)
    except:
        print('Some problem generating video.')

    # TTS. Marked slow=False meaning audio should have high speed
    myobj = gTTS(text=tldr, lang='en', slow=False)
    # Saving the converted audio in a mp3 file named
    myobj.save("welcome.mp3")
    # Playing the audio
    # os.system("mpg123 welcome.mp3")

    return tldr, detailed, image_path, video_path, "welcome.mp3", tldr_video_path


# Set the starting state to an empty string
#.launch(share=True)
def flip_text(audio):
    print("inputs --> ", audio)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    detected_language = max(probs, key=probs.get)
    print("detected_language --> ", detected_language)
    if detected_language == "en":
        print("Detected English language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language='en')
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "ta":
        print("Detected Tamil language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='ta', target='en').translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    query = result_text
    docs = vector_store.similarity_search(query)
    output = chain.run(input_documents=docs, question=query)
     # TTS. Marked slow=False meaning audio should have high speed
    myobj = gTTS(text=output, lang='en', slow=False)
    # Saving the converted audio in a mp3 file named
    myobj.save("welcome.mp3")
    return output, "welcome.mp3"

youtube_vector_store = Chroma.from_texts(["youtube"], HuggingFaceInstructEmbeddings(model_name=model_name))

def video_text(audio):
    print("inputs --> ", audio)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    detected_language = max(probs, key=probs.get)
    print("detected_language --> ", detected_language)
    if detected_language == "en":
        print("Detected English language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language='en')
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "hi":
        print("Detected hindi language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='hi', target='en').translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    query = result_text
    res = youtube_search(query)
    print(res["article"])
    raw_text = res["article"]
    texts = text_splitter.split_text(raw_text)
    youtube_vector_store.add_texts(texts)
   
    return res["video"]

def video_query(audio):
    print("inputs --> ", audio)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    detected_language = max(probs, key=probs.get)
    print("detected_language --> ", detected_language)
    if detected_language == "en":
        print("Detected English language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language='en')
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "hi":
        print("Detected hindi language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='hi', target='en').translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    query = result_text
    docs = youtube_vector_store.similarity_search(query)
    output = chain.run(input_documents=docs, question=query)
     # TTS. Marked slow=False meaning audio should have high speed
    myobj = gTTS(text=output, lang='en', slow=False)
    # Saving the converted audio in a mp3 file named
    myobj.save("welcome.mp3")
    return output, "welcome.mp3"

with gr.Blocks() as demo:
    gr.Markdown("Multi-Lingual (English/Hindi) AwesomeAI Tool")
    with gr.Tab("PDF voice Assitant"):
        pdf_inputs=gr.Audio(source="microphone", type="filepath", streaming=False)
        #text_output = gr.outputs["textbox", "audio"]
        pdf_text_button = gr.Button("Submit")
    with gr.Tab("Youtube Voice Assitance"):
        # gr.Interface(
        #     fn=transcribe,
        video_inputs=gr.Audio(source="microphone", type="filepath", streaming=False)
        video_query_inputs=gr.Audio(source="microphone", type="filepath", streaming=False)
        video_text_button = gr.Button("Submit to get the video")
        video_query_button = gr.Button("query the video")
        #outputs=["textbox", "textbox", "image", "video", "state", "audio", "video"]

    with gr.Tab("Genrative AI BOT"):
        inputs=gr.Audio(source="microphone", type="filepath", streaming=False)
        gen_ai_outputs=["textbox", "textbox", "image", "video", "state", "audio", "video"]
        all_in_one_bot_button = gr.Button("Submit query")

    pdf_text_button.click(flip_text, inputs= pdf_inputs, outputs=[gr.Textbox() ,gr.Audio()])
    video_text_button.click(video_text, inputs= video_inputs, outputs=[gr.Video()])
    video_query_button.click(video_query, inputs= video_query_inputs, outputs=[gr.Textbox() ,gr.Audio()])
    all_in_one_bot_button.click(transcribe, inputs=inputs, outputs=[gr.Textbox() , gr.Textbox(), gr.Image(), gr.Video() ,gr.Audio(), gr.Video()])

demo.launch(debug=True, share=True)