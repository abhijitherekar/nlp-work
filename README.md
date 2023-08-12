# nlp-project

Requirements: See requirements.txt file for packages.
SERP_API_KEY,
OPENWEATHERMAP_API_KEY,
OPEN_AI_KEY

Install mpg123 for playing mp3 files.

Overview:

Steps
Using Streamlit as UI
Speech input / image / video input and generate text files
Create embeddings (could be sentence transformer / Bert)
Exploring other models besides OPEN AI
Store embeddings in Pinecone/Chroma (figure out which database to use) —> further explore
How to use it etc
Packages to use
https://pypi.org/project/pipreqs/

Tasks: 

Text article: The text article must have at least a 1000 words.
Tl;dr: The response must be prefaced with a tl;dr of the article.
Generated images: An appropriate banner image that is generated to capture the context of the article. Besides this, there can be more generated images.
Public images: Appropriate images from the Unsplash image repository that is relevant to the article.
Speech rendering: A speech rendering of the text of the article.
Video: Optionally, a video rendering of the article content.
Langchain Agents
English speech to text ability to transcribe speech to text
At least one foreign language speech to english text In other words, the user may speak in a foreign language of your choice. The application then transcribes it to english, and proceeds from there. The article must be in english
Wikipedia ability to search wikipedia
Arxiv ability to query into arxiv paper repository
Gutenberg Ability to retrieve a book from the
Gutenberg free text library
Url Ability to scrape the contents of a webpage
Search Ability to search the web
Weather Find the weather
Query a relational database An ability to translate a natural language ask into an appropriate database query, for medical queries. See the open-source database of medical notes posted to the slack channel for this.
Math ability to perform basic math
Youtube ability to search for a relevant video in Youtube, and then to get its transcript
Three custom abilities Three custom abilities of your choice
Semantic search on PDF/book
thirukural
YouTube video timestamping for specific info
Brief summary of asif videos
Video transcription —> chat gpt to generate questions
Detecting language automatically and convert to English

Future Goals
Aim for all open source! No open ai
Graph database (possible implementation)




