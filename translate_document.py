from os import getenv
import requests
from pdfminer.high_level import extract_text_to_fp
from io import BytesIO
from dotenv import load_dotenv
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load document
load_dotenv()
drive_id = getenv("DRIVE_ID")
file = f"https://drive.google.com/uc?export=download&id={drive_id}"
response = requests.get(file)
nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer()

if response.status_code == 200:
    pdf_data = response.content
    # Extract document into text
    pdf_file = BytesIO(pdf_data)
    output_string = BytesIO()
    extract_text_to_fp(pdf_file, output_string)
    text = output_string.getvalue().decode('utf-8')

    # Splitting text into each sentence
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sents]
    for sentence in doc.sents:
        print(sentence.text)
    logging.info("Processed document into sentence.")

    # Vertorize sentence English
    vector_en = vectorizer.fit_transform(sentences)
    logging.info("TF-IDF vectors computed.")

    # Translate the sentence
    translated_sentences = []
    for sentence in sentences:
        translated_sentence = GoogleTranslator(source='english', target='french').translate(text=sentence)
        translated_sentences.append(translated_sentence)
    logging.info("Sentences translated.")

    # Vertorize sentence French
    vector_fr = vectorizer.transform(translated_sentences)
    logging.info("TF-IDF vectors for French sentences computed.")
    similarity_scores = cosine_similarity(vector_en, vector_fr).diagonal()
    logging.info("Similarity scores computed.")

    # Print output
    results = []
    for sentence_en, sentence_fr, similarity_score in zip(sentences, translated_sentences, similarity_scores):
        result = {
            "Content": sentence_en,
            "Translated": sentence_fr,
            "Similarity value": similarity_score
        }
        results.append(result)

    logging.info("Results computed.")
    print(json.dumps(results, indent=4, ensure_ascii=False))
    logging.info("Results printed.")
else:
    logging.error(f"Fail to download the file PDF. Code status: {response.status_code}")