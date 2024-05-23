from os import getenv
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import AstraDB
from flask import Flask, request, jsonify
import tiktoken
import threading
import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_community.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Load openai key
load_dotenv()
llm = OpenAI(openai_api_key=getenv('OPENAI_API_KEY'), temperature=0)
model = "text-embedding-ada-002"
outputParser = StrOutputParser()

# Load AstraDB
embedding = OpenAIEmbeddings(model=model)
vstore = AstraDB(
    embedding=embedding,
    collection_name="astra_vector_demo3",
    api_endpoint=getenv("ARTA_DB_API_ENDPOINT"),
    token=getenv("ASTRA_DB_APPLICATION_TOKEN")
)

# Download file pdf
def download_pdf(file):
    response = requests.get(file)
    if response.status_code == 200:
        pdf_data = response.content
        return pdf_data
    else:
        return None

# Load pdf file
def load_pdf(file):
    loaderPDFMiner = PDFMinerPDFasHTMLLoader(file)
    data = loaderPDFMiner.load()[0]
    soup = BeautifulSoup(data.page_content, 'html.parser')
    content = soup.find_all()
    return content

# Splitting data into 3 header
def splitting_data(content):
    markdown_header = ""
    for header in content:
        style = header.get('style')
        if style:
            font_size = re.search(r'font-size:(\d+)px', style)
            if font_size:
                font_size = int(font_size.group(1))
                text = header.get_text().strip()
                if font_size > 20 and len(text) < 100:
                    text = f"# {text} " if not text.startswith("#") else text
                elif 15 < font_size <= 20 and len(text) < 100:
                    text = f"## {text} " if not text.startswith("##") else text
                elif 12 < font_size <= 15 and len(text) < 100:
                    text = f"### {text} " if not text.startswith("###") else text
                markdown_header += text.replace('\n', '') + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False
    )

    text_sections = text_splitter.split_text(markdown_header)

    return text_sections

# Extract header for each chunk
def extract_headers(chunk):
    headers = {"header1": None, "header2": None, "header3": None}
    last_headers = {"header1": None, "header2": None,
                    "header3": None} 

    for line in chunk.split('\n'):
        if line.startswith("#"):

            if line:
                if line.startswith("###"):
                    last_headers["header3"] = line.strip()[4:]
                    last_headers["header2"] = None
                    last_headers["header1"] = None
                elif line.startswith("##"):
                    last_headers["header2"] = line.strip()[3:]
                    last_headers["header1"] = None
                else:
                    last_headers["header1"] = line.strip()[2:]

    headers.update({key: value for key, value in last_headers.items() if value})

    return headers

# Create chunk
def process_chunks(chunks, file_name, file_id):
    results = []
    last_headers = {"header1": "", "header2": "", "header3": ""}
    encoding = tiktoken.get_encoding("cl100k_base")

    for chunk in chunks:
        headers = extract_headers(chunk)
        num_tokens = len(encoding.encode(chunk))

        for key in headers:
            if headers[key]:
                last_headers[key] = headers[key]
            else:
                headers[key] = last_headers[key]

        metadata = {
            "header1": headers["header1"],
            "header2": headers["header2"],
            "header3": headers["header3"],
            "document_name": file_name,
            "document_id": file_id,
            "tokens_embedded": num_tokens
        }

        document = Document(page_content=chunk, metadata=metadata)
        results.append(document)

    # print(results)
    vstore.add_documents(results)

# Caclulate similarity chunk and answer
def calculate_similarity(chunk, answer):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([chunk, answer])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_score = similarity_matrix[0, 1]
    return similarity_score

# Retrieve data from document in AstraDB and generate the answer based on the question
def retriever(question, document_name, document_id, memory):
    retrieved_docs = ''
    retrieved_headers = []
    retriever = vstore.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold": 0.95,
                                                   'filter': {'document_name': document_name,
                                                              'document_id': document_id}})
    docs = retriever.get_relevant_documents(question)
    for doc in docs:
        retrieved_docs += doc.page_content
        retrieved_headers.append(doc.metadata)

    template = """
    Before asking a question, please make sure it is related to the content of the document.
    Only questions relevant to the document's topics will receive accurate responses.

    Use the provided context as the basis for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
    Your answers should be short and to the point, and don't answer about other topics out of context.

    Memory: take {memory} for reference the previous answer and relevant to the context, if not relevant please don't take context from outside
    Context: {context}
    Question: {question}
    Answer: fill the answer that related to the existing context, if not please return 'The question does not fit the existing context' """

    prompt = PromptTemplate(template=template, input_variables=["context", "question", "memory"])

    chain = (
            {"context": itemgetter("context"), "question": itemgetter("question"), "memory": itemgetter("memory")}
            | prompt
            | llm
            | outputParser
    )

    with get_openai_callback() as cb:
        result = chain.invoke({"context": retrieved_docs, "question": question, "memory": memory})
        headers = retrieved_headers

    return result, headers, cb, retrieved_docs

# Get header reference
def retrieve_header_ref(context_docs):
    header_refs = []
    for doc in context_docs:
        h1 = doc['header1']
        h2 = doc['header2']
        h3 = doc['header3']

        if h2 and h3:
            header_ref = "-{}>{}>{}>{}".format(doc['document_name'], h1, h2, h3)
        elif h2:
            header_ref = "-{}>{}>{}".format(doc['document_name'], h1, h2)
        elif h3:
            header_ref = "-{}>{}>{}".format(doc['document_name'], h1, h3)
        else:
            header_ref = "-{}>{}".format(doc['document_name'], h1)

        header_refs.append(header_ref)
    return "\n".join(header_refs)

# Process generate answer based on the document
@app.route("/retriever_data", methods=['POST'])
def retriever_data():
    question = request.json.get('question')
    document_name = request.json.get('document_name')
    document_id = request.json.get('document_id')
    chat_history = request.json.get('chat_history')

    memory = ConversationBufferMemory()
    previous_context = ""
    results = []
    if chat_history != []:
        for msg in chat_history:
            if msg['type'] == 'Human':
                results.append({"input": msg['content']})
            elif msg['type'] == 'AI':
                results.append({"output": msg['content']})
                previous_context += msg['content'] + "\n"
        for i in range(0, len(results), 2):
            human_input = results[i]
            ai_output = results[i + 1] if i + 1 < len(results) else None
            memory.save_context(human_input, ai_output)
    elif chat_history == []:
        memory.save_context({"input": ""}, {"input": ""})
    memory.chat_memory

    if previous_context.strip():
        combined_question = previous_context + question
    else:
        combined_question = question

    result, headers, cb, docs = retriever(combined_question, document_name, document_id, memory)
    header_ref = retrieve_header_ref(headers)
    cleaned_result = result.strip()

    # if cleaned_result == "The question does not fit the existing context.":
    #     header_ref = ''

    similarity_score = calculate_similarity(docs, cleaned_result)
    human = {'content': question, 'type': 'Human'}
    ai = {'content': cleaned_result, 'header_ref': header_ref, 'type': 'AI'}
    chat_history.append(human)
    chat_history.append(ai)

    response_data = {
        "chat_history": chat_history,
        "message": cleaned_result,
        "tokens_in": cb.prompt_tokens,
        "tokens_out": cb.completion_tokens
    }

    print(similarity_score)
    return jsonify(response_data), 200


# Save document into AstraDB
@app.route("/process_data", methods=['POST'])
def process_data():
    try:
        file = request.json.get("file")
        document_name = request.json.get("document_name")
        document_id = request.json.get("document_id")
        content = load_pdf(file)

        threading.Thread(target=process_chunks, args=(splitting_data(content), document_name, document_id)).start()
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)