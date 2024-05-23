from os import getenv
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import AstraDB
from langchain_core.documents import Document
import openai
from langchain.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load openai key
load_dotenv()
llm = OpenAI(openai_api_key=getenv('OPENAI_API_KEY'), temperature=0)
model = "text-embedding-ada-002"

# Load AstraDB
embedding = OpenAIEmbeddings(model=model)
vstore = AstraDB(
    embedding=embedding,
    collection_name="astra_vector_demo3",
    api_endpoint=getenv("ARTA_DB_API_ENDPOINT"),
    token=getenv("ASTRA_DB_APPLICATION_TOKEN")
)

# Load pdf document
def load_pdf(file) :
    loaderPDFMiner = PDFMinerPDFasHTMLLoader(file)
    data = loaderPDFMiner.load()[0]
    soup = BeautifulSoup(data.page_content, 'html.parser')
    content = soup.find_all('span', style=True)
    return content

# Splitting document into 3 header
def splitting_data(content, file_name):
    markdown_header = []
    for header in content:
        style = header.get('style')
        if style:
            font_size = re.search(r'font-size:(\d+)px', style)
            if font_size:
                font_size = int(font_size.group(1))
                text = header.get_text().strip()
                if font_size > 20:
                    markdown_header.append(f"# {text}")
                elif 15 < font_size <= 20:
                    markdown_header.append(f"## {text}")
                elif 12 < font_size <= 15:
                    markdown_header.append(f"### {text}")

    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3")
    ]

    markdown_document = ' '.join(markdown_header)

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    md_header_splits = markdown_splitter.split_text(markdown_document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    final_splits = []
    for text in md_header_splits:
        text_sections = text_splitter.split_documents([text])
        for section in text_sections:
            section.metadata["document_name"] = file_name
            final_splits.append(section)

    return final_splits

# Create output data
def output(splits):
    documents = []
    for split in splits:
        document = Document(page_content=split.page_content, metadata=split.metadata)
        documents.append(document)
    return documents

# Make RAG by retrieve data from AstraDB
def retriever(question):
    retriever = vstore.as_retriever()
    template = """
    Use the provided context as the basis for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
    Your answers must be concise and to the point, and refrain from answering about other topics than philosophy.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            |prompt
            | llm
            | StrOutputParser()
    )
    result = chain.invoke(question)

    return result

@app.route("/process_data", methods=['POST'])
def process_data():
    # Input document
    file = "bahasa_prancis.pdf"
    file_name = os.path.basename(file)
    # Load, split,and send document to AstraDB
    content = load_pdf(file)
    final_splits = splitting_data(content, file_name)
    documents = output(final_splits)
    print(documents)
    vstore.add_documents(documents)

# Retrieve data and generate QNA
# @app.route("/retriever_data", methods=['POST'])
def retriever_data():
    # question = request.form.get('question')
    question = "pouvez-vous nous expliquer les FONDEMENTS DU DISPOSITIF D’EVALUATION ?"
    answer = retriever(question)
    print(answer)
    # return jsonify({"message": "Successfully load"}), 200



if __name__ == "__main__":
    # app.run(debug=True)
    retriever_data()