from os import getenv
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from flask import Flask, request, jsonify
import threading
import requests
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_community.callbacks import get_openai_callback
from astrapy.db import AstraDB as AstraDBPy
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)

# Load openai key
load_dotenv()
llm = OpenAI(openai_api_key=getenv('OPENAI_API_KEY'), temperature=0.3)
model = "text-embedding-ada-002"

# Load AstraDB
embedding = OpenAIEmbeddings(model=model)
vstore = AstraDB(
    embedding=embedding,
    collection_name="astra_vector_demo3",
    api_endpoint=getenv("ASTRA_DB_API_ENDPOINT"),
    token=getenv("ASTRA_DB_APPLICATION_TOKEN")
)

astra_db_get_topics = AstraDBPy(token=getenv("ASTRA_DB_APPLICATION_TOKEN"),
                                api_endpoint=getenv("ASTRA_DB_API_ENDPOINT"))
topics_collection = astra_db_get_topics.collection(collection_name="astra_vector_demo3")

# Create class for prompting
class Multiple(BaseModel):
    question: str = Field(description="The question of the quiz.")
    option: list = Field(
        description=f"""List of strings containing options for the question.""")
    answer: str = Field(description="The correct answer to the question.")
    explanation: str = Field(description="Explanation of the correct answer.")

class ShortEssay(BaseModel):
    question: str = Field(description="The question generated based on the context.")
    answer: str = Field(description="The answer of the question.")

# Send result to webhook
def send_to_webhook(results, cb):
    load_dotenv()
    webhook_url = getenv("WEBHOOK_URL")
    headers = {"Content-Type": "application/json"}
    response_data = {
        "quiz_result": results,
        "tokens_in": cb.prompt_tokens,
        "tokens_out": cb.completion_tokens
    }
    requests.post(webhook_url, json=response_data, headers=headers)

# Setting option for multiple choice question
def generate_options(total_options):
    options = ["A) ...", "B) ...", "C) ...", "D) ...", "E) ..."]
    return options[:total_options]

# Get chunk 
def get_chunks(document_id, amount_of_quiz):
    array_header = set()
    generator = topics_collection.paginated_find(
        filter={"metadata.document_id": document_id},
        options={"limit": 4 * (amount_of_quiz * 4 + 8)}
    )
    for doc in generator:
        if 'metadata' in doc:
            for index in range(1, 3):
                header_key = f'header{index}'
                if header_key in doc['metadata']:
                    header_value = doc['metadata'][header_key]
                    header_value = header_value.replace('\t', ' ').replace('\n', ' ')
                    if header_value.strip():
                        array_header.add(header_value)
    shuffled_array_header = list(array_header)
    return shuffled_array_header

# Checking duplicate data and remove it if duplicate
def check_duplicates(result, chain, retrieved_docs, multiple_option_amount):
    seen_questions = set()

    for idx, question_answer_pair in enumerate(result):
        if isinstance(question_answer_pair, dict):
            question = question_answer_pair.get("question")
            if question:
                if question in seen_questions:
                    new_result = chain.invoke({"context": retrieved_docs, "number_of_quiz": 1,
                                               "multiple_option_amount": multiple_option_amount})[0]
                    result[idx] = new_result
                seen_questions.add(question)

    return result

# Make prompting for multiple choice and essay short question
def prompting(type_of_quiz, multiple_option_amount, lang):
    if type_of_quiz == 'multiple':
        options = generate_options(multiple_option_amount)
        language_prompt = "English" if lang == "en" else "French"

        final_prompt = f"""
            You will read this context, understand it, and create quiz for the students. The context is:
            {{context}}
        
            You have to create {{number_of_quiz}} questions. When creating multiple choices question, set the {{multiple_option_amount}} hard-to-answer 'options' in bullet points with only ONE right answer and put the right answer below it with explanation. Here is the format you MUST follow:
            {{format_instructions}}
            "question": This is the question. Use question sentence. Create question by using the 'context' from the document.
            "options": An array of {{multiple_option_amount}} strings which consists of {options}
            "answer": This is the correct answer to the question above.
            "explanation": This is the explanation.
        
            Return in array of object format with each of the above is within an object

            Languange : {language_prompt}
            """
    elif type_of_quiz == 'essay_short':
        language_prompt = "English" if lang == "en" else "French"
        final_prompt = f"""
              You are an expert of the document to create STRAIGHTFORWARD essay questions that requires short answer for students. Based on this context:
              {{context}}

              Create the question and then the answer below. Here is the format you MUST follow:
              {{format_instructions}}
              Create total {{number_of_quiz}} questions with the answers.
              "question": This is the question. Use question sentence. Create descriptive and easy-to-answer question in such students can answer in one short sentence.
              "answer": This is the answer to the question above

              Return in array of object format with each object representing question and answer.

              Languange : {language_prompt}
            """

    return final_prompt

# Create quiz based on document and input category question also total question
def create_quiz(source, document_id, type_of_quiz, number_of_quiz, multiple_option_amount, lang):
    retrieved_docs = ''
    all_results = []
    total_questions_generated = 0
    retriever = vstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k": 10,
                                                   'filter': {'document_name': source, 'document_id': document_id}
                                                   }
                                    )

    all_headers = get_chunks(document_id, number_of_quiz)
    batched_headers = [all_headers[i:i + 2] for i in range(0, len(all_headers), 2)]

    for batch_headers in batched_headers:
        header_batch_str = ', '.join(batch_headers)
        chunks = retriever.get_relevant_documents(header_batch_str)

        for chunk in chunks:
            retrieved_docs += chunk.page_content

        if type_of_quiz == "multiple":
            parser = JsonOutputParser(pydantic_object=Multiple)
        elif type_of_quiz == "essay_short":
            parser = JsonOutputParser(pydantic_object=ShortEssay)

        final_prompt = prompting(type_of_quiz, multiple_option_amount, lang)

        prompt = PromptTemplate(template=final_prompt,
                                input_variables=["context", "number_of_quiz", "multiple_option_amount"],
                                partial_variables={"format_instructions": parser.get_format_instructions()})

        chain = (
                {"context": itemgetter("context"),
                 "number_of_quiz": itemgetter("number_of_quiz"),
                 "multiple_option_amount": itemgetter("multiple_option_amount")}
                | prompt
                | llm
                | parser
        )

        with get_openai_callback() as cb:
            results = chain.invoke(
                {"context": retrieved_docs, "number_of_quiz": 2, "multiple_option_amount": multiple_option_amount})
            total_questions_generated += 2
            print(results)
            all_results.extend(results)

            if total_questions_generated >= number_of_quiz:
                new_results = check_duplicates(all_results, chain, retrieved_docs, multiple_option_amount)
                break

    all_results = new_results[:number_of_quiz]

    send_to_webhook(all_results, cb)

    return print(all_results)

# Create quiz builder automatically
@app.route("/quiz_builder", methods=['POST'])
def quiz_builder():
    quiz_builder_id = request.json.get('quiz_builder_id')
    source = request.json.get('source')
    document_id = request.json.get('document_id')
    type_of_quiz = request.json.get('type_of_quiz')
    number_of_quiz = request.json.get('number_of_quiz')
    multiple_option_amount = request.json.get('multiple_option_amount')
    lang = request.json.get('lang')

    threading.Thread(target=create_quiz,
                     args=(source, document_id, type_of_quiz, number_of_quiz, multiple_option_amount, lang)).start()

    return jsonify({"message": "Successfully Load"}), 200


if __name__ == "__main__":
    app.run(debug=True)