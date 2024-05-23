from os import getenv
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)

# Load openai key
load_dotenv()
llm = OpenAI(openai_api_key=getenv('OPENAI_API_KEY'), temperature=0)
outputParser = StrOutputParser()

# Create RAG with Buffer Memory
@app.route("/get_data", methods=['POST'])
def get_data():
    memory = ConversationBufferMemory(return_messages=True)
    totalToken = 0
    completionToken = 0
    costToken = 0
    try:
        while True:
            prompt = input("Question: ")
            with get_openai_callback() as cb:
                totalToken += cb.total_tokens
                completionToken += cb.completion_tokens
                costToken += cb.total_cost

            conversation = ConversationChain(
                llm=llm,
                verbose=True,
                memory=memory
            )
            parsed_result = outputParser.parse(prompt)
            conversation.invoke(input=parsed_result)
            # memory.save_context({"input": ai_response['input']}, {"output": ai_response['response']})

            all_memory = memory.load_memory_variables({})
            memoryData = []
            for message in all_memory['history']:
                if isinstance(message, HumanMessage):
                    memoryData.append({"Content": message.content, "Type": "Human"})
                elif isinstance(message, AIMessage):
                    memoryData.append({"Content": message.content, "Type": "AI"})
            print(memoryData)

            print(f"Tokens Use, Completion Tokens, Cost (USD) : {cb.total_tokens}, {cb.completion_tokens}, ${cb.total_cost}")
            print(f"Total Tokens, TotalCompletion Tokens, Total Cost (USD) : {totalToken}, {completionToken}, ${completionToken}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# Create RAG with Summary Memory
@app.route("/summary_data", methods=['POST'])
def summary_data():
    summaryMemory = ConversationSummaryMemory(llm=llm)
    totalToken = 0
    completionToken = 0
    costToken = 0
    try:
        while True:
            prompt = input("Question: ")
            if prompt.lower() == "quit":
                break
            with get_openai_callback() as cb:
                conversation = ConversationChain(
                    llm=llm,
                    verbose=True,
                    memory = summaryMemory
                )
                parsed_result = outputParser.parse(prompt)
                conversation.invoke(input=parsed_result)
            # summaryMemory.save_context({"input": ai_response['input']}, {"output": ai_response['response']})

            summary = summaryMemory.load_memory_variables({})
            print(summary['history'])

            totalToken += cb.total_tokens
            completionToken += cb.completion_tokens
            costToken += cb.total_cost
            print(f"Tokens Use, Completion Tokens, Cost (USD) : {cb.total_tokens}, {cb.completion_tokens}, ${cb.total_cost}")
            print(f"Total Tokens, TotalCompletion Tokens, Total Cost (USD) : {totalToken}, {completionToken}, ${completionToken}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)