import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


# Load Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print(OPENAI_API_KEY)

# Upload PDF Files
st.header("Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF File and start asking questions", type="pdf")


try:
    # Extract the Text
    if file is not None:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()


        # Break it into chunks 
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)

        # Generate the embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

        # create a vector store
        vector_store = FAISS.from_texts(chunks, embeddings)

        # get the input from user
        user_question = st.text_input("Please enter your question here")

        if user_question:
            # get the matches from vector store for user question
            match = vector_store.similarity_search(user_question)

            # create an LLM
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000
            )

            # chaining
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_question)
            st.write(response)
except Exception as ex:
    st.write("Something went wrong, Please try again later")
    print("Error", str(ex))