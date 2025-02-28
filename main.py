import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
# from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
import base64
import os
from src.pdf_loader import BytesIOPyMuPDFLoader
from src.mongodb import create_search_vector_index, setup_mongodb


def main():
    st.title("ChatPDF")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("API Key", value=st.secrets["Gemini"]["GEMINI_API_KEY"])
        if len(api_key) == 0:
            st.warning("Please enter your Google Gemini API key.")
            st.stop()

        if "GOOGLE_API_KEY" not in os.environ and len(api_key) > 0:
            os.environ["GOOGLE_API_KEY"] = api_key

        with st.expander("MongoDB Atlas Settings"):
            mongodb_uri = st.text_input("MongoDB URI",
                                        value=st.secrets["MongoDB"]["URI"])
            mongodb_db = st.text_input("MongoDB Database", value=st.secrets["MongoDB"]["DB"])
            mongodb_collection = st.text_input("MongoDB Collection", value=st.secrets["MongoDB"]["COLLECTION"])
        
            btn_reset = st.button("Reset Chat & Embeddings")
            btn_create_vector_index = st.button("Create Vector Search Index")

        if btn_reset:
            client, collection = setup_mongodb(mongodb_uri, mongodb_db, mongodb_collection)
            collection.delete_many({})
            client.close()
            st.session_state.messages = []
            st.success("Chat, cache and embeddings have been reset.")

        if btn_create_vector_index:
            create_search_vector_index(mongodb_uri, mongodb_db, mongodb_collection)
            st.success("Vector Search Index has been created.")

    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    # Set up MongoDB connection
    client, collection = setup_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

    pdf_files_info = {}
    for index, pdf_file in enumerate(pdf_files):
        pdf = PdfReader(pdf_file)
        number_of_pages = len(pdf.pages)
        pdf_files_info[pdf_file.name] = {}
        pdf_files_info[pdf_file.name]["index"] = index
        pdf_files_info[pdf_file.name]["pages"] = number_of_pages

        # creating embeddings
        # load pdf file
        loader = BytesIOPyMuPDFLoader(pdf_file)
        documents = loader.load()
        for doc in documents:
            doc.metadata['file_name'] = pdf_file.name

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Set up embeddings and vector store
        # embeddings = OpenAIEmbeddings()
        embeddings = GoogleGenerativeAIEmbeddings(model=st.secrets["Gemini"]["EMBED_MODEL"])
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            texts, embeddings, collection=collection, index_name="vector_index"
        )

    if pdf_files is not None and len(pdf_files) > 0:
        # show pdf
        with st.expander("PDF Viewer"):
            pdf_file_name = st.selectbox("Select PDF file", list(pdf_files_info.keys()))
            pdf_file = pdf_files[pdf_files_info[pdf_file_name]["index"]]
            
            base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')

            # Embedding PDF in HTML
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf">'

            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)

        # chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # question = st.text_input("Question", value="What is the document about?")
        prompt = st.chat_input("What is the document about?")

        # Set up retriever and language model
        retriever = vector_store.as_retriever(search_type="similarity",
                                              search_kwargs={"pre_filter":
                                                             {"file_name": {"$in": list(pdf_files_info.keys())}},
                                                             "k": 5})
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        llm = GoogleGenerativeAI(
                                    model=st.secrets["Gemini"]["LLM_MODEL"],
                                    temperature=0,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2
                                )

        # Set up RAG pipeline
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

        # RAG without embeddings
        # chain = load_qa_chain(llm, chain_type="stuff")
        # docs = retriever.invoke(prompt)
        # result = chain.run(input_documents=docs, question=prompt)

        # Function to process user query
        def process_query(prompt):
            result = qa_chain({"query": prompt})
            return result["result"], result["source_documents"]

        if prompt:
            st.chat_message("user").write(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            answer, sources = process_query(prompt)

            # st.chat_message("assistant").write(f"Raw: {result}")
            st.chat_message("assistant").write(f"{answer}")
            # Add AI message to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # for doc in sources:
            #     messages.chat_message("assistant").write(f"Sources: {doc.metadata['source']}: {doc.page_content[:100]}...")

        # Don't forget to close the MongoDB connection when done
        client.close()

if __name__ == "__main__":
    st.set_page_config(
        page_title="ChatPDF",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/kavehbc/chat-pdf',
            'Report a bug': "https://github.com/kavehbc/chat-pdf",
            'About': "ChatPDF with Gemini"
        }
    )

    main()
