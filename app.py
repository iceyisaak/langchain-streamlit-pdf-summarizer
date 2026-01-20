import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma


st.set_page_config(page_title="Langchain: Summarize Text from Webpage")
st.title("Langchain-Streamlit: PDF Summarizer")
# st.subheader("Summarize URL")


############################################################################ Sidebar

st.sidebar.title("Keys & Tokens")
groq_api_key=st.sidebar.text_input("Enter your Groq API Key: ",type="password")


############################################################################ App

if groq_api_key:
    llm=ChatGroq(groq_api_key=groq_api_key,model='llama-3.1-8b-instant')
    uploaded_files=st.file_uploader("Upload a PDF File", type="pdf",accept_multiple_files=True)



    # Prompt Template
    prompt_template = """
        Provide a summary of the following content in 500 words:
        Content: {text}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    #######################################################


    if uploaded_files:

            def create_vector_embedding():

                if 'store' not in st.session_state:
                    st.session_state.store={} 

                documents=[]
                for uploaded_file in uploaded_files:

                    if not os.path.exists("./document"):
                        os.makedirs("./document")

                    temp_pdf=f"./document/temp.pdf"
                    with open(temp_pdf,"wb") as file:
                        file.write(uploaded_file.getvalue())

                    pdf_loader=PyPDFLoader(temp_pdf)
                    docs=pdf_loader.load()
                    documents.extend(docs)

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    chunks = text_splitter.split_documents(docs)

                    # 3. Summarization chain
                    # MapReduce will handle the large token count automatically
                    summarization_chain = load_summarize_chain(
                        llm, 
                        chain_type="map_reduce", 
                        map_prompt=prompt, 
                        combine_prompt=prompt
                    )

                    # 4. Invoke using the chunks, not the full docs
                    output_summary = summarization_chain.invoke(chunks)
                    
                    # Invoke chain
                    st.success(output_summary['output_text'])


            if "vector_store" not in st.session_state:
                with st.spinner("Loading documents..."):
                    create_vector_embedding()

