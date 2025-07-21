# Save Streamlit app as app.py
%%writefile app.py
import streamlit as st
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load corpus
with open("corpus.txt", "r") as f:
    corpus = f.read()
documents = corpus.strip().split("\n\n")
docs = [Document(page_content=doc) for doc in documents]

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)

# Load LLM
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    max_new_tokens=50,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Set up memory and RAG chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)

# Streamlit app
st.title("RAG Chatbot")
st.write("Ask questions about AI, ML, or Transformers!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response using .invoke()
    result = rag_chain.invoke({"question": prompt})
    response = result["answer"]
    
    # Display and store response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
