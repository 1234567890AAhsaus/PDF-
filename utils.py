from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models.tongyi import ChatTongyi

from langchain_community.embeddings import TensorflowHubEmbeddings

def qa_agent(tongyi_api_ke, memory, uploaded_file, question):
    model = ChatTongyi(model='qwen-long', api_key = "sk-c29e90529fb749f390f02be4693112a2")
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings = TensorflowHubEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
