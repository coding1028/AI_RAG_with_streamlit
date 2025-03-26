import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote import logging

# API key 정보 로드
load_dotenv()

logging.langsmith("project_RAG_pdf")

# 캐시 디렉토리 생성
if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache")

#파일 업로드 전용 폴더
if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/files"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/files")

if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/embeddings"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/embeddings")

st.title("PDF 기반 QA")

if "messages" not in st.session_state:
    # 대화기록 저장
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = []

# 버튼 지정
with st.sidebar:
    # 초기화 버튼임
    clear_btn = st.button("대화 초기화")
    # 파일 업로더   
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    selected_prompt = "prompts/pdf-rag.yaml"

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index = 0)

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 파일을 캐시 저장(시간 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다....")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    # file_path = f"C:\\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache\files/{file.name}"
    file_path = rf"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache\files{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)   
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever

def create_chain(retriever, model_name = "gpt-4o"):
    # 프롬프트 적용
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    
    prompt = load_prompt(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\prompts\pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)
 
    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if uploaded_file:
    # 파일 업로드 후 검색 생성, (작업시간 오래결림) 
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 정리
print_messages()

user_input = st.chat_input("내용을 입력하세요.")

# 경고 메시지 띄우기 위한 빈 영역
warning_msg = st.empty()

#입력 시
if user_input:
    # chain = create_chain(selected_prompt)
    chain = st.session_state["chain"]

    # 파일 업로드 X시
    if chain is not None:
    # 스트리밍 호출
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()

            # AI_answer = answer 
            answer = ""
            for token in response:
                answer += token
                container.markdown(answer)
        # ai_answer = chain.invoke({"question": user_input})

        # 어시스턴트 답변
        # st.chat_message("assistant").write(ai_answer)
        # 대화기록 저장
        add_message("user", user_input)
        add_message("assistant", answer) 
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")