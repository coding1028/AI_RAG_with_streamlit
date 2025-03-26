import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
import os
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# API key 정보 로드
load_dotenv()

logging.langsmith("project multi turn 챗봇")

# 캐시 디렉토리 생성
if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache")

#파일 업로드 전용 폴더
if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/files"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/files")

if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/embeddings"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/embeddings")

st.title("대화내용 기억하는 챗봇")

if "messages" not in st.session_state:
    # 대화기록 저장
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

# 버튼 지정
with st.sidebar:
    # 초기화 버튼임
    clear_btn = st.button("대화 초기화")
    selected_prompt = "prompts/pdf-rag-xionic.yaml"

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index = 0)

    # 세션 아이디 저장 메뉴
    session_id = st.text_input(" 세션 아이디를 입력하세요.", "abc123")

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

def create_chain(model_name = "gpt-4o"):
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

# llm 생성
    llm = ChatOpenAI(model_name=model_name)

# 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()
    # 이미지 파일로 부터 질의(스트림 방식)

    chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)
    return chain_with_history

# 초기화 버튼 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 정리
print_messages()

user_input = st.chat_input("내용을 입력하세요.")

# 경고 메시지 띄우기 위한 빈 영역
warning_msg = st.empty()

if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(model_name = selected_model)

#입력 시
if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        response = chain.stream(
            # 질문 입력
            {"question": user_input},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )

        st.chat_message("user").write(user_input)

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
        # 이미지 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")