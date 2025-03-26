import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
import os
from langchain_teddynote import logging
from retriever import create_retriever 
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

# API key 정보 로드
load_dotenv()

logging.langsmith("project 이미지 인식")

# 캐시 디렉토리 생성
if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache")

#파일 업로드 전용 폴더
if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/files"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/files")

if not os.path.exists(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/embeddings"):
    os.mkdir(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache/embeddings")

st.title("이미지 인식 기반 챗봇")

if "messages" not in st.session_state:
    # 대화기록 저장
    st.session_state["messages"] = []

# 탭을 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화 내용"])
# 버튼 지정
with st.sidebar:
    # 초기화 버튼임
    clear_btn = st.button("대화 초기화")
    # 이미지 업로드  
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    selected_prompt = "prompts/pdf-rag-xionic.yaml"

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index = 0)

    # 시스템 프롬프트 추가
    system_prompt = st.text_area("시스템 프롬프트", "당신은 제무제표를 해석하는 금융 AI 어시스턴트입니다.", height = 200)

def print_messages():
    for chat_message in st.session_state["messages"]:
        main_tab2.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 이미지 캐시 저장(시간 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다....")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    # file_path = f"C:\\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache\files/{file.name}"
    file_path = rf"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\.cache\files{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

def generate_answer(image_filepath, system_prompt, user_prompt, model_name = "gpt-4o"):
    # 객체 생성
    llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    model_name=model_name,  # 모델명
)
    # 이미지 파일로 부터 질의(스트림 방식)
    multimodal_llm_with_prompt = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)
    answer = multimodal_llm_with_prompt.stream(image_filepath)

    return answer

# 초기화 버튼 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 정리
print_messages()

user_input = st.chat_input("내용을 입력하세요.")

# 경고 메시지 띄우기 위한 빈 영역
warning_msg = main_tab2.empty()

# 이미지 업로드 시
if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)   
    main_tab1.image(image_filepath)

#입력 시
if user_input:
    # 파일 업로드 확인
    if uploaded_file:
        image_filepath = process_imagefile(uploaded_file)

        # 답변 요청
        response = generate_answer(image_filepath, system_prompt, user_input, selected_model)

        main_tab2.chat_message("user").write(user_input)

        with main_tab2.chat_message("assistant"):
            container = st.empty()

            # AI_answer = answer 
            answer = ""
            for token in response:
                answer += token.content
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