import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
import glob
import os
# from langchain import hub

# API key 정보 로드
load_dotenv()

st.title("나의 챗 GPT")

if "messages" not in st.session_state:
    # 대화기록 저장
    st.session_state["messages"] = []

# 버튼 지정
with st.sidebar:
    # 초기화 버튼임
    clear_btn = st.button("대화 초기화")
    
    prompt_files = glob.glob(r"C:\Users\progr\Documents\.venv\19-Streamlit\MyProject\prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해 주세요.", prompt_files , index = 0)
    task_input = st.text_input("TASK 입력", "")

# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def create_chain(prompt_filepath, task=""):
    prompt = load_prompt(prompt_filepath, encoding="utf-8")
    if task:
        prompt = prompt.partial(task=task)

    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    output_parser =  StrOutputParser()
 
    # 체인 생성
    chain = prompt | llm | output_parser
    return chain

user_input = st.chat_input("내용을 입력하세요.")

# 초기화 버튼 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 정리
print_messages()

#입력 시
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    chain = create_chain(selected_prompt, task = task_input)
 
    # 스트리밍 호출
    response = chain.stream({"question": user_input})
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