
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage

from rag import rag_setup, create_rag_chain
import streamlit as st
import os

st.title('PDF 기반의 챗봇')

if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')

#============= Session 저장 코드 =============================

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if "store" not in st.session_state:
    st.session_state['store'] = {}

#Streamlit의 Sessionless을 보완하기 위한 코드
#대화 기록이 없다면
def add_history(role, message):
    st.session_state['chat_history'].append(ChatMessage(role=role, content=message))

#session state에 저장된 코드 출력
def print_history():
    for chat_msg in st.session_state['chat_history']:
        #메세지 출력
        st.chat_message(chat_msg.role).write(chat_msg.content)

# ===============================================================

with st.sidebar:
    api_key = st.text_input("Enter your API-Iey", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

with st.sidebar:
    selected_model = st.selectbox('모델선택', ['gpt-4o-mini','gpt-4o'], index=0)

with st.sidebar:
    uploaded_file = st.file_uploader('pdf file 업로드', type=['pdf'])

#파일 캐싱
@st.cache_resource(show_spinner='업로드한 파일을 처리중입니다. ')
def embed_file(file):
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)

    retriever = rag_setup(api_key, file_path, 300, 50)
    return retriever


if uploaded_file:
    retriever = embed_file(uploaded_file)
    rag_chain = create_rag_chain(retriever)
    #rag_chain = create_rag_quiz_chain(retriever)

#====== Session 저장 코드
print_history()


user_input = st.chat_input('궁금한 내용을 입력해 주세요')


if user_input:
    #사용자 입력 출력
    st.chat_message('user').write(user_input)

    with st.chat_message('ai'):
        chat_container = st.empty()
        answer = rag_chain.stream(user_input)

        #스트리밍 출력
        ai_answer=""
        for token in answer:
            ai_answer += token
            chat_container.markdown(ai_answer)


    #===== 기존 기록 저장
    add_history('user', user_input)
    add_history('ai', ai_answer)
