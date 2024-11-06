
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os


def rag_setup(api_key, file_path, chunk_size=600, chunk_overlap=50):

    os.environ["OPENAI_API_KEY"] = api_key
    
    #1. 문서 로드
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    #2. 데이터 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(docs)

    #3. 임베딩, 벡터 스토어
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_documents, embeddings)

    #4.Retriever
    retriever = vectorstore.as_retriever()

    return retriever

prompt_template = """
                    당신은 지능적이고 창의적인 AI 어시스턴트입니다. 주어진 질문에 대해 관련 정보를 조합하여 통찰력 있는 답변을 제공해야 합니다.
                    #Context:
                    {context}
                    #Question:
                    {question}
                    위의 정보를 바탕으로 다음 작업을 수행하세요:
                    1. 가중치가 높은 정보를 우선적으로 고려하되, 모든 관련 정보를 종합적으로 분석하세요.
                    2. 정보들 사이의 연관성을 파악하고, 이를 바탕으로 통찰력 있는 답변을 작성하세요.
                    3. 답변에 사용된 주요 정보의 출처를 간략히 언급하여 신뢰성을 높이세요.
                    4. 질문과 직접적으로 관련이 없는 정보는 과감히 제외하세요.
                    5. 필요하다면 제공된 정보를 넘어서는 합리적인 추론을 포함하세요.
                    답변은 한국어로 작성하며, 전문적이면서도 이해하기 쉽게 설명해주세요.
                    """

def create_rag_chain(retriever, model_name='gpt-4o'):

    #5.프롬프트 생성
    prompt = PromptTemplate.from_template(
        prompt_template
    )

    # prompt = PromptTemplate.from_template(
    #     """ 너는 삼성전자 SW팀에서 개발을 맡고 있는 책임자야
    #     참조하는 문서를 이용하여 삼성 가우스에 대한 평가문항을
    #     10문제만 출제해줘 그중 8문제는 객관식, 2문제는 주관식이야..

    # #Context:
    # {context}

    # #Question:
    # {question}

    # #Answer:"""
    # )

    #6. 모델 생성
    model = ChatOpenAI(model_name = model_name, temperature=0)

    #체인생성
    #RunnablePassthrough : RunnablePassthrough
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain
