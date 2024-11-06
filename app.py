import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import util
from PIL import Image

def load_model():
    with open("xgb_23rd.pkl", "rb") as file:
        return pickle.load(file)

def load_images():
    return {
        "pre_act_image" : Image.open('pre_act.png'),
        "pre_act_trend_image" : Image.open('pre_act_trend.png'),
        "shap_image" : Image.open('SHAP.png')
    }

model = load_model()
image = load_images()

def analyzeDF():
    msg = st.toast('Reading...', icon="📖")
    time.sleep(1)
    msg.toast('Analyzing...', icon="🧐")
    time.sleep(2)
    msg.toast('Ready!', icon = "🔔")

html_table = util.rank_table.to_html(escape=False, index=False)
styled_html_table = f"""
<style>
    table {{
        width: 100%;  /* Optional: Set table width */
        border-collapse: collapse; /* Optional: collapse borders */
    }}
    th, td {{
        text-align: left;  /* Align headers and cells to the left */
        border: 1px solid #ccc; /* Optional: Add border to cells */
        padding: 8px; /* Optional: Add padding */
    }}
</style>
{html_table}
"""

st.markdown("""
<style>
[role=radiogroup]{
    gap: 1rem;
}
</style>
""",unsafe_allow_html=True)

st.markdown("""
    <style>
    .stContainerStyle  {
        background-color: #f0f0f5; 
       
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title('페이지 선택')
    page = st.sidebar.selectbox('페이지 선택', ['신규 원유 평가', '기존 원유 평가', 'Information'], index=0, label_visibility='collapsed')
    st.markdown('---')
    with st.container(border=True):
        st.markdown('<div class="stContainerStyle">', unsafe_allow_html=True)
        st.subheader('페이지 설명')  
        st.markdown('- "신규 원유 평가" 페이지는 특정 원유의 blending 비율 0 ~ 30% 에서의 예상 VR Feed Property를 Input으로 하여 SHFT를 예측합니다.')
        st.markdown('- "기존 원유 평가" 페이지는 기존 Crude Assay에 등록된 원유에 대해 blending 비율 0 ~ 30% 에서의 SHFT 예측 결과를 제공합니다.')
        st.markdown('- "Information" 페이지는 예측 모델 관련 정보를 제공합니다.')
        st.markdown('</div>', unsafe_allow_html=True)
    
      

if page == '신규 원유 평가':
    st.header('신규 원유 VRHCR 영향성(SHFT) 예측')
    st.markdown('#### 아래에 CSV 파일을 드래그하여 업로드하세요')
    file = st.file_uploader(label="", type=['csv'], label_visibility = 'collapsed')
    
    if file is not None:
        df = pd.read_csv(file, index_col = 'Datetime')
        x = df.iloc[:,:-1].round(3)
        y = model.predict(x)  # Make predictions
        y_df = pd.DataFrame({'SHFT_prediction':y})
        x_df = x.reset_index(drop=True)
        xy = pd.concat([y_df, x_df], axis=1)

        # Apply the highlighting
        def highlight_first_column(s):
            return ['background-color: rgba(255, 255, 225, 0.5)' if s.name == xy.columns[0] else '' for _ in s]
        styled_xy = xy.style.apply(highlight_first_column)
        print(styled_xy)
        res = '🚫 worst' if y.max() > 0.15 else '⚠️ Bad' if y.max() > 0.1 else '👍 Good'
        rescolor = 'red' if y.max() > 0.15 else 'orange' if y.max() > 0.1 else 'green'

        if st.button('분석', icon='🖱️'):
            st.divider()
            analyzeDF()
            st.success('예측 완료', icon="✅")
            col1, col2 = st.columns([1,2])
            with col1:
                with st.container(border=True):
                    st.markdown("#### 등급")
                    st.markdown(f"#### <span style='color:{rescolor}'> {res} </span>", unsafe_allow_html=True)
            with col2:
                with st.container(border=True):
                    st.markdown("#### 등급 기준")    
                    st.markdown(styled_html_table, unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('#### 상세 결과')
                st.write(styled_xy) 
                st.line_chart(xy['SHFT_prediction'])

elif page == '기존 원유 평가':
# if st.button('기존 원유 평가', key='기존원유평가'):
    st.header('기존 원유 VRHCR 영향성(SHFT) 확인')
    st.markdown('#### 아래에서 원유를 선택하세요')
    selection = st.selectbox('', ('1st','2nd'), label_visibility='collapsed')
    if selection == '1st':
        col1, col2 = st.columns([1,2])
        with col1:
            with st.container(border=True):
                st.markdown("#### 등급")
                st.write('BAD', color='red')
        with col2:
            with st.container(border=True):
                st.markdown("#### 등급 기준")    
                st.markdown(styled_html_table, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('#### 상세 결과')             

elif page == 'Information':
    tab1, tab2, tab3 = st.tabs(["🔽 모델링 Input Data", "🔼 모델링 결과", "💧 원유 평가 기준"])
    
    with tab1:
        st.markdown(f'''
        - ##### 학습 데이터  
            - ***{util.input_data}***
            - 총 2118개 데이터 사용
        ---
        - ##### 사용된 변수 
            - ***{util.features}***
            - Feed Property \n
                {util.feed_property}
            - 운전 변수 \n
                {util.op_data_new}
        ---
        - ##### 학습 알고리즘
            - ***XGBoost Regressor***
            - 상세 정보 (Link) : https://xgboost.readthedocs.io/en/stable/
            - 상세 정보 (YouTube) : https://www.youtube.com/watch?v=TyvYZ26alZs
        ''')
        
    with tab2:
        st.markdown(f'''
        ##### 예측 성능
            - r2 score = 0.771
            - MAPE: 13.2 %
            - RMSE: 0.011 (SHFT %)
        ---
        ''')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('##### SHFT 실제(X) - 예측(Y) 비교 그래프')
            st.image(image['pre_act_image'], caption="XY Plot of Prediction-Actual SHFT", use_column_width=True, width=1)
        with col2:
            st.markdown("##### SHFT 예측 - 실제 Trend 일부 ('18-'21)")    
            st.image(image['pre_act_trend_image'], caption="Prediction(blue)-Actual(orange) SHFT trend, 2018-2021", use_column_width=True, width=1)
        st.markdown('---')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('##### Feature Importance')
            st.write("각 feature가 예측값(SHFT)에 얼마나 기여하는지를 나타내는 상대적인 지표")
            util.fi_df
        with col2:
            st.markdown('##### SHAP 분석 결과')
            st.write("예측값(SHFT)에 대한 각 feature의 기여도를 기반으로 중요도를 할당함")
            st.image(image['shap_image'], caption="SHAP (SHapley Additive exPlanations)", use_column_width=True, width=1)
        st.markdown('---')
          
    with tab3:
        st.markdown('##### 평가 방법')
        st.markdown('##### Base Case')
        st.markdown('---')
            
