import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import util
import fcn
from PIL import Image
import altair as alt


model = fcn.load_model()
image = fcn.load_images()
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

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

with st.sidebar:
    st.title('페이지 선택')
    page = st.sidebar.selectbox('페이지 선택', ['🔍 신규 원유 평가', '📜 기존 원유 평가 결과', 'ℹ️ Information'], index=0, label_visibility='collapsed')
    st.markdown('---')
    with st.container(border=True):
        st.subheader('📄 페이지 설명')  
        st.info('''
        - [🔍 ***신규 원유 평가***] 페이지는 특정 Crude 또는 VR Feed Property를 Input으로 하여 SHFT를 예측합니다.
        - [📜 ***기존 원유 평가 결과***] 페이지는 기존 Crude Assay에 등록된 원유에 대한 SHFT 예측 결과를 제공합니다.
        - [ℹ️ ***Information***] 페이지는 예측 모델과 관련된 정보를 제공합니다.
        ''')   

if page == '🔍 신규 원유 평가':
    st.header('신규 원유 VRHCR 영향성(SHFT) 예측')
    st.markdown('#### 아래에 CSV 파일을 드래그하여 업로드하세요')
    file = st.file_uploader(label="", type=['csv'], label_visibility = 'collapsed')
    if file is not None:
        df_input = pd.read_csv(file, encoding='cp949')
        df_info = fcn.convert_df_info(df_input)
        df = fcn.convert_df(df_input)
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        df_pca = pca.transform(df_scaled[['Ni ppm', 'V ppm']])
        df_scaled['Ni_V_PCA'] = df_pca
        df_scaled.drop(['Ni ppm', 'V ppm'], axis=1, inplace=True)
        x = df_scaled
        y = model.predict(x)  # Make predictions
        y = y*10000
        y_df = pd.DataFrame({'SHFT_prediction':y})
        x_df = x.reset_index(drop=True)
        df_feed = df.iloc[:,16:-1]
        xy = pd.concat([df_info, y_df, df_feed], axis=1)

        # Apply the highlighting
        def highlight_first_column(s):
            return ['background-color: rgba(255, 255, 225, 0.5)' if s.name == xy.columns[2] else '' for _ in s]
        styled_xy = xy.style.apply(highlight_first_column)
        print(styled_xy)
        res = '🚫 worst' if y.max() > 1500 else '⚠️ Bad' if y.max() > 1000 else '👍 Good'
        rescolor = 'red' if y.max() > 1500 else 'orange' if y.max() > 1000 else 'green'

        if st.button('분석', icon='🖱️'):
            st.divider()
            fcn.analyzeDF()
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
                    st.markdown('※ 24년 운전 조건 기준으로 예상 SHFT 산출했으며 절대값은 운전 조건에 따라 달라질 수 있음')
            with st.container(border=True):
                st.markdown('#### 상세 결과')
                st.write(styled_xy) 
                bar_chart = alt.Chart(xy).mark_bar().encode(x='Feedstock코드:O', y='SHFT_prediction:Q')
                horizontal_line_yellow = alt.Chart(pd.DataFrame({'y': [1000]})).mark_rule(color='yellow').encode(y='y:Q')
                horizontal_line_red = alt.Chart(pd.DataFrame({'y': [1500]})).mark_rule(color='red').encode(y='y:Q')

                final_chart = alt.layer(bar_chart, horizontal_line_yellow, horizontal_line_red)
                st.altair_chart(final_chart, use_container_width=True)



elif page == '📜 기존 원유 평가 결과':
# if st.button('기존 원유 평가', key='기존원유평가'):
    st.header('기존 원유 VRHCR 영향성(SHFT) 확인')
    st.markdown('#### 아래에서 원유를 선택하세요')
    selection = st.selectbox('', ('BAV','I$H'), label_visibility='collapsed')
    if selection == 'BAV':
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

elif page == 'ℹ️ Information':
    st.header('예측 모델 관련 정보')
    tab1, tab2, tab3 = st.tabs(["🔽 모델링 Input Data", "🔼 모델링 결과", "📈 기존 분석법"])
    with tab1:
        st.markdown(f'''
        - ##### 학습 데이터  
            - ***{util.input_data}***
            - 총 2241개 데이터 사용
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
            - r2 score = 0.804
            - MAPE: 12.4 %
            - RMSE: 0.008 (SHFT %)
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
            st.write(util.fi_df)
            # df 대신 horizontal bar graph로 보여주자
            
        with col2:
            st.markdown('##### SHAP 분석 결과')
            st.write("예측값(SHFT)에 대한 각 feature의 기여도를 기반으로 중요도를 할당함")
            st.image(image['shap_image'], caption="SHAP (SHapley Additive exPlanations)", use_column_width=True, width=1)
        st.markdown('---')
    with tab3:
        st.markdown('#### SEDEX')
        st.markdown('''
        1. SEDEX 관련 설명 
        2. SEDEX - SHFT 관계성
        3. 
        ''')
        
        st.markdown('#### CII')
        st.markdown('''
        1. CII 관련 설명 
        2. CII - SHFT 관계성
        3. 
        ''')
        
        st.markdown('#### FOI')
        st.markdown('''
        1. FOI 관련 설명 
        2. FOI - SHFT 관계성
        3. 
        ''')
        
