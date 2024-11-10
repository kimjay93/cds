import streamlit as st
import numpy as np
import pandas as pd
import math
import joblib
import time
import util
import fcn
from PIL import Image
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

model = fcn.load_model()
scaler = fcn.load_scaler()
pca = fcn.load_pca()
image = fcn.load_images()

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
html_table_SHFT = util.rank_table_SHFT.to_html(escape=False, index=False)
styled_html_table_SHFT = f"""
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
{html_table_SHFT}
"""

st.set_page_config(layout="wide")        

with st.sidebar:
    st.title('페이지 선택')
    page = st.sidebar.selectbox('페이지 선택', ['🔍 신규 원유 평가', '📜 기존 원유 영향성 확인', 'ℹ️ Information'], index=0, label_visibility='collapsed')
    st.markdown('---')
    with st.container(border=True):
        st.subheader('📄 페이지 설명')  
        st.info('''
        - [🔍 ***신규 원유 평가***] 페이지는 특정 Crude 또는 VR Feed Property를 Input으로 하여 SHFT 영향성을 예측합니다.
        - [📜 ***기존 원유 영향성 확인***] 페이지는 기존 Crude Assay에 등록된 원유에 대한 SHFT 영향성을 확인합니다.
        - [ℹ️ ***Information***] 페이지는 예측 모델과 관련된 정보를 제공합니다.
        ''')
        
if page == '🔍 신규 원유 평가':
    tab1, tab2 = st.tabs(["원유 SHFT 영향성 예측", "VR Feed SHFT 예측"])
    with tab1:
        st.header('원유 VRHCR SHFT 영향성 예측')
        st.markdown('#### 아래에 Assay CSV 파일을 드래그하여 업로드하세요')
        file = st.file_uploader(label="", type=['csv'], label_visibility = 'collapsed')
        if file is not None:
            df_input = pd.read_csv(file, encoding='cp949')
            df_info = fcn.convert_df_info(df_input)
            df = fcn.convert_df(df_input)
            df_original = df.copy()
            # vis100_sqrt feature 추가 후 기존 vis100 feature 제거
            df['vis100_sqrt'] = np.sqrt(df['V(100) cSt.'])
            df.drop('V(100) cSt.', axis=1, inplace=True)
            # scaling
            df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
            # pca
            df_pca = pca.transform(df_scaled[['Ni ppm', 'V ppm']])
            df_scaled['Ni_V_PCA'] = df_pca
            df_scaled.drop(['Ni ppm', 'V ppm'], axis=1, inplace=True)
            # 예측
            x = df_scaled
            y = model.predict(x)  # Make predictions
            # y 단위 변경 (% -> ppm)
            y = y*10000
            # SHFT가 음수로 예측된 경우 0으로 변경
            for i in range(len(y)):
                if y[i] < 0:
                    y[i] = 0             
            y_df = pd.DataFrame({'SHFT_prediction ppm':y})
            df_feed = df_original.loc[:,'API':'V(100) cSt.']
            xy = pd.concat([df_info, y_df, df_feed], axis=1)

            # Apply the highlighting
            def highlight_first_column(s):
                return ['background-color: rgba(255, 255, 225, 0.5)' if s.name == xy.columns[2] else '' for _ in s]
            styled_xy = xy.sort_values(by='SHFT_prediction ppm', ascending=False).style.apply(highlight_first_column).format(precision=1)
            res = '🚫 worst' if y.max() > 25000 else '⚠️ Bad' if y.max() > 3700 else '👍 Good'
            rescolor = 'red' if y.max() > 25000 else 'orange' if y.max() > 3700 else 'green'

            if st.button('원유 분석', icon='🖱️'):
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
                with st.container(border=True):
                    st.markdown('#### 상세 결과')
                    st.write(styled_xy) 
                    bar_chart = alt.Chart(xy).mark_bar().encode(x='Feedstock코드:O', y='SHFT_prediction ppm:Q')
                    horizontal_line_yellow = alt.Chart(pd.DataFrame({'y': [800]})).mark_rule(color='yellow').encode(y='y:Q')
                    horizontal_line_red = alt.Chart(pd.DataFrame({'y': [1500]})).mark_rule(color='red').encode(y='y:Q')
                    final_chart = alt.layer(bar_chart, horizontal_line_yellow, horizontal_line_red)
                    st.altair_chart(final_chart, use_container_width=True)
                    
    with tab2:
        st.header('VR Feed VRHCR SHFT 영향성 예측')
        st.markdown('#### 아래에 VR Feed Property를 입력하세요 (Default=2024년 평균)')
        names = ["API", "Sulfur %", "Nitrogen ppm", "MCRT wt. %", "Ni ppm", "V ppm", "Na ppm", "Fe ppm", "Ca ppm", "V(100) cSt."]
        default_values = [2.9, 5.6, 3533.5, 23.6, 50.5, 169.0, 7.6, 11.6, 3.6, 4032.3]
        numbers = []
        columns = st.columns(10)
        for i in range(10):
            with columns[i]:
                number = st.number_input(names[i], min_value=0.0, value=default_values[i], format='%.0f')
                numbers.append(number)
                
        st.markdown('#### 아래에 Operation Data를 입력하세요 (Default=2024년 평균)')
        names_op = ['612FC702.PV', '612TC138A.PV', '612FC124.PV', '612TC168A.PV', '612FC129.PV', '612TC170A.PV', '612FC195A.PV']
        default_values_op = [40.0, 319.1, 11.9, 356.2, 13.1, 408.0, 9.6]
        numbers_op = []
        columns_op = st.columns(7)
        for i in range(7):
            with columns_op[i]:
                number_op = st.number_input(names_op[i], min_value=0.0, value=default_values_op[i], format="%.0f")
                numbers_op.append(number_op)
        
        names_op2 = ['612FI109.PV', '612FI740.PV', '612FC123.MV', '612SI101.PV', '612SI102.PV', '612AI104E.PV', '612AI107E.PV','partial_p']
        default_values_op2 = [194.1, 58.4, 44.3, 1311.4, 1445.4, 873.9, 914.1, 156.8]
        numbers_op2 = []
        columns_op2 = st.columns(8)
        for i in range(8):
            with columns_op2[i]:
                number_op2 = st.number_input(names_op2[i], min_value=0.0, value=default_values_op2[i], format="%.0f")
                numbers_op2.append(number_op2)
                
        if st.button('SHFT 예측값 계산', icon='🖱️'):
            df = pd.DataFrame({'var':names_op+names_op2+names, 'val':numbers_op+numbers_op2+numbers})
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df_original = df.copy()
            # vis100_sqrt feature 추가 후 기존 vis100 feature 제거
            df['vis100_sqrt'] = math.sqrt(df['V(100) cSt.'])
            df.drop('V(100) cSt.', axis=1, inplace=True)
            # scaling
            df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
            # pca
            df_pca = pca.transform(df_scaled[['Ni ppm', 'V ppm']])
            df_scaled['Ni_V_PCA'] = df_pca
            df_scaled.drop(['Ni ppm', 'V ppm'], axis=1, inplace=True)
            # 예측
            x = df_scaled
            y = model.predict(x)  # Make predictions
            # y 단위 변경 (% -> ppm)
            y = y*10000
            # SHFT가 음수로 예측된 경우 0으로 변경
            if y < 0:
                y = 0             
            st.divider()
            fcn.analyzeDF()
            st.success('예측 완료', icon="✅")
            rescolor_SHFT = 'red' if y.max() > 1500 else 'orange' if y.max() > 800 else 'green'
            col1, col2 = st.columns([1,2])
            with col1:
                with st.container(border=True):
                    st.markdown("#### SHFT 예측값")
                    st.markdown(f"#### <span style='color:{rescolor_SHFT}'> {y.round(1)} </span>", unsafe_allow_html=True)
            with col2:
                with st.container(border=True):
                    st.markdown("#### 등급 기준")    
                    st.markdown(styled_html_table_SHFT, unsafe_allow_html=True)
            # horizontal bar 하나 넣으면 좋을 듯??

elif page == '📜 기존 원유 영향성 확인':
    st.header('기존 원유 VRHCR 영향성(SHFT) 확인')
    df_assay_info = pd.read_csv('./df_assay_info.csv')
    st.markdown('##### ※ SHFT 영향성')
    st.markdown('- 전체 Crude Assay(159개)에 대해 Rank=80인 중위값의 SHFT 영향성을 1.0으로 기준하여 산출한 SHFT 영향성') 
    st.markdown('- 높을수록 SHFT 상승 위험이 높음')
    def highlight_SHFT_column(s):
        return ['background-color: rgba(255, 255, 225, 0.5)' if s.name == styled_df_assay.columns[3] else '' for _ in s]
    styled_df_assay = df_assay_info.style.apply(highlight_SHFT_column).format(precision=1)
    st.write(styled_df_assay)
    
    # 트렌드 기능(아래)은 일단 보류 - 유의미한 정보 부족
#     feed_property_choice_box = st.selectbox('Feed Property - SHFT 영향성 확인', df_assay_info.columns[4:])
#     if feed_property_choice_box:
#         df_feedp_SHFT = df_assay_info[[feed_property_choice_box, 'SHFT 영향성']]
#         fig = make_subplots(specs=[[{"secondary_y": True}]])
#         fig.add_trace(
#             go.Line(x=df_feedp_SHFT.index, y=df_feedp_SHFT[feed_property_choice_box], name="Feed Property"),
#             secondary_y=False,
#         )

#         fig.add_trace(
#             go.Line(x=df_feedp_SHFT.index, y=df_feedp_SHFT['SHFT 영향성'], name="SHFT 영향성"),
#             secondary_y=True,
#         )

#         # Set titles for y-axes
#         fig.update_yaxes(title_text="Feed Property", secondary_y=False)
#         fig.update_yaxes(title_text="SHFT 영향성", secondary_y=True)

#         # Display in Streamlit
#         st.plotly_chart(fig)

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
            st.image(image['pre_act_image'], caption="XY Plot of Prediction-Actual SHFT", use_container_width=True, width=1)
        with col2:
            st.markdown("##### SHFT 예측 - 실제 Trend 일부 ('18-'21)")    
            st.image(image['pre_act_trend_image'], caption="Prediction(blue)-Actual(orange) SHFT trend, 2018-2021", use_container_width=True, width=1)
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
            st.image(image['shap_image'], caption="SHAP (SHapley Additive exPlanations)", use_container_width=True, width=1)
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
