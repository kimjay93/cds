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

st.set_page_config(
    page_title="VRHCR Sediment Prediction",
    page_icon="./ci_signature_en.png",  # Optional: You can use an emoji or a local image file path
    layout="wide"  # Optional: "wide" for full-width, "centered" for centered layout
)
    
with st.sidebar:
    st.title('페이지 선택')
    page = st.sidebar.selectbox('페이지 선택', ['🔍 신규 원유 / VR Feed 평가', '📚 기존 원유 영향성 확인', 'ℹ️ Information'], index=0, label_visibility='collapsed')
    st.markdown('---')
    with st.container(border=True):
        st.subheader('📄 페이지 설명')  
        st.info('''
        - [🔍 ***신규 원유 / VR Feed 평가***] 페이지는 특정 Crude 또는 VR Feed Property를 Input으로 하여 Sediment(SHFT) 영향성을 예측합니다.
        - [📚 ***기존 원유 영향성 확인***] 페이지는 기존 Crude Assay에 등록된 원유에 대한 Sediment(SHFT) 영향성을 확인합니다.
        - [ℹ️ ***Information***] 페이지는 예측 모델과 관련된 정보를 제공합니다.
        ''')
        
if page == '🔍 신규 원유 / VR Feed 평가':
    tab1, tab2 = st.tabs(["원유 Sediment(SHFT) 영향성 예측", "VR Feed Sediment(SHFT) 예측"])
    with tab1:
        st.header('🔍 원유 VRHCR Sediment(SHFT) 영향성 예측')
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

            # SHFT가 음수로 예측된 경우 0으로 변경
            for i in range(len(y)):
                if y[i] < 0:
                    y[i] = 0       
            # SHFT가 10,000 ppm(1%)를 넘으면 10,000으로 capping
            for i in range(len(y)):
                if y[i] > 10000:
                    y[i] = 10000
                
        
            y_df = pd.DataFrame({'SHFT_prediction ppm':y})
            df_feed = df_original.loc[:,'API':'V(100) cSt.']
            xy = pd.concat([df_info, y_df, df_feed], axis=1)

            # Apply the highlighting
            def highlight_first_column(s):
                return ['background-color: rgba(255, 255, 225, 0.5)' if s.name == xy.columns[2] else '' for _ in s]
            styled_xy = xy.sort_values(by='SHFT_prediction ppm', ascending=False).style.apply(highlight_first_column).format(precision=1)
            # 10000: 상위 5개, 2700: 상위 15개, 1250: 상위 50% 
            res = '🔴 worst' if y.max() >= 10000 else '🟠 Bad' if y.max() > 2700 else '🟡 Moderate' if y.max() > 1250 else '🟢 Good'   
            rescolor = 'red' if y.max() >= 10000 else 'orange' if y.max() > 2700 else 'yellow' if y.max() > 1250 else 'green'

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
                    horizontal_line_yellow = alt.Chart(pd.DataFrame({'y': [1250]})).mark_rule(color='yellow').encode(y='y:Q')
                    horizontal_line_orange = alt.Chart(pd.DataFrame({'y': [2700]})).mark_rule(color='orange').encode(y='y:Q')
                    horizontal_line_red = alt.Chart(pd.DataFrame({'y': [10000]})).mark_rule(color='red').encode(y='y:Q')
                    final_chart = alt.layer(bar_chart, horizontal_line_yellow, horizontal_line_orange, horizontal_line_red)
                    st.altair_chart(final_chart, use_container_width=True)
                    
    with tab2:
        st.header('🔍 VR Feed VRHCR Sediment(SHFT) 영향성 예측')
        st.markdown('#### 아래에 VR Feed Property를 입력하세요 (Default=2024년 평균)')
        names = ["API", "Sulfur %", "Nitrogen ppm", "MCRT wt. %", "Ni ppm", "V ppm", "Na ppm", "Fe ppm", "Ca ppm", "V(100) cSt.", "ASP(C7) wt. %"]
        default_values = [2.9, 5.6, 3533.5, 23.6, 50.5, 169.0, 7.6, 11.6, 3.6, 4032.3, 15.2] # 2024년 평균값
        numbers = []
        columns = st.columns(11)
        for i in range(11):
            with columns[i]:
                # number = st.number_input(names[i], min_value=0.0, value=default_values[i], format='%.1f')
                number = st.text_input(names[i], value=default_values[i])
                numbers.append(number)
                
        st.markdown('#### 아래에 Operation Data를 입력하세요 (Default=2024년 평균)')
        names_op = ['612FC702.PV', '612TC138A.PV', '612FC124.PV', '612TC168A.PV', '612FC129.PV', '612TC170A.PV', '612FC195A.PV']
        default_values_op = [40.0, 319.1, 11.9, 356.2, 13.1, 408.0, 9.6] # 2024년 평균값
        numbers_op = []
        columns_op = st.columns(7)
        for i in range(7):
            with columns_op[i]:
                # number_op = st.number_input(names_op[i], min_value=0.0, value=default_values_op[i], format="%.1f")
                number_op = st.text_input(names_op[i], value=default_values_op[i])
                numbers_op.append(number_op)
        
        names_op2 = ['612FI109.PV', '612FI740.PV', '612FC123.MV', '612SI101.PV', '612SI102.PV', '612AI104E.PV', '612AI107E.PV','partial_p']
        default_values_op2 = [194.1, 58.4, 44.3, 1311.4, 1445.4, 873.9, 914.1, 156.8] # 2024년 평균값
        numbers_op2 = []
        columns_op2 = st.columns(8)
        for i in range(8):
            with columns_op2[i]:
                # number_op2 = st.number_input(names_op2[i], min_value=0.0, value=default_values_op2[i], format="%.1f")
                number_op2 = st.text_input(names_op2[i], value=default_values_op2[i])
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
            y = model.predict(x)[0]  # Make predictions
            # SHFT가 음수로 예측된 경우 0으로 변경
            if y < 0:
                y = 0             
            st.divider()
            fcn.analyzeDF()
            st.success('예측 완료', icon="✅")
            rescolor_SHFT = 'red' if y > 1500 else 'orange' if y > 800 else 'green'
            col1, col2 = st.columns([2,1])
            # with col1:
            #     with st.container(border=True):
            #         st.markdown("#### SHFT 예측값")
            #         st.markdown(f"#### <span style='color:{rescolor_SHFT}'> {y} </span>", unsafe_allow_html=True)
            with col1:
                # horizontal bar 표시
                max_limit = 1500
                high_limit = 800
                progress_percentage = y / max_limit * 100
                value=y
    
                # Set up color-based labels and thresholds
                if value >= max_limit:
                    color = "red"
                    label = "Max"
                elif value >= high_limit:
                    color = "orange"
                    label = "High"
                else:
                    color = "green"
                    label = "Normal"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': "SHFT 예측값 (ppm)", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, max_limit*1.05], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': color},
                        'steps': [{'range': [0, high_limit], 'color': 'lightgreen'}, {'range': [high_limit, max_limit], 'color': '#FFCC99'}, {'range': [max_limit, max_limit*1.05], 'color':'lightcoral'}],
                        'threshold': {'line': {'color': "red", 'width': 8}, 'thickness': 0.75, 'value': max_limit}
                    },
                ))                
                # Display the bar in Streamlit
                st.plotly_chart(fig)

            with col2:
                with st.container(border=True):
                    st.markdown("#### 등급 기준")    
                    st.markdown(styled_html_table_SHFT, unsafe_allow_html=True)

            

elif page == '📚 기존 원유 영향성 확인':
    st.header('📚 기존 원유 VRHCR Sediment(SHFT)* 영향성 확인')
    df_assay_info = pd.read_csv('./df_assay_info.csv')
    st.markdown('- 전체 Crude Assay(159개)에 대해 SHFT 중위값(Rank #80)을 1로 기준하여 SHFT 영향성을 나타내는 간접 지표로 산출 → 높을수록 SHFT 상승 위험성 높음') 
    def highlight_SHFT_column(s):
        return ['background-color: rgba(255, 255, 225, 0.5)' if s.name == styled_df_assay.columns[3] else '' for _ in s]
    styled_df_assay = df_assay_info.style.apply(highlight_SHFT_column).format(precision=1)
    st.write(styled_df_assay)
    
    # 트렌드 기능 - 상위 10개에 대한 SHFT
    st.markdown('##### SHFT 영향성 상위 10개 Crude')
    sorted_df_assay_top = df_assay_info.sort_values(by='SHFT 영향성', ascending=False).head(10)
    sorted_df_assay_top['Feedstock코드'] = pd.Categorical(sorted_df_assay_top['Feedstock코드'], categories=sorted_df_assay_top['Feedstock코드'], ordered=True)
    st.bar_chart(data=sorted_df_assay_top, x='Feedstock코드', y='SHFT 영향성', x_label='원유', y_label='SHFT 영향성', use_container_width=True)


elif page == 'ℹ️ Information':
    st.header('ℹ️ VRHCR Sediment(SHFT) 예측 모델 관련 정보')
    tab1, tab2, tab3, tab4, tab5= st.tabs(["🔽 모델링 Input Data", "🔼 모델링 결과", "📜 모델링 History", "📊 (참고) Sediment 관련 지표", "📘 (참고) SHFT"])
    
    with tab1:
        st.markdown('#### 🧮 학습 데이터')
        st.markdown(f'''
        - **{util.input_data}**
        - 학습에 사용된 데이터 수(row): :blue[2086개]
        - 학습에 사용된 변수 개수(column): :blue[{util.features}]
            - :blue[Feed Property (11개)]
                - {util.feed_property}
            - :blue[운전 변수 (15개)]
                - {util.op_data_new}
        ---
        ''')
        st.markdown('#### 🤖 학습 알고리즘')
        st.markdown('''
        ##### ***:blue[Linear Regression]***
        - 상세 정보 (Wikipedia) : https://en.wikipedia.org/wiki/Linear_regression
        - 상세 정보 (YouTube) : https://www.youtube.com/watch?v=7ArmBVF2dCs
        ---
        ''')
        
    with tab2:
        st.markdown('#### 🎯 예측 성능')
        st.markdown('''
        - **:blue[r2 score = 0.584]**
        - **:blue[MAPE: 22.2 %]**
        - **:blue[RMSE: 130 (SHFT ppm)]**
        ---
        ''')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('#### SHFT 실제(X) - 예측(Y) 비교')
            st.write('2012-2024년 데이터 중 학습(fit)에 사용되지 않은 Data에 대한 예측값과 실제값의 XY Plot')
        with col2:
            st.markdown("#### SHFT 실제 - 예측 Trend 비교")   
            st.write('2012-2024년 데이터 중 학습(fit)에 사용되지 않은 Data에 대한 예측값과 실제값의 SHFT Trend')
        with col3:
            st.markdown('#### Linear Correlation')
            st.write('각 feature와 SHFT 간의 대략적인 선형 관계(상대적인 연관도와 방향성) 정도')
        st.markdown('---')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image['pre_act_image'], caption="XY Plot of Prediction-Actual SHFT", use_container_width=True, width=1)
        with col2: 
            st.image(image['pre_act_trend_image'], caption="Prediction(blue)-Actual(orange) SHFT trend", use_container_width=True, width=1)
        with col3:
            st.image(image['corr_image'], caption='Correlation between SHFT and each feature', use_container_width=True, width=1)
        st.markdown('---')
        
    with tab3:
        col1, col2, col3 = st.columns([4.9, 0.2, 4.9])
        with col1:
            st.markdown('#### 🧮 Data 선정 History')
            st.markdown('''
            ---
            ###### :blue[운전데이터]
            1. Train 1, 2의 Feed ~ Reactor ~ C109/209 구간의 주요 운전 데이터 선택
            2. Train 1, 2의 Data가 대부분 동일한 패턴을 보여 Train 2 Data 제거
            3. Domain Knowledge에 기반하여 SHFT에 영향이 없을 것으로 예상되는 운전 변수 제거
            4. 이상치 제거 
                - 1차 시도: 통계적 방법으로 일괄 제거한 경우 너무 많은 데이터가 삭제되어 성능 하락함
                - 2차 시도: 각 Feature 별로 하나씩 확인하여 공정 특이사항 또는 계기 오류에 해당하는 날짜만 제거
            5. SHFT와의 Correlation 낮은 Feature에 대해 Domain Knowledge에 근거하여 일부 제거
            6. Tree 기반 Regression Modeling 이후 SHFT와의 feature importance 낮은 변수에 대해 Domain Knowledge에 근거하여 일부 제거
            7. 2011년(Initial Start Up 이후) 데이터의 경우 SHFT 분포가 넓어 학습 데이터에 포함하려 했으나, 
            데이터의 불규칙성이 커 오히려 예측 성능을 저하시키는 것으로 확인, 최종적으로 제외함
            8. TA 주기(TA 직후, 1년/2년/3년 경과)에 따른 SHFT 차이를 확인하기 위해 해당 변수를 추가했으나, 
            특정 연도(2012, 2019)에 SHFT가 크게 높아 연도에 따른 경향성이 뚜렷하지 않아 제외함
            9. SHFT 값의 분포가 왼쪽으로 치우쳐 있어(positively skewed) Linear Regression Model 성능에 영향을 주는 것으로 판단, 
            skewness 개선을 위해 log transformation 등 여러 가지 시도한 결과, 
            정규분포에 가까워짐에 따라 예측 성능 자체는 소폭 상승했으나, SHFT에 대한 각 feature들의 영향성이 왜곡되는 것으로 판단하여 최종적으로 채택하지 않음
            10. 모델링 과정에서 모델 성능과 Domain Knowledge 활용하여 feature 추가 / 삭제 / Engineering 과정 반복 수행
             
            ---
            ###### :blue[Feed Property]
            1. Lab Data가 없는 날짜의 데이터 제외 → 전체 약 4,500개 Data 중 절반 가량 제외됨
                - 운전 변수 데이터는 매일(주 7일) 존재하는 반면 VR Feed Lab Data는 주 3-4일만 존재함
                - 예측하고자 하는 y 값(SHFT)이 Lab Data에 속하기 때문에, 운전 변수만 있는 데이터는 학습할 의미가 없다고 판단하여 제외함
            2. 데이터 개수가 부족한 (전체 Dataset의 절반 이하) Feature 제외
                - 'SARAII', 'Resin', 'SARA', 'Asphaltene', 'Saturates', 'Chloride', 'Aromatics'
                - 해당 분석값이 모두 존재하는 데이터만 학습에 사용할 경우 데이터 개수가 너무 줄어들어(4,500개 → 500~1000개 수준) 예측 성능 크게 저하됨
            3. 서로 중복되는 항목 제외 
                - API와 Specific Gravity, Viscosity @ 100'C와 Viscosity @ 175'C 등
            4. Feature Engineering
                - Sediment 관련 지표인 SEDEX 및 FOI(Feed Operability Index) 계산식에 사용되는 합성 변수(Nitrogen/Sulfur Ratio 등) 사용 시도
                - 예측 성능 개선에 큰 효과가 없고, 기존 변수와 영향성이 중첩되어 최종적으로는 제외함
            5. SEDEX 및 FOI의 경우, 실제로 SHFT와 상관관계가 존재하는 것으로 확인되었으나, 
            모델 Deployment 이후 실제 사용할 Assay Dataset에는 SEDEX 및 FOI 계산에 필요한 C5 Insoluble data가 없어 최종 모델 학습에는 포함하지 않음    
            6. 모델링 과정에서 모델 성능과 Domain Knowledge 활용하여 feature 추가 / 삭제 / Engineering 과정 반복 수행
            ---
            ''')
        with col2:
            pass

        with col3:
            st.markdown('#### 🤖 모델 선정 History ')
            st.markdown('''
            ---
            ###### :orange[Random Forest, XGBoost (기각)]
            1. 일반적으로 Regression 성능이 좋은 Decision Tree Based Ensemble 모델부터 시도함 → Random Forest, XGBoost Model
            2. Feature Selection/Englineering 및 Hyper-Parameter Tuning 이후 XGBoost 기준 최종 예측 성능 R2 Score = 0.848, MAPE = 10.6 % 
            ''')
            with st.expander("XGBoost Model 성능", icon= "🖱️"):
                st.image(image['pre_act_image_xgb'], caption="XY Plot of Prediction-Actual SHFT - XGBoost Model", use_container_width=True, width=1)
                st.image(image['pre_act_trend_image_xgb'], caption="Prediction(blue)-Actual(orange) - XGBoost Model", use_container_width=True, width=1)
            st.markdown('''
            3. 예측 정확도는 높으나 실제 활용에 있어 아래와 같은 한계점이 존재하여 최종적으로 기각함
                - SHFT 예측 모델의 최종 목적은 Feed property와 SHFT 간 상관관계를 파악하고, Feed Property 변화에 따른 SHFT 영향을 확인하는 것임
                - SHFT 예측값 계산 시 운전변수에 대한 영향을 크게 받고, Feed 변화에 대한 영향은 상대적으로 미미함
                - 다수의 Feed Property Feature에 대해 SHFT의 관계가 기존 Domain Knowledge와 불일치함
                - Feature와 SHFT 사이의 관계가 직관적이지 않음
                - 학습 Data 영역을 벗어나는 Input에 대해서는 사실상 예측이 불가능하였음
            ---
            ###### :blue[Linear Regression (최종 채택)]
            1. 위에서 언급한 Tree 기반 모델의 한계를 극복할 수 있는 모델로 Linear Regression 모델을 선정함
            2. 일반적으로 타 알고리즘에 비해 예측 정확도가 낮고, 데이터의 복잡도가 높아질수록 성능이 떨어진다는 단점이 있으나
            아래와 같은 장점도 존재함
                - Feature와 y(SHFT) 사이의 관계가 비교적 직관적임
                - 학습에 사용되지 않은 Data 영역에 대해서도 예측 가능함           
                - Feed Property Feature와 y(SHFT)의 관계가 기존 Domain Knowledge와 상당 부분 일치함
            ---
            ###### :orange[시계열 분석 (기각)]
            1. 공정 데이터 특성 상 시간에 따른 영향이 있을 것으로 생각하여 시계열 관련 변수를 추가하여 시도함
            2. 시계열 변수 추가 시 예측 정확도 자체는 높아지나, 최종 활용 목적에 맞지 않아 최종적으로 기각함
                - Deployment 이후 사용할 Input 데이터(특정 Crude Assay 또는 VR Feed의 Property)에는 시계열이 포함되지 않으므로, 
                시계열 데이터로 모델을 학습하는 것은 부적절할 것으로 판단함
            ---
            ###### :orange[Linear Regression + XGBoost Hybrid Model (기각)]
            1. XGBoost Model의 경우 정확도는 높으나 학습 데이터 구간을 벗어나는 Extrapolation은 불가하고, 
            Linear Regression은 데이터 구간에 상관 없이 예측이 가능하나 정확도가 다소 떨어지는 단점이 있음
            2. 두 모델을 혼합 사용하여 각 모델의 단점을 보완하고자 아래와 같이 시도함
                - Stacking Regressor 모델로 Linear Regression과 XGBoost 모델을 혼합함 
                → XGBoost의 영향을 크게 받아 예측 결과가 XGBoost 단독 모델과 거의 일치하게 나옴 (Train Data 범위를 벗어나는 Input에 대한 Extrapolation 불가)
                - Input Data의 각 feature 값이 train data 범위 이내인 경우 xgb model로, 그 외에는 linear model로 예측 
                → feature 값 증가에 따른 예측값이 불연속적인 지점이 발생함)
            ---
            ''')
    with tab4:
        with st.container():
            # with st.expander("SEDEX", icon= "🖱️"):
            st.markdown('#### :blue[*SEDEX*]')
            st.markdown('''
            1. Sediment 생성의 경향성을 확인하는 지표로, 절대값보다는 증감의 변화를 확인하는 목적으로 주로 사용함 
            2. SEDEX 계산식: :blue[(C7A * CCR * Nitrogen * sqrt(Ni + V) * 11.37) / (10 * (C5A - C7A) * Sulfur)]
                - 단위는 모두 %로 통일해야 함 (ppm → % 변환 필요)
            3. 해석 
                - 분자에 있는 인자가 많을수록 Sediment가 증가하는 방향
                    - :blue[C7 Insoluble]: C7 Heptane으로 녹이고 남은 침전물 = Asphaltene
                    - :blue[Nitrogen]: 일반적으로 Asphaltene에 많이 결합되어 있기 때문에, Nitrogen이 많을수록 Asphaltene 함량이 많다는 것을 간접적으로 확인할 수 있음
                - 분모에 있는 인자가 많을 수록 Sediment가 감소하는 방향
                    - :blue[C5-C7 insolube 차이]: Resin 함량을 의미함 (C7에서는 Resin이 용해됨). 즉 C5A-C7A 차이가 작다는 것은 Asphaltene 침전물을 녹여주는 역할(Heptizing)을 하는 Resin(Polar Aromatic)이 적다는 것을 의미함 
                        - 일반적인 hydrocarbon은 Non-polar이지만 Metal, S, N이 붙으면서 Polar로 전환됨. Polar-Asphaltene을 녹여주기 위해서는 Resin과 같은 Polar Aromatic이 필요함
                    - :blue[Sulfur]: Nitrogen과 달리 VR 분자구조의 외부에 많이 생성되어 있어 Treating이 더 쉬움 → 상대적으로 Aromatic/Resin 함량이 많다는 간접 지표로 사용됨 
            4. SEDEX - SHFT 관계
                - 대체로 양의 상관관계를 보이나, 비례 관계가 뚜렷하지는 않으며 추세가 일치하지 않는 경우도 일부 존재함
            ''')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1: 
                st.image(image['Corr_withSEDEX'], caption="SEDEX-SHFT Correlation", use_container_width=True, width=1)
            with col2:
                st.image(image['SHFT_SEDEX'], caption="SEDEX-SHFT Trend", use_container_width=True, width=1)
            with col3:
                st.image(image['SHFT_SEDEX_scatter'], caption="SEDEX-SHFT Scatterplot", use_container_width=True, width=1)
            st.markdown('---')

        # with st.expander("CII", icon= "🖱️"):
        with st.container():            
            st.markdown('#### :blue[*CII (Colloidal Instability Index)*]')
            st.markdown(''' 
            1. SARA(Saturates, Aromatics, Resins, Asphaltenes) 비율에 따른 유분의 Sediment 형성 정도, 즉 Instability를 나타내는 지표
            2. CII 계산식: :blue[(Saturates + Asphaltenes) / (Aromatics + Resins)]
            3. 해석
                - :blue[Aromatic]은 일종의 solvent처럼 작용하여 Asphaltene이 고르게 퍼져 있도록 하여 Sediment 형성을 방지함
                - :blue[Resin]은 Asphaltene 입자를 둘러싸서 일종의 barrier를 형성함으로써 Asphaltene 입자들이 서로 응축되지 않고 퍼져 있도록 함
                - :blue[Saturate]는 Sediment 형성에 직접적인 영향을 주지는 않으나, Saturate 비율이 높음은 곧 Sediment 형성을 완화하는 Aromatic과 Resin의 비율이 낮음을 의미 → 높은 Saturate 비율이 Sediment 형성을 유발하는 간접 인자가 될 수 있음
                - :blue[Asphaltene]은 Sediment를 형성하는 주요 성분으로, Asphaltene 비율이 높을수록 Sediment 형성에 악영향을 줌
            4. 분석 방법 자체가 ASTM standards에 기반하지 않아 재현성/반복성에 대한 이슈가 존재함
            5. CII - SHFT 관계
                - CII와 SHFT의 추세가 일치하지 않는 경우가 다수 존재하며 비례 관계가 비교적 불명확함
            ''')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1: 
                st.image(image['Corr_withSARA'], caption="SARA-SHFT Correlation", use_container_width=True, width=1)
            with col2:
                st.image(image['SHFT_SARA'], caption="SARA-SHFT Trend", use_container_width=True, width=1)
            with col3:
                st.image(image['SHFT_SARA_scatter'], caption="SARA-SHFT Scatterplot", use_container_width=True, width=1)
            st.markdown('---')
    
        # with st.expander("FOI", icon= "🖱️"):
        with st.container():
            st.markdown('#### :blue[*FOI (Feed Operability Index)*]')
            st.markdown('''
            1. 개별 Feedstock의 Property로부터 계산된 일종의 '처리 난이도' 
            2. FOI 계산식: :blue[exp( 3.9544 - 0.3548(Sulfur) - 0.9482(C5A/C7A) + 5.3940(4*Ni+V) + 1.9480(N) )]
                - 단위는 모두 %로 통일해야 함 (ppm → % 변환 필요)
            3. 값이 1 - 4 사이이면 Easy Feed / 5 - 9  사이이면 Medium Feed / 10 이상이면 Difficult Feed로 분류하며, 값이 커질수록 SHFT도 높아지는 양상을 보임            
            4. FOI - SHFT 관계
                - 대체로 양의 상관관계를 보이나, 비례 관계가 뚜렷하지는 않으며 추세가 일치하지 않는 경우도 일부 존재함
            ''')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1: 
                st.image(image['Corr_withFOI'], caption="FOI-SHFT Correlation", use_container_width=True, width=1)
            with col2:
                st.image(image['SHFT_FOI'], caption="FOI-SHFT Trend", use_container_width=True, width=1)
            with col3:
                st.image(image['SHFT_FOI_scatter'], caption="FOI-SHFT Scatterplot", use_container_width=True, width=1)
            st.markdown('---')
    
        # with st.expander("기타 분석 방법", icon= "🖱️"):
        with st.container():
            st.markdown('#### :blue[기타 분석 방법]')
            st.markdown('''
            그 밖에 아래와 같은 분석 기법들이 존재하나 실공정 적용에는 어려움이 있음
            - Compatibility Heithaus Titration → ASTM D6703 Heithaus Titration method 분석 필요
            - Solubility Fractionation Test (5-Solvent Asphaltene Fractionation) → 개발 진행 중
            - Spent catalyst advanced characterization → Spent catalyst에 대한 성분 분석 결과 필요
            ''')
            st.markdown('---')
    with tab5:
        st.markdown('#### 📘 용어 설명')
        st.markdown(''' 
        - :blue[**SHFT (Shell Hot Filtration Test)**]
            - :blue[Shell]: Heavy Residues에 포함된 Sediment 측정 목적으로 :blue[Shell]에서 고안해 낸 분석법. (ASTM D4870, IP-375, ISO-10307에 상응)	
            - :blue[Hot]: 분석을 실시하는 동안 :blue[100℃]를 유지 (Viscosity를 감소시켜 효율적으로 분석을 진행하기 위함)	
            - :blue[Filtration]: Filter를 통해 :blue[시료를 여과]시킨 후, Sediment의 양을 측정함	
            - SHFT가 높을수록 Downstream Equipment의 :blue[Fouling이 심화]됨 
            - VRHCR 공정에서는 매일 612C-109/209 (Train 1/2 MPHT Separator) Bottom Sample을 채취하여 SHFT를 분석함 (C-109: 21시 / C-209: 14시)
        - :blue[**Sediment와 Asphaltene**]
            - :blue[Sediment]: Filter를 통과하지 않고 걸러진 모든 물질로, Inorganics, Coke, :blue[Insoluable Asphaltene]이 포함됨
            - :blue[Asphaltene]
                - 과량의 n-pentane이나 n-heptane을 첨가하였을 때 침적되며, Toluene에는 용해되는 High Molecular Weight Hydrocarbon (700~1000g/mol)	
                - SARA Matrix 내에서 콜로이드 형태로 존재함
                - Oil의 Stability가 감소함에 따라, 침적되거나 Flocculation에 의해 Sludge를 형성함
        ---
        ''')
        st.markdown('#### 🧪 SHFT 분석 방법 (개요)')
        st.markdown('''
        1) :blue[**시료 및 Filter 준비**]
            - 전체 시료를 30초 동안 흔들어 :blue[균일질의 시료]가 되도록 한다.
            - 매 시험마다 Filter를 특정 온도/건조제 조건에서 :blue[건조 후 무게 측정]한다. 
        2) :blue[**분석 (Filtration)**]
            - Filter 및 Chamber 설치 후 진공압력을 Full open하고, 진공 Pump 스위치를 Off한 후 10분 동안 steam을 통과시켜 100 ± 1℃로 유지한다. 	
            - 준비한 시료 약 11g를 30ml 용량의 Beaker에 가하고 0.01g까지 무게를 측정한다. (:blue[filtration 전 시료 무게 측정])
            - Beaker를 Hot plate 위에 올려 놓고 98 - 102℃가 유지되도록 온도계로 저으며 가열한다.	
            - 시료가 98 - 102℃ 사이로 가열되면 :blue[시료를 filter 중앙에 부어 Filtering을 실시]한다. 
            - Beaker에 남아있는 잔량시료와 온도계에 남아있는 시료는 씻지 않고 무게를 측정하여 최초 시료무게에서 빼준다. 
            - 여과가 25분 이내에 종료되지 않으면 시험을 중지하고 5 ± 0.3g의 시료를 사용하여 다시 시험한다.	
            - 2차 여과도 25분 초과 시 시험을 중지하고 Lims 비고란에 “5g여과시간 25분 초과” 로 결과를 입력한다.
        3) :blue[**SHFT 계산**]
            - 다음 식에 따라 0.01%(M/M)까지 총 침전물의 질량 %를 계산한다.
            - S = ((M5-M4)-(M3-M2)) / 10M1 :blue[→ 시료 Filtration 이후 Filter에 남아있는 시료의 비율 = SHFT]
                - S = 총 침전물, %(M/M) 
                - M1 = 시료의 질량, g
                - M2 = 여과 전의 아래쪽 Filter의 질량, ㎎
                - M3 = 여과 후의 아래쪽 Filter의 질량, ㎎
                - M4 = 여과 전의 윗쪽 Filter의 질량, ㎎
                - M5 = 여과 후의 윗쪽 Filter의 질량, ㎎	
        4) 상세 내용 업무절차서 참고
            - 정유품질보증팀 [PR-LAB1-01-05-0255] "Total sediment in residual fuel oils (by SHFT) 분석"   
        ---
        ''')
