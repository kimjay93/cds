import streamlit as st
import pandas as pd
import numpy as np
import time
import util
import datetime as dt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

def load_model():
    return joblib.load("lr.pkl")

def load_scaler():
    return joblib.load('scaler.pkl')

def load_pca():
    return joblib.load('pca.pkl')

def load_images():
    return {
        "pre_act_image" : Image.open('pre_act.png'),
        "pre_act_trend_image" : Image.open('pre_act_trend.png'),
        "shap_image" : Image.open('SHAP_fd.png')
    }
    
def analyzeDF():
    msg = st.toast('Reading...', icon="📖")
    time.sleep(1)
    msg.toast('Analyzing...', icon="🧐")
    time.sleep(2)
    msg.toast('Ready!', icon = "🔔")
    
def convert_df(df_input):
    # VR Feed 관련 열: [118:152]
    df_input_vr_feed = df_input.iloc[:, 118:152]
    
    # 의미없는 맨 첫 열 제거하고 두번째 열을 column 이름으로 사용
    df_input_vr_feed.columns = df_input_vr_feed.iloc[0]
    df_input_vr_feed = df_input_vr_feed[1:].reset_index(drop=True)
    
    # 제거할 열: 0, 2, 5, 7,11, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33
    col_to_drop = df_input_vr_feed.columns[[0, 2, 5, 7,11, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]]
    df_input_vr_feed.drop(col_to_drop, axis=1, inplace=True)
    df_input_vr_feed.drop('SEDEX', axis=1, inplace=True)
    # df dtype 변경
    for col in df_input_vr_feed.columns:
        df_input_vr_feed[col] = df_input_vr_feed[col].astype('float64')
    
    # 결측치는 평균으로 채우기
    for col in df_input_vr_feed.columns:
        df_input_vr_feed[col].fillna(df_input_vr_feed[col].mean(), inplace=True)
    
    # 위 df에 2024년 평균 운전 데이터를 merge
    # 먼저 2024년 평균 운전 데이터를 df로 구성
    df = pd.read_csv('df_38thtrial.csv')
    df['Datetime'] = df['Datetime'].astype('datetime64[ns]')
    df.set_index('Datetime', drop=True, inplace=True)
    df = df.loc[df.index.year == 2024]
    opavg2024_list = []
    for col in df.columns[:15]:
        opavg2024_list.append(df[col].mean())
    opavg2024_df = pd.merge(pd.DataFrame(df.columns[:15]), pd.DataFrame(opavg2024_list), left_index=True, right_index=True)
    opavg2024_df.columns = ['variable', 'avg']
    opavg2024_df_t = opavg2024_df.transpose()
    opavg2024_df_t.columns = opavg2024_df_t.iloc[0]
    opavg2024_df_t = opavg2024_df_t[1:].reset_index(drop=True)
    opavg2024_df_t_repeated = pd.DataFrame([opavg2024_df_t.iloc[0]] * len(df_input_vr_feed), columns=opavg2024_df_t.columns)
    opavg2024_df_t_repeated.reset_index(drop=True, inplace=True)
    
    df = pd.concat([opavg2024_df_t_repeated, df_input_vr_feed], axis=1)
    return df
    # 이후 scaler.pkl을 이용한 표준화, Ni, V에 대한 pca 및 ni, v 제거, vis100을 boxcox로 변환해야 SHFT 예측 가능함
    return df
    # 이후 scaler.pkl을 이용한 standardization, pca를 이용한 Ni, V 관련 pca 추가 및 ni, v 제거까지 해야 SHFT 예측 가능함

def convert_df_info(df_input):
    # crude information 담고 있는 df 따로 추출
    df_info = df_input.iloc[:,4:6]
    df_info.columns = df_info.iloc[0]
    df_info = df_info[1:].reset_index(drop=True)
    return df_info
