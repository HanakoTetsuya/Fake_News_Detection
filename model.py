import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# モデルのロード
model = pickle.load(open('text_classification_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlitアプリのタイトル
st.title('Fake News Detection App')

# ユーザー入力
user_input = st.text_area("Input News Text", "")

# ニュースの判定を実行
if st.button('Judge'):
    # 入力されたテキストのベクトル化
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    # モデルでの予測
    prediction = model.predict(user_input_tfidf)
    result = 'Real News' if prediction[0] == 0 else 'Fake News'

    # 結果の表示
    st.write(f'Result: **{result}**')
