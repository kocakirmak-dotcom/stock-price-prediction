import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("AAPL Hisse Tahmin Uygulaması")

# Yan menü (Sidebar) ekleyelim
st.sidebar.header("Ayarlar")
hisse_kodu = st.sidebar.text_input("Hisse Kodu Giriniz", "AAPL")
gun_sayisi = st.sidebar.slider("Geçmiş Gün Aralığı", 30, 365, 250)

# Veriyi çek
data = yf.download(hisse_kodu, period=f"{gun_sayisi}d")
df = data.reset_index()
df['Day_Index'] = np.arange(len(df))

# Regresyon Modeli
model = LinearRegression()
model.fit(df[['Day_Index']], df['Close'])
df['Prediction'] = model.predict(df[['Day_Index']])

# Grafik çizimi
fig, ax = plt.subplots()
ax.plot(df['Close'], label='Gerçek Fiyat')
ax.plot(df['Prediction'], label='Tahmin (Regresyon)', color='red')
ax.legend()
st.pyplot(fig)

# İstatistiksel özet
st.write("Seçilen döneme ait fiyat istatistikleri:")
st.write(df['Close'].describe())
