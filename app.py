import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("AAPL Hisse Tahmin Uygulaması")

# Veri çekme
data = yf.download("AAPL", start="2025-08-19", end="2026-08-19")
df = data.reset_index()
df['Day_Index'] = np.arange(len(df))

# Regresyon Modeli
model = LinearRegression()
model.fit(df[['Day_Index']], df['Close'])
df['Prediction'] = model.predict(df[['Day_Index']])

# Grafik çizimi
fig, ax = plt.subplots()
ax.plot(df['Close'], label='Gerçek Fiyat')
ax.plot(df['Prediction'], label='Tahmin', color='red')
st.pyplot(fig)

# Tahmin
st.write(f"Model Hata Payı (MAE): ${model.score(df[['Day_Index']], df['Close']):.2f}")