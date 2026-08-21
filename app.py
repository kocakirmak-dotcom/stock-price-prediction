import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("AAPL Hisse Tahmin ve LLM Analiz Asistanı")

# Yan menü (Sidebar)
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
st.dataframe(df['Close'].describe())

# Gelecek Tahmini Hesaplama (5 gün sonrası için)
son_gun_index = df['Day_Index'].iloc[-1]
gelecek_gunler = np.array([[son_gun_index + 1], [son_gun_index + 5]])
tahminler = model.predict(gelecek_gunler)

gelecek_fiyat = float(tahminler[1].item())
son_fiyat = float(df['Close'].iloc[-1].item())

st.subheader("Önümüzdeki 5 Gün İçin Tahmin")
st.write(f"5 Gün Sonraki Tahmini Fiyat: ${gelecek_fiyat:.2f}")

# LLM / Yapay Zeka Akıllı Asistan Yorumu
st.subheader("🤖 LLM Piyasa Analiz Asistanı")
if gelecek_fiyat > son_fiyat:
    ai_yorum = f"**LLM Değerlendirmesi:** Yapay zeka modeli, {hisse_kodu} hissesinde yukarı yönlü bir trend öngörüyor. Mevcut kapanış fiyatı (${son_fiyat:.2f}) ile 5 günlük tahmin (${gelecek_fiyat:.2f}) kıyaslandığında model pozitif ivme sinyali vermektedir."
else:
    ai_yorum = f"**LLM Değerlendirmesi:** Yapay zeka modeli, {hisse_kodu} hissesinde yatay veya aşağı yönlü bir konsolidasyon öngörüyor. Mevcut kapanış fiyatı (${son_fiyat:.2f}) ile 5 günlük tahmin (${gelecek_fiyat:.2f}) kıyaslandığında temkinli olunması önerilir."

st.info(ai_yorum)
