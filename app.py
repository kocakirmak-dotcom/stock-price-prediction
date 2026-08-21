import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests

st.title("📈 Çoklu Hisse Analiz ve Foundry Local Asistanı")

# Yan menü - Hisse Seçimi
st.sidebar.header("Ayarlar")
hisse_listesi = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
hisse_kodu = st.sidebar.selectbox("Hisse Seçin", hisse_listesi)

gun_sayisi = st.sidebar.slider("Geçmiş Gün Sayısı", 30, 365, 250)

# Yerel LLM / Foundry Local Ayarları
st.sidebar.subheader("Foundry Local")
local_endpoint = st.sidebar.text_input("Uç Nokta", "http://localhost:11434/api/generate")
model_adi = st.sidebar.text_input("Model Adı", "phi3")

# Veriyi Yahoo Finance'den çekme
data = yf.download(hisse_kodu, period=f"{gun_sayisi}d")
df = data.reset_index()

# Gün indeksini sütun olarak ekleme
df['Day_Index'] = range(len(df))

# Lineer Regresyon Modeli Kurulumu
model = LinearRegression()
X = df[['Day_Index']]
y = df['Close']

model.fit(X, y)
df['Prediction'] = model.predict(X)

# Fiyat Grafiği Çizimi
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Close'], label='Gerçek Fiyat', color='blue')
ax.plot(df['Prediction'], label='Regresyon Çizgisi', color='red')
ax.legend()
st.pyplot(fig)

# İstatistiksel Özet Tablosu
st.subheader("İstatistiksel Özet")
st.write(df['Close'].describe())

# 5 Gün Sonraki Fiyat Tahmini (Garanti Tip Dönüşümü)
son_index = df['Day_Index'].iloc[-1]
tahmin_5_gun = model.predict([[son_index + 5]])

# Verinin tipinden bağımsız olarak ilk elemanı güvenle çekiyoruz
son_fiyat = float(pd.Series(df['Close']).iloc[-1])
gelecek_fiyat = float(tahmin_5_gun[0])
fark_yuzde = ((gelecek_fiyat - son_fiyat) / son_fiyat) * 100

# Ekranda Gösterme
col1, col2 = st.columns(2)
with col1:
    st.metric("Son Kapanış Fiyatı", f"${son_fiyat:.2f}")
with col2:
    st.metric("5 Gün Sonraki Tahmin", f"${gelecek_fiyat:.2f}", delta=f"%{fark_yuzde:.2f}")

# Foundry Local / LLM Akıllı Asistan Kısmı
st.subheader("🤖 Foundry Local Risk Analisti")

if st.button("Risk Analizi Üret"):
    prompt = f"{hisse_kodu} hissesi için mevcut fiyat ${son_fiyat:.2f} ve 5 günlük tahmin ${gelecek_fiyat:.2f}. Yatırımcıya kısa bir risk değerlendirmesi yap."
    
    try:
        payload = {"model": model_adi, "prompt": prompt, "stream": False}
        cevap = requests.post(local_endpoint, json=payload, timeout=4)
        
        if cevap.status_code == 200:
            st.success(cevap.json().get("response"))
        else:
            st.info(f"Yerel servis kapalı. Otomatik Not: {hisse_kodu} için değişim beklentisi %{fark_yuzde:.2f} seviyesindedir.")
    except:
        st.info(f"Otomatik Risk Notu: Model trendine göre portföyde çeşitlendirme yapılması önerilir.")
