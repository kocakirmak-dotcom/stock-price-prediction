import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests

st.title("📈 Akıllı Hisse Analiz ve Foundry Local LLM Asistanı")

# Yan menü (Sidebar)
st.sidebar.header("Proje Ayarları")
hisse_kodu = st.sidebar.text_input("Hisse Kodu Giriniz", "AAPL")
gun_sayisi = st.sidebar.slider("Geçmiş Gün Aralığı", 30, 365, 250)

# Yerel LLM / Foundry Local Bağlantı Ayarları
st.sidebar.subheader("Microsoft Foundry Local")
local_endpoint = st.sidebar.text_input("Yerel LLM Uç Noktası", "http://localhost:11434/api/generate")
model_adi = st.sidebar.text_input("Yerel Model Adı", "phi3")

# Veriyi çek
data = yf.download(hisse_kodu, period=f"{gun_sayisi}d")
df = data.reset_index()
df['Day_Index'] = np.arange(len(df))

# Regresyon Modeli
model = LinearRegression()
model.fit(df[['Day_Index']], df['Close'])
df['Prediction'] = model.predict(df[['Day_Index']])

# Grafik çizimi
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Close'], label='Gerçek Fiyat', color='blue')
ax.plot(df['Prediction'], label='Lineer Regresyon Tahmini', color='red', linestyle='--')
ax.legend()
st.pyplot(fig)

# İstatistiksel özet
st.subheader("📊 İstatistiksel Fiyat Özeti")
st.dataframe(df['Close'].describe())

# Gelecek Tahmini Hesaplama (5 gün sonrası için)
son_gun_index = df['Day_Index'].iloc[-1]
gelecek_gunler = np.array([[son_gun_index + 1], [son_gun_index + 5]])
tahminler = model.predict(gelecek_gunler)

gelecek_fiyat = float(tahminler[1].item())
son_fiyat = float(df['Close'].iloc[-1].item())

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Mevcut Kapanış Fiyatı", value=f"${son_fiyat:.2f}")
with col2:
    st.metric(label="5 Gün Sonraki Tahmini Fiyat", value=f"${gelecek_fiyat:.2f}", delta=f"{((gelecek_fiyat - son_fiyat)/son_fiyat)*100:.2f}%")

# Foundry Local / Yerel LLM Entegrasyon Bloğu
st.subheader("🤖 Foundry Local Akıllı Finansal Asistan")

prompt_text = (
    f"Sen profesyonel bir veri analistisin. "
    f"{hisse_kodu} hissesi için yapılan regresyona göre mevcut fiyat ${son_fiyat:.2f}, "
    f"5 gün sonraki tahmin ${gelecek_fiyat:.2f} bekleniyor. Kısa bir piyasa analizi sun."
)

if st.button("Yerel LLM (Foundry Local) ile Analiz Üret"):
    with st.spinner("Yerel model üzerinden yanıt alınıyor..."):
        try:
            payload = {"model": model_adi, "prompt": prompt_text, "stream": False}
            response = requests.post(local_endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                st.success(result.get("response", "Model yanıtı alındı."))
            else:
                st.warning("Yerel Foundry Local servisi kapalı. Güvenli mod analizine geçiliyor:")
                if gelecek_fiyat > son_fiyat:
                    st.info(f"**Yerel Analiz:** {hisse_kodu} için model yukarı yönlü eğilim gösteriyor.")
                else:
                    st.info(f"**Yerel Analiz:** {hisse_kodu} için yatay/aşağı yönlü konsolidasyon öngörülüyor.")
        except Exception:
            st.info(f"**Yerel Model Değerlendirmesi:** {hisse_kodu} hissesinde mevcut fiyat ${son_fiyat:.2f} ve 5 günlük tahmin ${gelecek_fiyat:.2f} olarak hesaplanmıştır.")
