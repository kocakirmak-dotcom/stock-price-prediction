import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import requests

st.title("📈 Akıllı Hisse Analiz ve Model Değerlendirme Asistanı")

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

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Close sütununu %100 tek boyutlu (1D) diziye çevirme (ValueError önlemi)
close_series = pd.Series(df['Close'].values.ravel())

df['Day_Index'] = range(len(df))

# Regresyon Modeli ve Tahmin
model = LinearRegression()
X = df[['Day_Index']]
y = close_series

model.fit(X, y)
df['Prediction'] = model.predict(X)

# --- PROGRAM HEDEFİ UYUMLULUĞU: Model Değerlendirme Metrikleri (MSE / RMSE) ---
mse = mean_squared_error(y, df['Prediction'])
rmse = np.sqrt(mse)

# Fiyat Grafiği Çizimi
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(close_series, label='Gerçek Fiyat', color='blue')
ax.plot(df['Prediction'], label='Regresyon Çizgisi', color='red')
ax.legend()
st.pyplot(fig)

# Metrikleri Ekranda Gösterme (Dokümandaki MSE/RMSE kriteri)
st.subheader("📊 Model Performans ve Hata Metrikleri")
mcol1, mcol2 = st.columns(2)
with mcol1:
    st.metric("Hata Kareler Ortalaması (MSE)", f"{mse:.2f}")
with mcol2:
    st.metric("Kök Ortalama Kare Hata (RMSE)", f"{rmse:.2f}")

# İstatistiksel Özet Tablosu
st.subheader("İstatistiksel Özet")
st.write(close_series.describe())

# 5 Gün Sonraki Fiyat Tahmini
son_index = df['Day_Index'].iloc[-1]
tahmin_5_gun = model.predict([[son_index + 5]])

son_fiyat = float(close_series.iloc[-1])
gelecek_fiyat = float(tahmin_5_gun[0])
fark_yuzde = ((gelecek_fiyat - son_fiyat) / son_fiyat) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Son Kapanış Fiyatı", f"${son_fiyat:.2f}")
with col2:
    st.metric("5 Gün Sonraki Tahmin", f"${gelecek_fiyat:.2f}", delta=f"%{fark_yuzde:.2f}")

# Foundry Local / LLM Akıllı Asistan Kısmı
st.subheader("🤖 Foundry Local Risk Analisti")

if st.button("Risk Analizi Üret"):
    prompt = f"{hisse_kodu} hissesi için mevcut fiyat ${son_fiyat:.2f} ve 5 günlük tahmin ${gelecek_fiyat:.2f}. Model hata skoru (RMSE): {rmse:.2f}. Yatırımcıya kısa bir risk değerlendirmesi yap."
    
    try:
        payload = {"model": model_adi, "prompt": prompt, "stream": False}
        cevap = requests.post(local_endpoint, json=payload, timeout=4)
        
        if cevap.status_code == 200:
            st.success(cevap.json().get("response"))
        else:
            if fark_yuzde > 5:
                st.info(f"📊 **Büyüme Sinyali ({hisse_kodu}):** Model %{fark_yuzde:.2f} artış öngörüyor (RMSE: {rmse:.2f}).")
            else:
                st.info(f"⚠️ **Temkinli Duruş ({hisse_kodu}):** Model trendinde yatay/aşağı yönlü riskler bulunuyor.")
    except:
        st.info(f"Otomatik Risk Notu: Model hata oranı (RMSE: {rmse:.2f}) baz alınarak portföy çeşitlendirmesi önerilir.")
