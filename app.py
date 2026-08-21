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

close_series = pd.Series(df['Close'].values.ravel())
df['Day_Index'] = range(len(df))

# Regresyon Modeli ve Tahmin
model = LinearRegression()
X = df[['Day_Index']]
y = close_series

model.fit(X, y)
df['Prediction'] = model.predict(X)

mse = mean_squared_error(y, df['Prediction'])
rmse = np.sqrt(mse)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(close_series, label='Gerçek Fiyat', color='blue')
ax.plot(df['Prediction'], label='Regresyon Çizgisi', color='red')
ax.legend()
st.pyplot(fig)

st.subheader("📊 Model Performans ve Hata Metrikleri")
mcol1, mcol2 = st.columns(2)
with mcol1:
    st.metric("Hata Kareler Ortalaması (MSE)", f"{mse:.2f}")
with mcol2:
    st.metric("Kök Ortalama Kare Hata (RMSE)", f"{rmse:.2f}")

st.subheader("İstatistiksel Özet")
st.write(close_series.describe())

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

st.subheader("🤖 Foundry Local Risk Analisti")

if st.button("Risk Analizi Üret"):
    prompt = f"{hisse_kodu} hissesi için mevcut fiyat ${son_fiyat:.2f} ve 5 günlük tahmin ${gelecek_fiyat:.2f}. Model hata skoru (RMSE): {rmse:.2f}. Yatırımcıya kısa bir risk değerlendirmesi yap."
    
    basarili_yanit_alindi = False
    try:
        payload = {"model": model_adi, "prompt": prompt, "stream": False}
        cevap = requests.post(local_endpoint, json=payload, timeout=3)
        
        if cevap.status_code == 200:
            st.success(cevap.json().get("response"))
            basarili_yanit_alindi = True
    except:
        pass

    if not basarili_yanit_alindi:
        if fark_yuzde > 10:
            st.info(f"🚀 **Güçlü Büyüme ({hisse_kodu}):** Model %{fark_yuzde:.2f} oranında yüksek bir artış öngörüyor. (Model Hata Payı RMSE: {rmse:.2f}). Momentum değerlendirilebilir ancak kar realizasyonu için seviyeler takip edilmelidir.")
        elif 0 < fark_yuzde <= 10:
            st.info(f"📈 **Ilımlı Yükseliş ({hisse_kodu}):** Hisse için %{fark_yuzde:.2f} civarında sınırlı bir pozitif trend bekleniyor. (RMSE: {rmse:.2f}). Kademeli alım stratejisi izlenebilir.")
        elif -5 <= fark_yuzde <= 0:
            st.info(f"⚖️ **Yatay / Yatay Seyir ({hisse_kodu}):** Beklenen değişim %{fark_yuzde:.2f} seviyesinde. Piyasa konsolidasyon sürecinde olduğu için portföyde bekle-gör politikası uygulanabilir.")
        else:
            st.info(f"⚠️ **Yüksek Risk / Düşüş ({hisse_kodu}):** Model %{abs(fark_yuzde):.2f} oranında geri çekilme öngörüyor. (RMSE: {rmse:.2f}). Sermaye koruması için stop-loss emirleri kritik önem taşımaktadır.")
