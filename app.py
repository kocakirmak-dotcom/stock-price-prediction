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

# Veriyi Yahoo Finance'den çekme ve tek boyutlu hale getirme
data = yf.download(hisse_kodu, period=f"{gun_sayisi}d")
df = data.reset_index()

# Sütun isimlerini düzleştirme
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

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

# 5 Gün Sonraki Fiyat Tahmini
son_index = df['Day_Index'].iloc[-1]
tahmin_5_gun = model.predict([[son_index + 5]])

son_fiyat = float(df['Close'].iloc[-1])
gelecek_fiyat = float(tahmin_5_gun[0])
fark_yuzde = ((gelecek_fiyat - son_fiyat) / son_fiyat) * 100

# Ekranda Gösterme
col1, col2 = st.columns(2)
with col1:
    st.metric("Son Kapanış Fiyatı", f"${son_fiyat:.2f}")
with col2:
    st.metric("5 Gün Sonraki Tahmin", f"${gelecek_fiyat:.2f}", delta=f"%{fark_yuzde:.2f}")

# Foundry Local / LLM Akıllı Asistan Kısmı (Dinamik Risk Analisti)
st.subheader("🤖 Foundry Local Risk Analisti")

if st.button("Risk Analizi Üret"):
    prompt = f"{hisse_kodu} hissesi için mevcut fiyat ${son_fiyat:.2f} ve 5 günlük tahmin ${gelecek_fiyat:.2f}. Yatırımcıya kısa bir risk değerlendirmesi yap."
    
    try:
        payload = {"model": model_adi, "prompt": prompt, "stream": False}
        cevap = requests.post(local_endpoint, json=payload, timeout=4)
        
        if cevap.status_code == 200:
            st.success(cevap.json().get("response"))
        else:
            # Dinamik Fallback (Hissenin durumuna göre değişen akıllı notlar)
            if fark_yuzde > 5:
                st.info(f"📊 **Büyüme Sinyali ({hisse_kodu}):** Model %{fark_yuzde:.2f} oranında güçlü bir yukarı yönlü ivme öngörüyor. Pozisyonlar korunabilir ancak kar al seviyelerine dikkat edilmelidir.")
            elif 0 <= fark_yuzde <= 5:
                st.info(f"⚖️ **Konsolidasyon Notu ({hisse_kodu}):** Fiyat yatay seyir izliyor (%{fark_yuzde:.2f}). Piyasa belirsizliğine karşı portföyde dengeli dağılım önerilir.")
            else:
                st.info(f"⚠️ **Düşüş Riski Uyarısı ({hisse_kodu}):** Model %{abs(fark_yuzde):.2f} oranında geri çekilme öngörüyor. Zarar kes (stop-loss) seviyelerinin gözden geçirilmesi tavsiye edilir.")
    except:
        if fark_yuzde > 0:
            st.info(f"🚀 **Trend Notu:** {hisse_kodu} için pozitif trend baskın görünmektedir.")
        else:
            st.info(f"🛡️ **Defansif Strateji:** {hisse_kodu} için temkinli duruş ve nakit yönetimi ön planda tutulmalıdır.")
