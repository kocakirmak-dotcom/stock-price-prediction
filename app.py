import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests

st.title("📈 Çoklu Hisse Analiz ve Foundry Local LLM Asistanı")

# Yan menü (Sidebar) - Çoklu Hisse Seçimi
st.sidebar.header("Proje Ayarları")
hisse_listesi = ["AAPL (Apple)", "MSFT (Microsoft)", "GOOGL (Google)", "AMZN (Amazon)", "TSLA (Tesla)"]
secilen_secenek = st.sidebar.selectbox("Analiz Edilecek Hisseyi Seçin", hisse_listesi)
hisse_kodu = secilen_secenek.split(" ")[0]

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
ax.plot(df['Close'], label=f'{hisse_kodu} Gerçek Fiyat', color='blue')
ax.plot(df['Prediction'], label='Lineer Regresyon Tahmini', color='red', linestyle='--')
ax.legend()
st.pyplot(fig)

# İstatistiksel özet
st.subheader(f"📊 {hisse_kodu} - İstatistiksel Fiyat Özeti")
st.dataframe(df['Close'].describe())

# Gelecek Tahmini Hesaplama (5 gün sonrası için)
son_gun_index = df['Day_Index'].iloc[-1]
gelecek_gunler = np.array([[son_gun_index + 1], [son_gun_index + 5]])
tahminler = model.predict(gelecek_gunler)

# Güvenli tip dönüşümü (Hata almamak için .iloc[0] kullanıyoruz)
gelecek_fiyat = float(tahminler[1])
son_fiyat = float(df['Close'].iloc[-1].iloc[0]) if isinstance(df['Close'].iloc[-1], pd.Series) else float(df['Close'].iloc[-1])
degisim_yuzde = ((gelecek_fiyat - son_fiyat) / son_fiyat) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Mevcut Kapanış Fiyatı", value=f"${son_fiyat:.2f}")
with col2:
    st.metric(label="5 Gün Sonraki Tahmini Fiyat", value=f"${gelecek_fiyat:.2f}", delta=f"{degisim_yuzde:.2f}%")

# Foundry Local / Akıllı Asistan (Strateji ve Risk Analizi)
st.subheader("🤖 Foundry Local Strateji ve Risk Analisti")

if st.button("Yerel LLM ile Stratejik Risk Raporu Oluştur"):
    with st.spinner("Yerel model risk analizi hazırlıyor..."):
        try:
            prompt_text = (
                f"Sen kıdemli bir risk analistisin. {hisse_kodu} hissesi için beklenen değişim %{degisim_yuzde:.2f}. "
                f"Bu veriye dayanarak yatırımcıya agresif veya defansif bir strateji öner, "
                f"teknik riskleri 2 madde halinde özetle ve kesinlikle fiyat sayılarını tekrarlama."
            )
            payload = {"model": model_adi, "prompt": prompt_text, "stream": False}
            response = requests.post(local_endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                st.success(result.get("response", "Stratejik analiz tamamlandı."))
            else:
                st.warning("Yerel servis kapalı. Akıllı Risk Modu Devrede:")
                if degisim_yuzde > 0:
                    st.info(f"**Strateji Notu:** {hisse_kodu} için büyüme odaklı (growth) pozisyonlar korunabilir, ancak volatiliteye karşı stop-loss seviyeleri ihmal edilmemelidir.")
                else:
                    st.info(f"**Strateji Notu:** {hisse_kodu} tarafında aşağı yönlü baskı gözlendiği için nakit oranını artırmak veya defansif sektörlere yönelmek mantıklı olabilir.")
        except Exception:
            st.info(f"**Strateji Notu:** Seçilen dönemdeki trend yönüne göre portföyde çeşitlendirme yapılması ve risk yönetimi kurallarına uyulması tavsiye edilir.")
