import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Sayfa genişliğini tam ekran (wide) yaparak içeriğin sıkışmasını önleyelim
st.set_page_config(page_title="PyTorch LSTM Hisse Analizi", layout="wide")

st.markdown("### 📈 PyTorch LSTM Tabanlı Hisse Tahmini ve Risk Asistanı")

# Yan menü - Ayarlar
st.sidebar.header("Proje Ayarları")
hisse_listesi = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
hisse_kodu = st.sidebar.selectbox("Hisse Seçin", hisse_listesi)

gun_sayisi = st.sidebar.slider("Geçmiş Gün Sayısı", 60, 365, 180)
epochs = st.sidebar.slider("Eğitim Döngüsü (Epochs)", 10, 100, 30)

st.sidebar.subheader("Foundry Local")
local_endpoint = st.sidebar.text_input("Uç Nokta", "http://localhost:11434/api/generate")
model_adi = st.sidebar.text_input("Model Adı", "phi3")

# 1. Veriyi Çekme ve Temizleme
data = yf.download(hisse_kodu, period=f"{gun_sayisi}d")
df = data.reset_index()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close_prices = df['Close'].values.ravel().astype(float)

# 2. Ölçeklendirme ve Pencereleme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

def create_dataset(dataset, look_back=5):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 5
X_np, y_np = create_dataset(scaled_data, look_back)

X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

# 3. LSTM Mimarisi
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Modeli Eğitme
with st.spinner("Model eğitiliyor..."):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    train_predict = model(X_tensor)
    son_pencere = torch.tensor(scaled_data[-look_back:].reshape(1, look_back, 1), dtype=torch.float32)
    gelecek_tahmin_scaled = model(son_pencere)

predicted_prices = scaler.inverse_transform(train_predict.numpy())
actual_prices = scaler.inverse_transform(y_tensor.numpy())
gelecek_fiyat = float(scaler.inverse_transform(gelecek_tahmin_scaled.numpy())[0][0])

mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
son_fiyat = float(close_prices[-1])
fark_yuzde = ((gelecek_fiyat - son_fiyat) / son_fiyat) * 100

# --- KOMPAKT 2 SÜTUNLU YERLEŞİM (TEK KAREYE SIĞMASI İÇİN) ---
col_sol, col_sag = st.columns([1.2, 1])

with col_sol:
    # Grafik
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(actual_prices, label='Gerçek Fiyat', color='blue', linewidth=1.5)
    ax.plot(predicted_prices, label='LSTM Tahmin', color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f"{hisse_kodu} - Zaman Serisi Tahmini", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

with col_sag:
    st.markdown("##### 📊 Model Performans Metrikleri")
    m1, m2 = st.columns(2)
    m1.metric("MSE", f"{mse:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    
    st.markdown("##### 💰 Fiyat Beklentisi")
    f1, f2 = st.columns(2)
    f1.metric("Son Fiyat", f"${son_fiyat:.2f}")
    f2.metric("5 Gün Sonra", f"${gelecek_fiyat:.2f}", delta=f"%{fark_yuzde:.2f}")

# Alt kısım: Risk Asistanı
st.markdown("---")
st.markdown("##### 🤖 Foundry Local Risk Analisti")
if st.button("Risk Analizi Üret"):
    prompt = f"PyTorch LSTM modeli ile analiz edilen {hisse_kodu} hissesi için mevcut fiyat ${son_fiyat:.2f}, RMSE: {rmse:.2f}. Yatırımcıya risk tavsiyesi ver."
    basarili = False
    try:
        payload = {"model": model_adi, "prompt": prompt, "stream": False}
        cevap = requests.post(local_endpoint, json=payload, timeout=2)
        if cevap.status_code == 200:
            st.success(cevap.json().get("response"))
            basarili = True
    except:
        pass

    if not basarili:
        if fark_yuzde > 0:
            st.info(f"🚀 **LSTM Büyüme Sinyali ({hisse_kodu}):** Yukarı yönlü trend öngörülüyor. (Değişim: %{fark_yuzde:.2f}, RMSE: {rmse:.2f}).")
        else:
            st.info(f"⚠️ **LSTM Risk Uyarısı ({hisse_kodu}):** Dalgalanma/aşağı yönlü hareket bekleniyor. (RMSE: {rmse:.2f}).")
else:
    if fark_yuzde > 0:
        st.info(f"🚀 **Özet Sinyal ({hisse_kodu}):** %{fark_yuzde:.2f} artış bekleniyor. (RMSE: {rmse:.2f})")
    else:
        st.info(f"⚠️ **Özet Sinyal ({hisse_kodu}):** %{abs(fark_yuzde):.2f} geri çekilme öngörülüyor. (RMSE: {rmse:.2f})")
