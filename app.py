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

st.title("📈 PyTorch LSTM Tabanlı Hisse Tahmini ve Risk Asistanı")

# Yan menü - Hisse Seçimi ve Ayarlar
st.sidebar.header("Proje Ayarları")
hisse_listesi = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
hisse_kodu = st.sidebar.selectbox("Hisse Seçin", hisse_listesi)

gun_sayisi = st.sidebar.slider("Geçmiş Gün Sayısı", 60, 365, 180)
epochs = st.sidebar.slider("Model Eğitim Döngüsü (Epochs)", 10, 100, 30)

# Yerel LLM / Foundry Local Ayarları
st.sidebar.subheader("Foundry Local")
local_endpoint = st.sidebar.text_input("Uç Nokta", "http://localhost:11434/api/generate")
model_adi = st.sidebar.text_input("Model Adı", "phi3")

# 1. Veriyi Çekme ve Temizleme
data = yf.download(hisse_kodu, period=f"{gun_sayisi}d")
df = data.reset_index()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close_prices = df['Close'].values.ravel().astype(float)

# 2. Veriyi Ölçeklendirme (MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

# Zaman Serisi Pencereleme (Sliding Window)
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

# 3. PyTorch LSTM Model Mimarisi
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

# 4. Modeli Eğitme
with st.spinner("PyTorch LSTM modeli eğitiliyor..."):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

# Tahminler
model.eval()
with torch.no_grad():
    train_predict = model(X_tensor)
    # Son 5 günlük pencere ile geleceği tahmin etme
    son_pencere = torch.tensor(scaled_data[-look_back:].reshape(1, look_back, 1), dtype=torch.float32)
    gelecek_tahmin_scaled = model(son_pencere)

predicted_prices = scaler.inverse_transform(train_predict.numpy())
actual_prices = scaler.inverse_transform(y_tensor.numpy())
gelecek_fiyat = float(scaler.inverse_transform(gelecek_tahmin_scaled.numpy())[0][0])

# 5. Değerlendirme Metrikleri (MSE / RMSE)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)

# 6. Görselleştirme
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, label='Gerçek Fiyatlar', color='blue')
ax.plot(predicted_prices, label='PyTorch LSTM Tahminleri', color='red', linestyle='--')
ax.set_title(f"{hisse_kodu} - PyTorch LSTM Zaman Serisi Tahmini")
ax.legend()
st.pyplot(fig)

st.subheader("📊 PyTorch Model Performans Metrikleri")
mcol1, mcol2 = st.columns(2)
with mcol1:
    st.metric("Hata Kareler Ortalaması (MSE)", f"{mse:.2f}")
with mcol2:
    st.metric("Kök Ortalama Kare Hata (RMSE)", f"{rmse:.2f}")

st.subheader("İstatistiksel Özet")
st.write(pd.Series(close_prices).describe())

son_fiyat = float(close_prices[-1])
fark_yuzde = ((gelecek_fiyat - son_fiyat) / son_fiyat) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Son Kapanış Fiyatı", f"${son_fiyat:.2f}")
with col2:
    st.metric("5 Gün Sonraki Tahmin (LSTM)", f"${gelecek_fiyat:.2f}", delta=f"%{fark_yuzde:.2f}")

# Foundry Local / LLM Asistanı
st.subheader("🤖 Foundry Local Risk Analisti")

if st.button("PyTorch Tabanlı Risk Analizi Üret"):
    prompt = f"PyTorch LSTM modeli ile analiz edilen {hisse_kodu} hissesi için mevcut fiyat ${son_fiyat:.2f}, model RMSE hata skoru: {rmse:.2f}. Yatırımcıya derin öğrenme sonuçlarına göre risk tavsiyesi ver."
    
    basarili = False
    try:
        payload = {"model": model_adi, "prompt": prompt, "stream": False}
        cevap = requests.post(local_endpoint, json=payload, timeout=3)
        if cevap.status_code == 200:
            st.success(cevap.json().get("response"))
            basarili = True
    except:
        pass

    if not basarili:
        if fark_yuzde > 0:
            st.info(f"🚀 **LSTM Büyüme Sinyali ({hisse_kodu}):** Derin öğrenme modelimiz yukarı yönlü trend yakaladı. (Tahmini Değişim: %{fark_yuzde:.2f}, RMSE: {rmse:.2f}).")
        else:
            st.info(f"⚠️ **LSTM Risk Uyarısı ({hisse_kodu}):** Model dalgalanma ve aşağı yönlü hareket öngörüyor. (RMSE: {rmse:.2f}). Stop-loss seviyelerine dikkat edilmelidir.")
