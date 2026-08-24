# 📈 PyTorch LSTM Tabanlı Hisse Senedi Fiyat Tahmini ve Risk Asistanı

Bu proje, finansal zaman serisi verilerini modern derin öğrenme araçlarıyla analiz eden ve kullanıcı dostu bir arayüz sunan web tabanlı bir veri bilimi uygulamasıdır.

---

### 🎥 3 Dakikalık Proje Sunum Videosu
Projenin mimarisini, kod akışını ve çalışma mantığını anlattığım 3 dakikalık sunum videosuna aşağıdaki bağlantıdan ulaşabilirsiniz:
https://youtu.be/Xeq07SU3lCM

---

### 🚀 Kullanılan Teknolojiler ve Mimari
* **Arayüz:** Streamlit
* **Veri Kaynağı ve İşleme:** yfinance, Pandas, NumPy
* **Derin Öğrenme / Model:** PyTorch (LSTM - Long Short-Term Memory, Kayan Pencere / Sliding Window yaklaşımı)
* **Model Değerlendirme:** scikit-learn (MinMaxScaler, MSE - Mean Squared Error, RMSE - Root Mean Squared Error)
* **Yapay Zeka Entegrasyonu:** Microsoft Foundry Local / Yerel LLM tabanlı dinamik risk ve strateji analizi

---

### 📊 Proje Özellikleri
1. **Çoklu Hisse Desteği:** Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN) ve Tesla (TSLA) gibi küresel hisseler arasında geçiş yapabilme imkanı.
2. **Dinamik Eğitim:** Kullanıcının geçmiş gün sayısını ve eğitim döngüsünü (epochs) ayarlayabilmesi.
3. **Performans Metrikleri:** Modelin başarı başarımını gösteren anlık MSE ve RMSE hata skorları.
4. **Akıllı Risk Asistanı:** Modelin tahmin ettiği fiyat ve hata payı üzerinden yatırımcıya özel stratejik uyarılar üretilmesi.
