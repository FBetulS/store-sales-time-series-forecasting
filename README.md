# store-sales-time-series-forecasting
# Store Sales - Time Series Forecasting

## 📖 Proje Açıklaması
Bu proje, Kaggle'daki "Store Sales - Time Series Forecasting" yarışmasında mağaza satışlarını tahmin etmek amacıyla geliştirilmiştir. Proje, zaman serisi analizi ve XGBoost regresyon modeli kullanarak eğitim ve test verileri üzerinde satış tahmini yapmaktadır.



## 🔗 Hugging Face
Hugging Face üzerindeki [Sales Time Series Forecasting](https://huggingface.co/spaces/btulftma/sales-time-seriesforecasting) uygulamasını ziyaret edebilirsiniz.

## 🔗 Veri Kümesi
Veriler, Kaggle'daki [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview) yarışmasından alınmıştır. Proje, aşağıdaki veri dosyalarını kullanmaktadır:
- `train.csv`: Eğitim verileri
- `test.csv`: Test verileri
- `stores.csv`: Mağaza bilgileri
- `oil.csv`: Petrol fiyatları
- `holidays_events.csv`: Tatil etkinlikleri
- `transactions.csv`: İşlem verileri

## 🛠️ Kullanılan Kütüphaneler
- `pandas`: Veri analizi ve manipülasyonu için.
- `numpy`: Sayısal işlemler için.
- `matplotlib` ve `seaborn`: Verilerin görselleştirilmesi için.
- `statsmodels`: Zaman serisi analizi için.
- `sklearn`: Model değerlendirme ve metrikler için.
- `xgboost`: Makine öğrenimi modeli için.
- `holidays`: Tatil verilerinin işlenmesi için.

## 📊 Model Eğitimi ve Tahmin
Proje, aşağıdaki adımları içermektedir:
1. **Veri Yükleme**: Eğitim ve test verileri, ilgili dosyalardan yüklenir ve birleştirilir.
2. **Öznitelik Mühendisliği**: Tarih verilerinden yıl, ay, gün gibi özellikler elde edilir. Tatil bilgileri işlenir.
3. **Model Eğitimi**: Zaman serisi cross-validation kullanılarak XGBoost regresyon modeli eğitilir.
4. **Tahmin**: Test verisi üzerinde tahmin yapılır ve sonuçlar `submission.csv` dosyasına kaydedilir.

## 📈 Sonuç
Modelin performansı, Root Mean Squared Error (RMSE) metriği ile değerlendirilmiştir. Elde edilen RMSE skoru: **524.39**. Ayrıca, mevcut kodlarım ve çözümümle elde ettiğim skor: **550**.  
**Kullanıcı Adı**: Fatma Betül SIRIM  
**Leaderboard Skoru**: 1.78486  
**Zaman**: 3 dakika  
**Geri Bildirim**: 🙂 "Your First Entry! Welcome to the leaderboard!"
