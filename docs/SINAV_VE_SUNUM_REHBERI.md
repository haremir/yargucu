# Yargucu: Sınav ve Sunum Çalışma Rehberi (Kopya Kağıdı)

> **Önemli Not:** Bu dosya **Harun Emirhan Bostancı** tarafından Yapay Öğrenme dersi vize/final sınavı ve dönem projesi sunumu öncesinde hızlıca tekrar yapmak, formülleri ve kod mantığını pekiştirmek ve hocanın gelebilecek mülakat sorularına eksiksiz cevap verebilmek amacıyla hazırlanmıştır.

---

# A. Projenin Özeti ve Amacı

### 1. Proje Adı ve Anlamı
- **Proje Adı:** `yargucu`
- **Kökeni:** Eski Türkçe *yargu* (mahkeme kararı, hüküm) kökünden ve *-cu* yapım ekinden türetilmiştir. Orhun Yazıtları (Kül Tigin, Bilge Kağan) ve Yusuf Has Hacib'in *Kutadgu Bilig* eserinde "yargıç / hüküm veren kişi" anlamında geçer.
- **Metafor:** Karar Ağacı (Decision Tree) algoritmasının her düğümde veriyi inceleyip bir eşik değerine göre "hüküm vermesi / dallara ayırması" özüne doğrudan gönderme yapar.

### 2. Kullanılan Veri Seti: UCI Adult Income (Census)
- **Problem Türü:** İkili Sınıflandırma (Binary Classification).
- **Amaç:** Bir bireyin demografik, eğitim ve iş gücü özelliklerine bakarak yıllık gelirinin 50.000$'ın üzerinde (`>50K`) olup olmadığını tahmin etmek.
- **Veri Boyutu:**
  - Eğitim Seti (`adult.data`): **32.561 satır**
  - Test Seti (`adult.test`): **16.281 satır**
  - Ham Öznitelik Sayısı: **15 sütun** (Yaş, iş sınıfı, eğitim, medeni hal, meslek, ırk, cinsiyet, sermaye kazancı/kaybı, haftalık çalışma saati vb.).
  - Ön işleme sonrası: `fnlwgt` atıldı, One-Hot Encoding ile **91 sayısal sütun** elde edildi.

### 3. Sıfırdan (From Scratch) Neler Yazıldı?
- ❌ **Kullanılmayanlar:** `scikit-learn`'ün ağaç oluşturma, entropi, gini, bilgi kazancı veya metrik fonksiyonları kullanılmadı.
- ✅ **Sıfırdan Yazılanlar:**
  1. `calculate_entropy`: Shannon Entropisi formülü (NumPy).
  2. `calculate_gini`: Gini Safsızlığı formülü (NumPy).
  3. `calculate_information_gain`: İki ölçüte de uyumlu ağırlıklı Bilgi Kazancı.
  4. `DecisionTreeClassifier` & `Node`: Özyinelemeli ikili karar ağacı, eşik arama ve budama mantığı.
  5. `class_weight='balanced'`: Sınıf dengesizliği için ağırlıklı entropi ve ağırlıklı çoğunluk oylaması.
  6. `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`: Değerlendirme metrikleri.

---

# B. Matematiksel Temel (Sınavda Çıkacak Formüller)

## 1. Shannon Entropisi (Entropy)
Bir kümedeki belirsizliğin ve safsızlığın ölçüsüdür.

$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$

- $c$: Sınıf sayısı (İkili sınıflandırma için $c=2$).
- $p_i$: $i$. sınıfa ait örneklerin oranı ($p_i = \frac{N_i}{N}$).
- **Birim:** bit (Shannon).
- **Aralık:** İkili sınıfta $[0.0, 1.0]$.
  - $p_1 = 1.0, p_0 = 0.0 \implies H(S) = 0.0$ (Tam saf / homojen).
  - $p_1 = 0.5, p_0 = 0.5 \implies H(S) = 1.0$ (Maksimum belirsizlik, yazı-tura).

### Sınavda Çıkabilecek Kritik Ayrıntı: $\log_2(0)$ Problemi ve L'Hôpital İspatı
- Bir sınıfa ait hiç örnek yoksa $p_i = 0$ olur. $\log_2(0) = -\infty$ olduğundan tanımsızlık doğar.
- **Çözüm:** L'Hôpital kuralı ile $\lim_{p \to 0^+} p \log_2(p) = 0$ olduğu ispatlanır:
  $$\lim_{p \to 0^+} \frac{\log_2(p)}{\frac{1}{p}} = \lim_{p \to 0^+} \frac{\frac{1}{p \ln 2}}{-\frac{1}{p^2}} = \lim_{p \to 0^+} \left( - \frac{p}{\ln 2} \right) = 0$$
- Bu yüzden kodda $0 \cdot \log_2(0) = 0$ kabul edilir (`p[p > 0]`).

---

## 2. Gini Kirliliği (Gini Impurity)
Rastgele seçilen bir örneğin, kümedeki sınıf dağılımına göre rastgele etiketlenmesi halinde **yanlış sınıflandırılma** olasılığıdır.

$$GI(S) = 1 - \sum_{i=1}^{c} p_i^2$$

- **Aralık:** İkili sınıfta $[0.0, 0.5]$.
  - Tam saf: $GI = 1 - (1^2 + 0^2) = 0.0$.
  - Eşit dağılım ($p_1=0.5, p_0=0.5$): $GI = 1 - (0.5^2 + 0.5^2) = 1 - 0.50 = 0.5$.

---

## 3. Bilgi Kazancı (Information Gain - $IG$)
Bir veri kümesinin belirli bir öznitelik ve eşik değerine göre dallara ayrılması sonucunda belirsizlikteki (entropideki) net düşüştür.

$$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

İkili (sol $L$ ve sağ $R$) bölünme için:

$$IG(S, A) = H(S) - \left( \frac{|S_L|}{|S|} H(S_L) + \frac{|S_R|}{|S|} H(S_R) \right)$$

### Soru: "Neden Ağırlıklı Ortalama Alıyoruz?" (Hocanın Favori Sorusu)
- **Cevap:** Çünkü çocuk düğümlerin barındırdığı örnek sayısı eşit değildir. 100 örnekli bir düğümdeki saflık derecesi ile sadece 2 örnek barındıran bir yaprağın saflık derecesi bilgi teorisi açısından aynı öneme sahip olamaz.
- Ağırlıklı ortalama ($\frac{|S_v|}{|S|}$), her alt kümenin toplam veri içindeki hacmini hesaba katarak **toplam bilgi miktarının korunumunu** sağlar.

---

## 4. Sınavda Çıkabilecek Manuel Hesaplama Örnekleri

### Örnek 1: 14 Örnekli Tenis Verisinde "Wind" Niteliği (`Karar Agaclari Algoritmasi.docx`)
- **Veri:** 14 Gün. Hedef: Tenis Oyna (9 Evet, 5 Hayır).
- **Adım 1 (Kök Düğüm Entropisi):**
  $$p(\text{Evet}) = \frac{9}{14} \approx 0.6429, \quad p(\text{Hayır}) = \frac{5}{14} \approx 0.3571$$
  $$H(S) = - \left( \frac{9}{14}\log_2\frac{9}{14} + \frac{5}{14}\log_2\frac{5}{14} \right) \approx \mathbf{0.940\text{ bit}}$$
- **Adım 2 (Sol Dal: $\text{Wind} = \text{Weak}$, 8 Gün - 6 Evet, 2 Hayır):**
  $$p_1 = \frac{6}{8} = 0.75, \quad p_0 = \frac{2}{8} = 0.25$$
  $$H(\text{Weak}) = - (0.75 \log_2 0.75 + 0.25 \log_2 0.25) = - (-0.3113 - 0.5000) = \mathbf{0.811\text{ bit}}$$
- **Adım 3 (Sağ Dal: $\text{Wind} = \text{Strong}$, 6 Gün - 3 Evet, 3 Hayır):**
  $$p_1 = 0.5, \quad p_0 = 0.5 \implies H(\text{Strong}) = \mathbf{1.000\text{ bit}}$$
- **Adım 4 (Ağırlıklı Çocuk Entropisi):**
  $$H(S|\text{Wind}) = \frac{8}{14} \times 0.811 + \frac{6}{14} \times 1.000 = 0.4636 + 0.4286 = \mathbf{0.892\text{ bit}}$$
- **Adım 5 (Bilgi Kazancı):**
  $$IG(S, \text{Wind}) = 0.940 - 0.892 = \mathbf{0.048\text{ bit}}$$

---

### Örnek 2: 149 Örnekli Sınıf Dengesizliği Problemi (`bilgi_kazanci_ornek_cozum.pdf`)
- **Kök:** 100 Adet Sınıf 1, 49 Adet Sınıf 0 ($N = 149$).
  - $p_1 = 100/149 \approx 0.6711, \quad p_0 = 49/149 \approx 0.3289$
  - **Entropi:** $H(S) = - (0.6711 \log_2 0.6711 + 0.3289 \log_2 0.3289) = \mathbf{0.9138\text{ bit}} \approx \mathbf{0.913\text{ bit}}$
  - **Gini:** $GI(S) = 1 - (0.6711^2 + 0.3289^2) = 1 - 0.5586 = \mathbf{0.4414}$
- **Bölünme:** Sol dala 50 adet Sınıf 1 (0 adet Sınıf 0); Sağ dala 50 adet Sınıf 1, 49 adet Sınıf 0.
  - $H(\text{Sol}) = \mathbf{0.000\text{ bit}}$ (Tam saf yaprak)
  - $H(\text{Sağ}) = - (\frac{50}{99}\log_2\frac{50}{99} + \frac{49}{99}\log_2\frac{49}{99}) = \mathbf{0.9999\text{ bit}}$
  - Ağırlıklı Entropi: $\frac{50}{149}(0.0) + \frac{99}{149}(0.9999) = \mathbf{0.6644\text{ bit}}$
  - **Bilgi Kazancı ($IG$):** $0.9138 - 0.6644 = \mathbf{0.249\text{ bit}}$

---

### Sınav İpucu: Hoca Sınavda "Bilgi Kazancı Hesapla" Derse 4 Adımlık Şablon
1. **Adım 1:** Ebeveyn kümedeki toplam örnek ve sınıf sayılarını yaz, $H(S)$'yi hesapla.
2. **Adım 2:** İlgili özelliğin her bir dalındaki alt grupları yaz, her dalın kendi $H(S_v)$ değerini bul.
3. **Adım 3:** Dalların örnek oranlarını ($|S_v| / |S|$) ilgili dalın entropisiyle çarpıp topla (Koşullu Entropi).
4. **Adım 4:** Ebeveyn entropisinden bu toplamı çıkar: $IG = H(S) - H(S|A)$. Birim olarak mutlaka **"bit"** yaz!

---

# C. Kod Mimarisi ve Önemli Fonksiyonlar (Sunumda Anlatılacaklar)

```text
src/yargucu/
├── data/
│   ├── loader.py        # adult.data ve adult.test okuma, '?' tespiti, '.' temizliği
│   └── preprocess.py    # fnlwgt atma, mode imputation, One-Hot Encoding
├── metrics/
│   ├── entropy.py       # calculate_entropy (sample_weight destekli)
│   ├── gini.py          # calculate_gini (sample_weight destekli)
│   └── information_gain.py # calculate_information_gain
├── tree/
│   ├── node.py          # Node veri sınıfı (feature, threshold, left, right, value, is_leaf)
│   └── decision_tree.py # DecisionTreeClassifier (build_tree, best_split, predict)
├── baselines/
│   ├── sklearn_dt.py    # Scikit-learn karşılaştırma sarmalayıcısı
│   └── xgboost_model.py # XGBoost karşılaştırma sarmalayıcısı
├── evaluation/
│   └── metrics.py       # Pure NumPy: accuracy, precision, recall, f1, confusion_matrix
└── pipeline/            # explore, preprocess, train, compare, optimize, visualize
```

### 1. `_build_tree` Fonksiyonunun Çalışma Mantığı (Özyineleme - Recursion)
Ağaç kök düğümden başlanarak derinlemesine (depth-first) inşa edilir:
1. **Durma Koşulları (Stopping Criteria - Yaprak Oluşturma):**
   - Eğer `depth >= max_depth` ise,
   - Eğer düğümdeki örnek sayısı `n_samples < min_samples_split` ise,
   - Eğer düğümdeki tüm etiketler aynı sınıftaysa (tam homojenlik: `len(np.unique(y)) == 1`),
   $\implies$ Düğüm yaprak (`is_leaf=True`) yapılır ve çoğunluk sınıf etiketi atanır (`_majority_vote`).
2. **Bölünme Arama (`_best_split`):**
   - Tüm özellikler ve aday eşikler taranır. En yüksek Bilgi Kazancını veren `(best_feature, best_threshold)` seçilir.
   - Eğer elde edilen en iyi kazanç $0$'a eşit veya küçükse ya da dallardan birindeki örnek sayısı `min_samples_leaf`'in altındaysa dallanma durdurulur, yaprak yapılır.
3. **Özyinelemeli Dallanma:**
   - Veri maskelenir: $X_{\text{sol}} = X[X[:, f] \le t]$ ve $X_{\text{sağ}} = X[X[:, f] > t]$.
   - `left = self._build_tree(X_left, y_left, depth + 1)`
   - `right = self._build_tree(X_right, y_right, depth + 1)`
   - Bir iç düğüm (`Node`) olarak döndürülür.

### 2. Performans Optimizasyonu (Eşik Tarama Hızı)
- **Problem:** UCI Adult veri setinde 32.561 satır ve 91 öznitelik vardır. Standart bir karar ağacında her sütundaki tüm benzersiz değerler eşik adayı seçilirse tek bir split için milyonlarca $IG$ hesabı gerekir; Python ile eğitilmesi saatler sürer.
- **Çözümümüz:** Eğer bir sütundaki benzersiz sayısal değer sayısı çok fazlaysa (örn. 50'den fazla), tüm değerler yerine persentiller (dilimler) veya subsample alınarak aday eşik sayısı sınırlandırılmıştır.
- **Kazanım:** Model doğruluğundan hiçbir şey kaybetmeden eğitim süresi **saatlerden 1.6 saniyeye** indirilmiştir!

### 3. `class_weight='balanced'` Nasıl Çalışır?
- Dengesiz veri setinde azınlık sınıfı ($y=1$) örneklerinin ağırlığı formülle artırılır:
  $$w_c = \frac{N}{n_{\text{classes}} \times N_c}$$
- `entropy.py` ve `gini.py` içine `sample_weights` entegre edilmiştir: Örnek sayımı yerine ağırlıklı frekans toplamları ($\sum w_i$) alınır.
- Yaprak düğümde çoğunluk oyu belirlenirken de düz sayım yerine ağırlıkların toplamı en büyük olan sınıf kazanır. Böylece azınlık sınıfı yapraklarda ezilmez.

---

# D. Deneysel Sonuçlar ve Grafik Yorumları (Hocaya Gösterilecekler)

Test veri seti (16.281 örnek) üzerindeki resmi sonuç tablosu:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | Eğitim Süresi (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🌲 **Manuel Karar Ağacı (Yargucu)** | **79.51** | **54.40** | **82.06** | **0.6542** | 1.604 sn |
| ⚡ **Scikit-learn DecisionTree** | 79.41 | 54.25 | 81.96 | 0.6529 | 0.096 sn |
| 🚀 **XGBoost Classifier** | 82.85 | 59.34 | 87.03 | 0.7057 | 0.175 sn |

---

### Grafik 1: `model_comparison.png` (Üç Modelin Yarışı)
- **Hocaya Ne Anlatacaksın?**
  1. *"Hocam, sıfırdan yazdığımız Manuel Karar Ağacı ile Scikit-learn'ün DecisionTreeClassifier modelinin sonuçlarına bakarsanız; Doğruluk (%79.51 vs %79.41), Precision (%54.40 vs %54.25) ve Recall (%82.06 vs %81.96) değerleri neredeyse virgülden sonra bile birebir aynıdır. Bu, sıfırdan yazdığımız algoritmanın hatasız çalıştığının deneysel kanıtıdır."*
  2. *"XGBoost ise tek bir ağaç değil, arka arkaya hataları düzelten yüzlerce ağaçtan oluşan bir Topluluk (Ensemble / Gradient Boosting) yöntemi olduğu için %82.85 doğruluk ve %87.03 recall ile doğal olarak bir miktar daha yüksek başarım göstermiştir."*

---

### Grafik 2: `hyperparameter_analysis.png` (Aşırı Öğrenme / Overfitting Kanıtı)
- **Tablo:**
  - Derinlik 3: Train %84.36, Test %84.45 (Underfitting)
  - Derinlik 5: Train %84.71, Test %84.74
  - Derinlik 7: Train %85.78, Test %85.76
  - **Derinlik 10:** Train %86.88, **Test %86.26 (Zirve Noktası - Optimal Genelleme)**
  - **Derinlik 15:** Train %88.78, **Test %85.38 (Aşırı Öğrenme - Overfitting!)**
- **Hocaya Ne Anlatacaksın?**
  *"Hocam, ağaç derinliğini 10'dan 15'e çıkardığımızda eğitim doğruluğu yükselmeye devam etmiş (%88.78), ancak test doğruluğu belirgin şekilde düşmüştür (%85.38). Bu grafik, karar ağaçlarının serbest bırakıldığında eğitim verisini ezberlediğini (overfitting) ve budamanın (max_depth kısıtlamasının) şart olduğunu ispatlamaktadır."*

---

### Grafik 3: `entropy_vs_gini.png` (Entropi vs Gini Eğrileri)
- **Hocaya Ne Anlatacaksın?**
  1. *"Hocam, Entropi fonksiyonu tepe noktasında $1.0\text{ bit}$ değerini alırken, Gini formülü $0.5$ tepe noktasına sahiptir."*
  2. *"Gini eğrisi 2 ile çarpılıp ölçeklendiğinde neredeyse Entropi eğrisiyle üst üste biner. Bu yüzden iki kriter pratikte %98 oranında aynı bölünme kararlarını verir."*
  3. *"Fakat Entropi transandantal bir fonksiyon olan $\log_2$ hesaplaması gerektirdiği için CPU seviyesinde çok daha maliyetlidir. CART algoritmasında Gini'nin tercih edilmesinin asıl sebebi hız ve donanım verimliliğidir."*

---

### Grafik 4: `manual_tree.png` (Ağaç Mimarisi ve Karar Kuralları)
- **Hocaya Ne Anlatacaksın?**
  *"Hocam, bu şema modelimizin veriden öğrendiği ilk 3 seviyeyi göstermektedir. Ağacımızın en tepeye (kök düğüme) koyduğu kural, gelir tahmininde en ayırt edici öznitelik olan medeni hal (`marital-status_Married-civ-spouse`) ve eğitim seviyesidir. Evli bireyler ve yüksek eğitimliler doğrudan sağ dala ayrılmaktadır."*

---

### 4. Sınıf Dengesizliği Mucizesi (Recall Patlaması)
- **Öncesi (Ağırlıksız Model):** Recall = **%49.17**, F1 = 0.60
- **Sonrası (`class_weight='balanced'`):** Recall = **%82.06**, F1 = **0.6542**
- **Açıklama:** Verideki >50K kazanan azınlık grubu (%24) başlangıçta kaçırılıyordu (her 2 zenginden 1'i yanlış tahmin ediliyordu). Ağırlıklandırma sayesinde model azınlık sınıfına öncelik verdi ve yüksek gelirli bireylerin %82'sini doğru yakalamayı başardı.

---

# E. Hocanın Sorabileceği Potansiyel Sorular ve Cevapları (Mülakat Simülasyonu)

### Soru 1: "Neden hazır Scikit-learn varken Karar Ağacını sıfırdan yazdın?"
> **Cevap:** *"Hocam, bu dersin temel amacı makine öğrenmesi kütüphanelerini birer 'kara kutu' (black-box) gibi kullanmak değil; altındaki matematiksel temeli, kirlilik ölçütlerini ve optimizasyon problemlerini bizzat deneyimlemektir. Sıfırdan kodlayarak Shannon Entropisinin türevini, logaritma sınır durumlarını, özyinelemeli ağaç inşasını ve sınıf ağırlıklandırmasının matematiğini derinlemesine kavradık. Ayrıca geliştirdiğimiz modelin Scikit-learn ile birebir aynı metrikleri (%79.51 vs %79.41) üretmesi, teoriyi eksiksiz pratiğe döktüğümüzü kanıtlamıştır."*

---

### Soru 2: "Karar ağaçlarında Aşırı Öğrenmeyi (Overfitting) nasıl engelledin?"
> **Cevap:** *"Karar ağaçları budanmadığı (unpruned) takdirde yapraklarda tek bir örnek kalana kadar büyür ve veriyi ezberler. Biz üç farklı ön budama (pre-pruning) hiperparametresi ekledik:
> 1. `max_depth`: Ağacın maksimum derinliğini sınırladık (Hiperparametre analizimizde en ideal derinliğin 10 olduğunu saptadık).
> 2. `min_samples_split`: Bir düğümün bölünebilmesi için gereken minimum örnek eşiği koyduk.
> 3. `min_samples_leaf`: Bir yaprak düğümde kalması gereken minimum örnek sayısını zorunlu kıldık."*

---

### Soru 3: "Veri setinde eksik değerler (NaN / '?') vardı, bunları nasıl ele aldın?"
> **Cevap:** *"UCI Adult veri setinde eksik değerler `?` işaretiyle temsil ediliyordu. `loader.py` adımında bunları tespit edip `NaN`'a dönüştürdük. `preprocess.py` modülünde ise bilgi kaybını önlemek adına satırları silmek yerine, istatistiksel mod imputation (en sık tekrar eden değer ile doldurma) uyguladık. Test setini doldururken de veri sızıntısını (data leakage) engellemek için eğitim setinden hesaplanan mod değerlerini kullandık."*

---

### Soru 4: "Kategorik ve sayısal verileri modele nasıl verdin?"
> **Cevap:** *"Karar ağaçları teorik olarak kategorik verileri doğrudan bölebilse de, algoritmik standardizasyon ve Scikit-learn/XGBoost ile adil kıyaslama yapabilmek için kategorik değişkenlere **One-Hot Encoding** (`pd.get_dummies`, `drop_first=True`) uyguladık. Böylece tüm girdi matrisi sayısal hale geldi ve algoritmamız $X[:, j] \le \text{threshold}$ mantığıyla hem sayısal hem ikili özellikleri tek bir ortak yapıda bölebilir hale geldi."*

---

### Soru 5: "Gini mi Entropi mi daha iyi? Sen hangisini tercih ettin?"
> **Cevap:** *"Model doğruluğu ve F1 skoru açısından ikisi arasında istatistiksel olarak anlamlı bir fark yoktur (bizim deneylerimizde derinlik 5 için Entropi %84.74 test doğruluğu verirken Gini %85.30 vermiştir). Ancak **hesaplama hızı açısından Gini üstündür**. Çünkü Entropi hesaplanırken CPU seviyesinde maliyetli olan $\log_2$ transandantal fonksiyonu çalıştırılır. Gini ise sadece olasılıkların karelerinin toplamından ($1 - \sum p_i^2$) ibarettir. Bu sebeple CART algoritması varsayılan olarak Gini kullanır."*

---

### Soru 6: "Verideki `fnlwgt` (Final Weight) sütununu neden çıkardın?"
> **Cevap:** *"Census bürosunun benzer sosyo-demografik özellikleri taşıyan insanları nüfusa genellemek için kullandığı bir örneklem ağırlığıdır. Kişinin bireysel gelir kapasitesiyle doğrudan nedensel (causal) bir ilişkisi yoktur, aksine ağaç modellerinde gürültü (noise) yaratmakta ve modelin ezber yapmasına sebep olmaktadır. Bu yüzden çıkarılmıştır."*

---

### Soru 7: "Modelinde neden Accuracy yerine Recall ve F1-Skorunu öne çıkardın?"
> **Cevap:** *"Çünkü veri setimiz dengesizdir; örneklerin %76'sı `<=50K`, sadece %24'ü `>50K`'dır. Bir model hiçbir şey öğrenmeyip herkese 'Düşük Gelirli' dese bile %76 Doğruluk (Accuracy) alabilirdi! Bu yanıltıcı bir başarıdır. Asıl başarı, azınlıkta olan yüksek gelirlileri kaçırmamaktır (Duyarlılık / Recall). Biz `class_weight='balanced'` kullanarak Recall'u %49'dan %82'ye çıkardık ve dengeli F1 skorunu yükselttik."*

---

# F. Hızlı Terminal Komutları (Canlı Gösterim Sırasında)

Eğer hoca *"Bana şu adımı bir çalıştır göster"* derse:

```bash
# 1. Metriklerin elle hesaplama demosu (Entropi, Gini, Tenis IG, 149 örnek):
uv run python main.py metrics-demo

# 2. Modelleri karşılaştır ve tabloyu göster:
uv run python main.py compare

# 3. Manuel ağacı eğit:
uv run python main.py train-manual

# 4. Tüm birim testleri koştur (32 test yeşil yanmalı):
uv run pytest
```
