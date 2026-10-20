# Gini Kirliliği (Gini Impurity) Türetmesi ve Manuel Hesaplamalar

## 1. Tanım ve Matematiksel Formül

**Gini Kirliliği (Gini Impurity - $GI$)**, bir veri kümesinden rastgele seçilen bir örneğin, kümedeki sınıf dağılımına uygun olarak rastgele sınıflandırılması halinde yanlış etiketlenme olasılığını ölçen istatistiksel bir metriktir. **CART (Classification and Regression Trees)** algoritmasının varsayılan safsızlık (kirlilik) ölçütüdür.

Matematiksel formülü:

$$GI(S) = 1 - \sum_{i=1}^{c} p_i^2$$

Alternatif olarak, yanlış sınıflandırma olasılığı cinsinden:

$$GI(S) = \sum_{i=1}^{c} p_i (1 - p_i) = \sum_{i=1}^{c} (p_i - p_i^2) = \sum_{i=1}^{c} p_i - \sum_{i=1}^{c} p_i^2 = 1 - \sum_{i=1}^{c} p_i^2$$

Burada:
- $c$: Veri kümesindeki sınıf sayısı ($c = 2$ için ikili sınıflandırma).
- $p_i$: $i$. sınıfın veri kümesindeki frekans oranı ($p_i = \frac{|S_i|}{|S|}$).
- $\sum_{i=1}^{c} p_i = 1$ olduğu için $GI(S) \in [0, 1 - \frac{1}{c}]$. İkili sınıflarda maksimum değer **$0.5$**'tir.

---

## 2. Gini Kazancı (Gini Gain / Kirlilik Azalması)

Bir karar ağacında $A$ özniteliğine göre ikiye ayrılan bir düğümde Gini kirliliğindeki net azalma (Gini Gain):

$$\Delta GI(S, A) = GI(S) - \left( \frac{|S_L|}{|S|} GI(S_L) + \frac{|S_R|}{|S|} GI(S_R) \right)$$

Algoritma, adayı olan tüm öznitelikler ve eşik değerleri (thresholds) arasından $\Delta GI$ değerini **maksimize eden** bölünmeyi seçer.

---

## 3. Ders Notu Örneği: 149 Örnekli Kök Düğümün Adım Adım Hesabı
*(Kaynak: `bilgi_kazanci_ornek_cozum.pdf` ve `tests/test_gini.py`)*

### Problem Verisi:
- **Toplam Örnek Sayısı ($N$):** $149$
- **Sınıf 1 Örnek Sayısı ($N_1$):** $100$
- **Sınıf 0 Örnek Sayısı ($N_0$):** $49$

### Adım 1: Sınıf Frekans Oranlarının Hesaplanması
$$p_1 = \frac{100}{149} \approx 0.671141$$
$$p_0 = \frac{49}{149} \approx 0.328859$$

### Adım 2: Olasılık Karelerinin Hesaplanması
$$p_1^2 = (0.671141)^2 \approx 0.450430$$
$$p_0^2 = (0.328859)^2 \approx 0.108148$$

### Adım 3: Kareler Toplamının Alınması
$$\sum_{i=0}^{1} p_i^2 = p_1^2 + p_0^2 = 0.450430 + 0.108148 = 0.558578$$

### Adım 4: Gini Kirliliğinin Bulunması
$$GI(S) = 1 - \sum_{i=0}^{1} p_i^2 = 1 - 0.558578 = 0.441422 \approx \mathbf{0.4414}$$

---

### Sol ve Sağ Dalların Gini Kirliliği ve Gini Kazancı

149 örnekli senaryodaki ikili bölünme göz önüne alındığında:
1. **Sol Dal ($S_L$ - 50 adet Sınıf 1, 0 adet Sınıf 0):**
   - $p_1 = 1.0, p_0 = 0.0$
   - $GI(S_L) = 1 - (1.0^2 + 0.0^2) = \mathbf{0.0000}$ *(Tam saf yaprak)*

2. **Sağ Dal ($S_R$ - 50 adet Sınıf 1, 49 adet Sınıf 0):**
   - $N_R = 99$, $p_1 = \frac{50}{99} \approx 0.50505$, $p_0 = \frac{49}{99} \approx 0.49495$
   - $p_1^2 \approx 0.25508, \quad p_0^2 \approx 0.24498 \implies \sum p_i^2 \approx 0.50005$
   - $GI(S_R) = 1 - 0.50005 = 0.49995 \approx \mathbf{0.4999}$

3. **Ağırlıklı Çocuk Kirliliği:**
   $$\frac{50}{149} \times 0.0 + \frac{99}{149} \times 0.49995 = 0 + 0.33218 = \mathbf{0.3322}$$

4. **Gini Kazancı ($\Delta GI$):**
   $$\Delta GI = 0.4414 - 0.3322 = \mathbf{0.1092}$$

---

## 4. Gini ile Shannon Entropisinin Karşılaştırması

| Kriter | Shannon Entropisi ($H$) | Gini Kirliliği ($GI$) |
|:---|:---:|:---:|
| **Formül** | $- \sum p_i \log_2(p_i)$ | $1 - \sum p_i^2$ |
| **İkili Sınıfta Aralık** | $[0.0, \, 1.0]$ bit | $[0.0, \, 0.5]$ |
| **Maksimum Belirsizlik ($p=0.5$)** | $1.0\text{ bit}$ | $0.5$ |
| **Tam Saflık Durumu ($p=1.0$)** | $0.0\text{ bit}$ | $0.0$ |
| **Hesaplama Karmaşıklığı** | **Yüksek** ($\log_2$ transandantal işlem, donanımda yavaş) | **Düşük** (Sadece çarpma ve çıkarma, vektörize ve çok hızlı) |
| **Kullanan Algoritmalar** | ID3, C4.5 | CART (Classification and Regression Trees), Breiman Random Forest |
| **Ağaç Yapısına Etkisi** | Genellikle Gini ile %98+ aynı dallanma kararlarını üretir | Entropiye kıyasla neredeyse farksız ağaç yapısı üretir, ancak çok daha hızlı eğitilir |

### Neden CART Algoritmasında Gini Tercih Edilir?

1. **İşlemci / Donanım Verimliliği:**
   - Logaritma fonksiyonu ($\log_2 x$), CPU seviyesinde Taylor serisi açılımları veya donanım döngüleri ile hesaplanır ve temel kayan nokta (floating point) çarpmalarına göre 10 ila 50 kat daha maliyetlidir.
   - Karar ağacı inşasında milyonlarca eşik değeri adayı (threshold) taranırken her potansiyel bölme için safsızlık hesaplanır. Gini formülü sadece kare alma ($p_i \times p_i$) ve toplama gerektirdiğinden veri boyutu büyüdükçe muazzam bir hız avantajı sağlar.
2. **Model Doğruluğunda Benzerlik:**
   - İki eğri incelendiğinde (bkz: `outputs/figures/entropy_vs_gini.png`), Gini eğrisinin ölçeklenmiş hali ($2 \times GI$) Entropi eğrisine neredeyse teğettir. Bu matematiksel benzerlik sebebiyle pratik uygulamalarda doğruluk kaybı olmadan Gini tercih edilir.

---

## 5. Kod ve Birim Test Doğrulaması

Projede geliştirilen `src/yargucu/metrics/gini.py` modülü:

- `tests/test_gini.py::test_lecture_149_samples_gini`:
  - `calculate_gini(root_y)` $\implies \mathbf{0.4414}$
  - `calculate_gini(left_y)` $\implies \mathbf{0.0000}$
  - `calculate_gini(right_y)` $\implies \mathbf{0.4999}$
  Testler bu teorik değerleri $10^{-4}$ hata payı içerisinde doğrulamaktadır.
