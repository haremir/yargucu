# Bilgi Kazancı (Information Gain) Türetmesi ve Manuel Hesaplamalar

## 1. Tanım ve Matematiksel Formül

**Bilgi Kazancı (Information Gain - $IG$)**, bir veri kümesinin belirli bir özniteliğe ($A$) göre dallara ayrılması durumunda belirsizlikte (entropide) meydana gelen beklenen azalmayı ifade eder. Karar ağaçlarında (özellikle ID3 algoritmasında ve C4.5'in temelinde), ağacın her düğümünde en yüksek bilgi kazancını sağlayan öznitelik ve eşik değeri seçilir.

Genel formül:

$$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

İkili (binary split) karar ağaçları için (sol $L$ ve sağ $R$ alt düğümler):

$$IG(S, A) = H(S) - \left( \frac{|S_L|}{|S|} H(S_L) + \frac{|S_R|}{|S|} H(S_R) \right)$$

Burada:
- $H(S)$: Ebeveyn (parent) düğümün bölünmeden önceki başlangıç entropisi.
- $|S|$: Ebeveyn düğümdeki toplam örnek sayısı.
- $|S_L|, |S_R|$: Bölünme sonrası sol ve sağ çocuk düğümlere düşen örnek sayıları ($|S| = |S_L| + |S_R|$).
- $H(S_L), H(S_R)$: Sol ve sağ çocuk düğümlerin entropileri.
- $\sum \frac{|S_v|}{|S|} H(S_v)$: Bölünme sonrasındaki ağırlıklı ortalama kirlilik (koşullu entropi: $H(S|A)$).

---

## 2. Ders Notu Örneği 1: 14 Örnekli Play Tennis Verisinde "Wind" Özelliği
*(Kaynak: `Karar Agaclari Algoritmasi.docx`)*

Hava durumu koşullarına göre tenis oynanıp oynanmayacağını tahmin eden klasik 14 örnekli veri setinde `Wind` (Rüzgar) niteliğinin bilgi kazancı adımları:

### Başlangıç Durumu (Ebeveyn Düğüm $S$):
- **Toplam Örnek ($N$):** $14$
- **Pozitif Sınıf (Yes):** $9$
- **Negatif Sınıf (No):** $5$

#### Adım 1: Ebeveyn Entropisinin Hesaplanması ($H(S)$)
$$p(\text{Yes}) = \frac{9}{14} \approx 0.64286, \quad p(\text{No}) = \frac{5}{14} \approx 0.35714$$
$$H(S) = - \left( \frac{9}{14} \log_2\frac{9}{14} + \frac{5}{14} \log_2\frac{5}{14} \right)$$
$$H(S) = - \big( 0.64286 \times (-0.63743) + 0.35714 \times (-1.48543) \big)$$
$$H(S) = - \big( -0.40978 - 0.53051 \big) = 0.94029 \approx \mathbf{0.940\text{ bit}}$$

---

### Dallara Göre Ayrım (Wind: Weak vs Strong):

#### Adım 2: Sol Dal ($\text{Wind} = \text{Weak}$) Entropisi ($H(S_{\text{Weak}})$)
- **Örnek Sayısı ($N_{\text{Weak}}$):** $8$
- **Yes:** $6$, **No:** $2$
$$p(\text{Yes}) = \frac{6}{8} = 0.75, \quad p(\text{No}) = \frac{2}{8} = 0.25$$
$$H(S_{\text{Weak}}) = - \left( 0.75 \log_2(0.75) + 0.25 \log_2(0.25) \right)$$
$$H(S_{\text{Weak}}) = - \big( 0.75 \times (-0.41504) + 0.25 \times (-2.00000) \big)$$
$$H(S_{\text{Weak}}) = - \big( -0.31128 - 0.50000 \big) = 0.81128 \approx \mathbf{0.811\text{ bit}}$$

#### Adım 3: Sağ Dal ($\text{Wind} = \text{Strong}$) Entropisi ($H(S_{\text{Strong}})$)
- **Örnek Sayısı ($N_{\text{Strong}}$):** $6$
- **Yes:** $3$, **No:** $3$
$$p(\text{Yes}) = \frac{3}{6} = 0.50, \quad p(\text{No}) = \frac{3}{6} = 0.50$$
$$H(S_{\text{Strong}}) = - \left( 0.50 \log_2(0.50) + 0.50 \log_2(0.50) \right)$$
$$H(S_{\text{Strong}}) = - \big( 0.50 \times (-1) + 0.50 \times (-1) \big) = \mathbf{1.000\text{ bit}}$$

---

#### Adım 4: Ağırlıklı Kalan Entropi (Koşullu Entropi $H(S|\text{Wind})$)
$$H(S|\text{Wind}) = \frac{|S_{\text{Weak}}|}{|S|} H(S_{\text{Weak}}) + \frac{|S_{\text{Strong}}|}{|S|} H(S_{\text{Strong}})$$
$$H(S|\text{Wind}) = \left(\frac{8}{14} \times 0.81128\right) + \left(\frac{6}{14} \times 1.00000\right)$$
$$H(S|\text{Wind}) \approx 0.46359 + 0.42857 = \mathbf{0.89216\text{ bit}}$$

#### Adım 5: Bilgi Kazancı ($IG(S, \text{Wind})$)
$$IG(S, \text{Wind}) = H(S) - H(S|\text{Wind})$$
$$IG(S, \text{Wind}) = 0.94029 - 0.89216 = 0.04813 \approx \mathbf{0.048\text{ bit}}$$

> **Yorum:** Rüzgar niteliğine göre bölündüğünde veri setindeki belirsizlik $0.940\text{ bit}$ seviyesinden $0.892\text{ bit}$ seviyesine düşmekte, net bilgi kazancı **$0.048\text{ bit}$** olmaktadır.

---

## 3. Ders Notu Örneği 2: 149 Örnekli Kök Düğüm Ayrımı
*(Kaynak: `bilgi_kazanci_ornek_cozum.pdf`)*

Sınıf dengesizliği içeren 149 örnekli kümede bir özniteliğe göre yapılan ikili bölünme:

### Başlangıç Durumu (Ebeveyn Düğüm $S$):
- **Toplam ($N$):** $149$
- **Sınıf 1:** $100$, **Sınıf 0:** $49$
- Ebeveyn Entropisi (bölüm 01'den): $H(S) = \mathbf{0.9138\text{ bit}}$

### Ayrım Sonrası Alt Düğümler:
1. **Sol Düğüm ($S_L$ - Tam Saf Dal):**
   - $N_L = 50$ (Tümü Sınıf 1, Sınıf 0 = 0)
   - $p_1 = 1.0, p_0 = 0.0 \implies H(S_L) = \mathbf{0.0000\text{ bit}}$

2. **Sağ Düğüm ($S_R$ - Karışık Dal):**
   - $N_R = 99$ (Sınıf 1 = 50, Sınıf 0 = 49)
   - $p_1 = \frac{50}{99} \approx 0.50505, \quad p_0 = \frac{49}{99} \approx 0.49495$
   - $H(S_R) = - \left( \frac{50}{99}\log_2\frac{50}{99} + \frac{49}{99}\log_2\frac{49}{99} \right) \approx \mathbf{0.9999\text{ bit}}$

### Ağırlıklı Çocuk Entropisi:
$$\sum \frac{|S_v|}{|S|} H(S_v) = \left(\frac{50}{149} \times 0.0000\right) + \left(\frac{99}{149} \times 0.99986\right) \approx 0 + 0.66440 = \mathbf{0.6644\text{ bit}}$$

### Bilgi Kazancı:
$$IG = H(S) - \sum \frac{|S_v|}{|S|} H(S_v) = 0.91377 - 0.66440 = 0.24937 \approx \mathbf{0.249\text{ bit}}$$

---

## 4. Kod ve Birim Test Doğrulaması

Projede geliştirilen `src/yargucu/metrics/information_gain.py` modülü, bu analitik hesaplamaları tam olarak doğrulamaktadır:

- `tests/test_information_gain.py::test_tennis_wind_information_gain`:
  - `calculate_information_gain(tennis_parent, tennis_weak, tennis_strong)` $\implies \mathbf{0.048\text{ bit}}$ (Hassasiyet: $10^{-3}$)
- `tests/test_information_gain.py::test_lecture_149_samples_information_gain`:
  - `calculate_information_gain(parent_y, left_y, right_y)` $\implies \mathbf{0.249\text{ bit}}$ (Hassasiyet: $10^{-3}$)

---

## 5. Hoca Sunumu ve Sınav Özeti

| Bölünme Senaryosu | $H(\text{Ebeveyn})$ | $H(\text{Sol Düğüm})$ | $H(\text{Sağ Düğüm})$ | Ağırlıklı Kalan Entropi | Bilgi Kazancı ($IG$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Tenis (Wind Ayrımı)** | $0.940\text{ bit}$ | $0.811\text{ bit}$ | $1.000\text{ bit}$ | $0.892\text{ bit}$ | **$0.048\text{ bit}$** |
| **Ders Notu (149 Örnek)** | $0.913\text{ bit}$ | $0.000\text{ bit}$ | $0.9999\text{ bit}$ | $0.664\text{ bit}$ | **$0.249\text{ bit}$** |
| **İdeal (Tam) Ayrım** | $1.000\text{ bit}$ | $0.000\text{ bit}$ | $0.000\text{ bit}$ | $0.000\text{ bit}$ | **$1.000\text{ bit}$** |
