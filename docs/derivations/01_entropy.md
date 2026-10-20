# Shannon Entropisi Türetmesi ve Manuel Hesaplamalar

## 1. Tanım ve Matematiksel Formül

**Shannon Entropisi** ($H(S)$), bilgi teorisinde bir rastgele değişkenin veya veri kümesinin içerdiği ortalama belirsizlik (safsızlık / kirlilik) düzeyini ölçer. Karar ağaçlarında (ID3 ve C4.5 algoritmalarında), bir düğümdeki sınıf homojenliğini değerlendirmek ve en bilgilendirici bölme eşiğini seçmek için kullanılır.

$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$

Burada:
- $c$: Veri kümesindeki farklı sınıf sayısı (ör. ikili sınıflandırma için $c = 2$).
- $p_i$: $i$. sınıfa ait örneklerin toplam örnek sayısına oranı ($p_i = \frac{|S_i|}{|S|}$).
- $\log_2$: 2 tabanında logaritma. Bilgi birimi **bit**tir (Shannon).

---

## 2. $p = 0$ Sınır Durumu: $\log_2(0)$ Neden 0 Kabul Edilir?

$\log_2(0)$ matematiksel olarak eksi sonsuza ($-\infty$) gider ve tanımsızdır. Ancak bir sınıfa ait hiç örnek yoksa ($p_i = 0$), o sınıfın sistemin belirsizliğine hiçbir katkısı olmamalıdır. 

Bu durum analitik olarak **L'Hôpital Kuralı** ile doğrulanır:

$$\lim_{p \to 0^+} p \log_2(p) = \lim_{p \to 0^+} \frac{\log_2(p)}{\frac{1}{p}} = \lim_{p \to 0^+} \frac{\frac{1}{p \ln 2}}{-\frac{1}{p^2}} = \lim_{p \to 0^+} \left( - \frac{p}{\ln 2} \right) = 0$$

> **Kodlama Kuralı:** Yazılım implementasyonunda $p_i = 0$ olan sınıflar toplam formülüne dahil edilmez veya `p[p > 0]` filtresi uygulanarak $0 \cdot \log_2(0) = 0$ kabul edilir. Bu sayede sayısal tanımsızlıklar (`NaN` / `inf`) önlenir.

---

## 3. Ders Notu Örneği: 149 Örnekli Kök Düğümün Adım Adım Hesabı
*(Kaynak: `bilgi_kazanci_ornek_cozum.pdf`)*

### Problem Verisi:
- **Toplam Örnek Sayısı ($N$):** $149$
- **Sınıf 1 Örnek Sayısı ($N_1$):** $100$
- **Sınıf 0 Örnek Sayısı ($N_0$):** $49$

### Adım 1: Sınıf Olasılıklarının Hesaplanması
$$p_1 = \frac{100}{149} \approx 0.67114$$
$$p_0 = \frac{49}{149} \approx 0.32886$$

### Adım 2: 2 Tabanında Logaritmaların Hesaplanması
$$\log_2(p_1) = \frac{\ln(0.67114)}{\ln(2)} \approx \frac{-0.39878}{0.69315} \approx -0.57531$$
$$\log_2(p_0) = \frac{\ln(0.32886)}{\ln(2)} \approx \frac{-1.11213}{0.69315} \approx -1.60447$$

### Adım 3: Çarpımların Alınması
$$p_1 \cdot \log_2(p_1) = 0.67114 \times (-0.57531) \approx -0.38612$$
$$p_0 \cdot \log_2(p_0) = 0.32886 \times (-1.60447) \approx -0.52765$$

### Adım 4: Entropi Toplamı
$$H(S) = - \Big( [p_1 \cdot \log_2(p_1)] + [p_0 \cdot \log_2(p_0)] \Big)$$
$$H(S) = - \Big( -0.38612 - 0.52765 \Big) = - (-0.91377)$$
$$H(S) \approx \mathbf{0.9138\text{ bit}} \quad (\approx \mathbf{0.913\text{ bit}})$$

---

## 4. Kritik Referans Değerleri (Sunum / Sınav Notu)

| Düğüm Durumu | Dağılım ($p_1, p_2$) | Entropi Değeri | Anlamı |
|:---|:---:|:---:|:---|
| **Tam Saf (Homojen)** | $p_1 = 1.0, p_2 = 0.0$ | **$0.000\text{ bit}$** | Sıfır belirsizlik, yaprak düğüm |
| **Ders Notu (149 Örnek)** | $p_1 = \frac{100}{149}, p_0 = \frac{49}{149}$ | **$0.913\text{ bit}$** | Yüksek belirsizlik |
| **Maksimum Belirsizlik** | $p_1 = 0.5, p_2 = 0.5$ | **$1.000\text{ bit}$** | Yazı-tura belirsizliği, eşit dağılım |
