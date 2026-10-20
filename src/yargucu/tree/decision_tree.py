"""Sıfırdan Karar Ağacı (Decision Tree) sınıfı implementasyonu."""

import numpy as np

from yargucu.metrics.information_gain import calculate_information_gain
from yargucu.tree.node import Node


class DecisionTreeClassifier:
    """Sıfırdan implemente edilen Karar Ağacı sınıflandırıcısı.

    Attributes:
        max_depth: Ağacın ulaşabileceği maksimum derinlik.
        min_samples_split: Bir düğümün bölünmesi için gereken minimum örnek sayısı.
        min_samples_leaf: Bir yaprak düğümde bulunması gereken minimum örnek sayısı.
        criterion: Safsızlık ölçütü ("entropy" veya "gini").
        class_weight: Sınıf dengeleme stratejisi (None, "balanced" veya {sinif: agirlik} sözlüğü).
        max_thresholds: Sürekli özelliklerde denenecek maksimum eşik sayısı (performans için).
        root_: Eğitilmiş karar ağacının kök düğümü.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "entropy",
        class_weight: str | dict | None = None,
        max_thresholds: int = 50,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.max_thresholds = max_thresholds
        self.root_: Node | None = None

    def fit(
        self,
        X: object,
        y: object,
        sample_weight: object | None = None,
    ) -> "DecisionTreeClassifier":
        """Karar ağacı modelini verilen eğitim verisi ile eğitir.

        Args:
            X: Öznitelik matrisi (n_samples, n_features).
            y: Hedef sınıf etiketleri (n_samples,).
            sample_weight: Örnek ağırlıkları dizisi (opsiyonel).

        Returns:
            DecisionTreeClassifier: Eğitilmiş model nesnesi.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(f"X iki boyutlu bir matris olmalıdır, alınan boyut: {X_arr.ndim}")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X ({len(X_arr)}) ve y ({len(y_arr)}) satır sayıları uyuşmuyor."
            )

        # Sınıf dengeleme (class_weight) ağırlıklarını hesapla
        if self.class_weight == "balanced":
            classes, counts = np.unique(y_arr, return_counts=True)
            n_samples = len(y_arr)
            n_classes = len(classes)
            weight_dict = {
                cls: float(n_samples / (n_classes * count))
                for cls, count in zip(classes, counts)
            }
            weights: np.ndarray | None = np.array(
                [weight_dict[label] for label in y_arr], dtype=float
            )
        elif isinstance(self.class_weight, dict):
            weights = np.array([self.class_weight[label] for label in y_arr], dtype=float)
        elif sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
        else:
            weights = None

        self.root_ = self._build_tree(X_arr, y_arr, weights, depth=0)
        return self

    def _majority_vote(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> object:
        """Verilen etiketler arasındaki çoğunluk sınıfını (ağırlıklı mod) bulur."""
        if len(y) == 0:
            return None

        classes = np.unique(y)
        if sample_weight is None:
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        # Ağırlıklı çoğunluk oyu
        weighted_counts = [float(np.sum(sample_weight[y == c])) for c in classes]
        return classes[np.argmax(weighted_counts)]

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        depth: int,
    ) -> Node:
        """Karar ağacını özyinelemeli (recursive) olarak inşa eder.

        Args:
            X: Mevcut düğümdeki öznitelik matrisi.
            y: Mevcut düğümdeki hedef etiketler.
            sample_weight: Mevcut düğümdeki örnek ağırlıkları.
            depth: Ağacın mevcut derinliği.

        Returns:
            Node: Oluşturulan iç düğüm veya yaprak düğüm.
        """
        n_samples = len(y)
        unique_labels = np.unique(y)

        # 1. Durdurma koşulları kontrolü:
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(unique_labels) <= 1
        ):
            return Node(value=self._majority_vote(y, sample_weight), is_leaf=True)

        # 2. En iyi bölünmeyi (best split) bul
        best_feature, best_threshold, best_gain = self._best_split(
            X, y, sample_weight
        )

        # Eğer geçerli bir bölünme bulunamadıysa yaprak düğüm oluştur
        if best_feature is None or best_gain < 0.0:
            return Node(value=self._majority_vote(y, sample_weight), is_leaf=True)

        # 3. Veriyi sol ve sağ alt kümelere böl
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        w_left = sample_weight[left_mask] if sample_weight is not None else None
        w_right = sample_weight[right_mask] if sample_weight is not None else None

        # 4. min_samples_leaf kuralını kontrol et
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return Node(value=self._majority_vote(y, sample_weight), is_leaf=True)

        # 5. Alt çocuk düğümleri özyinelemeli inşa et
        left_child = self._build_tree(X_left, y_left, w_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, w_right, depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            is_leaf=False,
        )

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[int | None, float | None, float]:
        """Tüm özellikler ve eşik değerleri arasından en yüksek kazancı sağlayanı bulur.

        Args:
            X: Öznitelik matrisi.
            y: Hedef etiketler.
            sample_weight: Örnek ağırlıkları.

        Returns:
            tuple: (best_feature_index, best_threshold, best_gain)
        """
        best_gain = -1.0
        best_feature: int | None = None
        best_threshold: float | None = None
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            col = X[:, feat_idx]
            unique_vals = np.unique(col)

            # Tekil değer sayısı 1 ise bölünme yapılamaz
            if len(unique_vals) <= 1:
                continue

            # İkili (one-hot) değişkenler için tek eşik (0.5)
            if len(unique_vals) == 2:
                thresholds = [(unique_vals[0] + unique_vals[1]) / 2.0]
            elif len(unique_vals) > self.max_thresholds:
                # Çok fazla değer varsa yüzdelik dilimler (quantiles) ile sınırla
                percentiles = np.linspace(1, 99, self.max_thresholds)
                thresholds = np.unique(np.percentile(col, percentiles))
            else:
                # Ardışık değerlerin ortalamalarını eşik olarak al
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left

                # min_samples_leaf kısıtını sağla
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_w = sample_weight[left_mask] if sample_weight is not None else None
                right_w = sample_weight[~left_mask] if sample_weight is not None else None

                gain = calculate_information_gain(
                    parent_y=y,
                    left_y=y[left_mask],
                    right_y=y[~left_mask],
                    criterion=self.criterion,
                    parent_weight=sample_weight,
                    left_weight=left_w,
                    right_weight=right_w,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = float(thresh)

        return best_feature, best_threshold, best_gain

    def _predict_single(self, x: np.ndarray) -> object:
        """Tek bir örnek için kökten yaprağa giderek tahmin üretir."""
        if self.root_ is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen önce fit() çağırınız.")

        node = self.root_
        while not node.is_leaf:
            if node.feature_index is None or node.threshold is None:
                break
            if x[node.feature_index] <= node.threshold:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right

        return node.value

    def predict(self, X: object) -> np.ndarray:
        """Verilen örnekler için sınıf tahminleri üretir.

        Args:
            X: Tahmin edilecek örnekler matrisi.

        Returns:
            np.ndarray: Tahmin edilen sınıf etiketleri dizisi.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            return np.array([self._predict_single(X_arr)])
        return np.array([self._predict_single(row) for row in X_arr])

    def score(self, X: object, y: object) -> float:
        """Modelin doğruluk (accuracy) skorunu hesaplar."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y_true))
