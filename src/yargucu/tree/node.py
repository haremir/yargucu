"""Karar ağacı düğüm (Node) sınıfı."""


class Node:
    """Karar ağacı iç veya yaprak düğümünü temsil eden sınıf.

    Attributes:
        feature_index: Bölünmenin yapıldığı özelliğin sütun indeksi (iç düğümlerde).
        threshold: Bölünme eşik değeri (x[feature_index] <= threshold -> sol).
        left: Sol alt çocuk düğüm.
        right: Sağ alt çocuk düğüm.
        value: Düğümün tahmin ettiği sınıf etiketi (yalnızca yaprak düğümlerde).
        is_leaf: Düğümün bir yaprak olup olmadığını belirtir.
    """

    def __init__(
        self,
        feature_index: int | None = None,
        threshold: float | None = None,
        left: "Node | None" = None,
        right: "Node | None" = None,
        value: object | None = None,
        is_leaf: bool = False,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

        # Eğer düğüm yapraksa sol ve sağ çocuklar None olmalı, aksi halde value None olmalı
        if self.is_leaf:
            self.left = None
            self.right = None
        else:
            self.value = None
