"""Karar ağacı algoritması birim testleri."""

import numpy as np

from yargucu.tree.decision_tree import DecisionTreeClassifier
from yargucu.tree.node import Node


def test_node_attributes() -> None:
    """Node sınıfının yaprak ve iç düğüm niteliklerini doğrular."""
    # Yaprak düğüm
    leaf = Node(value=1, is_leaf=True)
    assert leaf.is_leaf is True
    assert leaf.value == 1
    assert leaf.left is None
    assert leaf.right is None

    # İç düğüm
    internal = Node(feature_index=0, threshold=0.5, left=leaf, right=leaf, is_leaf=False)
    assert internal.is_leaf is False
    assert internal.value is None
    assert internal.feature_index == 0
    assert internal.threshold == 0.5
    assert internal.left is leaf
    assert internal.right is leaf


def test_xor_problem() -> None:
    """XOR problemini (doğrusal ayrılamaz) karar ağacı ile sınıflandırma testi."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    clf.fit(X, y)

    predictions = clf.predict(X)
    assert np.array_equal(predictions, y)
    assert clf.score(X, y) == 1.0


def test_and_gate() -> None:
    """AND kapısı verisi üzerinde sınıflandırma testi."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    predictions = clf.predict(X)
    assert np.array_equal(predictions, y)


def test_pure_node_early_exit() -> None:
    """Tüm örneklerin aynı sınıfa ait olduğu durumda tek yaprak düğüm üretilmelidir."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 1, 1])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    assert clf.root_ is not None
    assert clf.root_.is_leaf is True
    assert clf.root_.value == 1


def test_gini_criterion_training() -> None:
    """Gini kriteri ile eğitim ve tahmin testi."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
    clf.fit(X, y)

    predictions = clf.predict(X)
    assert np.array_equal(predictions, y)


def test_max_depth_constraint() -> None:
    """max_depth kısıtının ağaç derinliğini sınırlandırdığını doğrular."""
    # Derinlik 1 ile XOR çözülemez
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)

    assert clf.root_ is not None
    assert clf.root_.is_leaf is False
    assert clf.root_.left is not None and clf.root_.left.is_leaf is True
    assert clf.root_.right is not None and clf.root_.right.is_leaf is True


def test_multi_class_classification() -> None:
    """Çok sınıflı veri üzerinde eğitim ve tahmin doğrulaması."""
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
        [9.0, 9.0],
        [9.1, 9.2],
    ])
    y = np.array(["A", "A", "B", "B", "C", "C"])

    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)
    assert clf.score(X, y) == 1.0


def test_single_sample_predict() -> None:
    """Tek bir satır vektörü (1D) verildiğinde tahmin üretebilmelidir."""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    single_pred = clf.predict(np.array([0, 0]))
    assert single_pred[0] == 0


def test_class_weight_balanced() -> None:
    """Sınıf dengeleme (class_weight='balanced') doğrulaması."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 0, 1])

    clf_bal = DecisionTreeClassifier(max_depth=1, class_weight="balanced")
    clf_bal.fit(X, y)
    assert clf_bal.root_ is not None
