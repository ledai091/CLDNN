import numpy as np

def get_stratified_test_set(X, y, n_samples_per_class=10):
    indices_class_0 = np.where(y == 0)[0]
    indices_class_1 = np.where(y == 1)[0]

    test_indices_class_0 = np.random.choice(indices_class_0, n_samples_per_class, replace=False)
    test_indices_class_1 = np.random.choice(indices_class_1, n_samples_per_class, replace=False)

    test_indices = np.concatenate([test_indices_class_0, test_indices_class_1])

    mask = np.zeros(len(y), dtype=bool)
    mask[test_indices] = True

    X_test, X_remainder = X[mask], X[~mask]
    y_test, y_remainder = y[mask], y[~mask]

    return X_remainder, X_test, y_remainder, y_test