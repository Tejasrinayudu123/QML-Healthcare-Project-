from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepare_data(X, y, test_size=0.2, random_state=42, n_components=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        return X_train_pca, X_test_pca, y_train, y_test, scaler, pca

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, None
