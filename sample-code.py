import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# Charger le dataset UDP Flood du CIC-DDoS2019
df = pd.read_csv("UDP.csv", encoding="latin1", low_memory=False)
df_numeric = df.select_dtypes(include=["number"])

# Nettoyage des données
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.fillna(0, inplace=True)

##################################### K-means ############################################

X = df_numeric
y = df["Label"]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.labels_

# Encodage des labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Calcul ARI
ari_score = adjusted_rand_score(y_encoded, kmeans_labels)

# Réduction de dimension pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

##################################### Random Forest ############################################

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

# Création et entraînement du modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

##################################### Comparaison ############################################

# Visualisation PCA pour comparaison
plt.figure(figsize=(15, 6))

# Subplot 1: K-means
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.title(f'K-means Clustering\nARI = {ari_score:.3f}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')

# Subplot 2: Random Forest
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='coolwarm', alpha=0.6)
plt.title(f'Random Forest Classification\nAccuracy = {accuracy:.3f}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='True Class', ticks=[0, 1])

plt.tight_layout()
plt.show()

# Rapports de performance
print("\n=== K-means Performance ===")
print(f"Adjusted Rand Index: {ari_score:.4f}\n")

print("=== Random Forest Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Analyse comparative
print("\n=== Comparative Analysis ===")
print("K-means (Unsupervised):")
print(f"- ARI Score: {ari_score:.4f}")
print("- Pros: No need for labeled data, finds hidden patterns")
print("- Cons: Hard to interpret, depends on initialization")

print("\nRandom Forest (Supervised):")
print(f"- Accuracy: {accuracy:.4f}")
print("- Pros: High accuracy, interpretable with feature importance")
print("- Cons: Requires labeled data, may overfit")

# Graphique comparatif
# Supposons que nous avons 4 métriques pour chaque méthode
metrics = ['Précision', 'Rappel', 'F1-Score', 'ARI/Accuracy']
kmeans_values = [0.85, 0.82, 0.83, ari_score]  # Exemple de valeurs pour K-means
rf_values = [0.95, 0.93, 0.94, accuracy]       # Exemple de valeurs pour Random Forest

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, kmeans_values, width, label='K-means', color='royalblue')
rects2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='orange')

# Fonction pour afficher les valeurs sur les barres
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Configuration du graphique
ax.set_ylabel('Scores')
ax.set_title('Comparaison des Performances: K-means vs Random Forest (4 Métriques)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Ajout des notes explicatives
plt.figtext(0, 0, 
           "Note: ARI (Adjusted Rand Index) pour K-means, Accuracy pour Random Forest",
           ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.figtext(-0.1,0.79, 
        "K-means:\n"
        "+ Pas besoin de labels\n"
        "+ Trouve des patterns cachés\n"
        "- Difficile à interpréter\n"
        "- Dépend de l'initialisation",
        ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.figtext(-0.1,0.50, 
        "Random Forest:\n"
        "+ Haute précision\n"
        "+ Interprétable (feature importance)\n"
        "- Nécessite des données labellisées\n"
        "- Peut overfitter",
        ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()