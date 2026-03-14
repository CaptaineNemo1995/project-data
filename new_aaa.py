import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.utils import resample
import tkinter as tk
from tkinter import messagebox
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score


# Charger le dataset UDP Flood du CIC-DDoS2019 (exemple : fichier CSV)
df = pd.read_csv("UDP.csv", encoding="latin1", low_memory=False)
df_numeric = df.select_dtypes(include=["number"])  # Ne garder que les colonnes numériques

# Remplacer les valeurs infinies par NaN
print("\nRemplacer les valeurs infinies par:")
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.fillna(0, inplace=True)  # Option : remplacer NaN par 0

# # Aperçu des premières lignes du dataset
# print("Aperçu du dataset :")
# print(df.head())

# # Résumé statistique
# print("\nRésumé statistique :")
# print(df_numeric.describe())

# # Informations sur le dataset
# print("\nInformations sur le dataset :")
# print(df.info())

# # Analyse des colonnes non numériques
# print("\nAnalyse des colonnes non numériques :")
# print(df.select_dtypes(exclude=["number"]).nunique())

# # Visualisation des valeurs manquantes
# print("\nVisualisation des valeurs manquantes:")
# msno.matrix(df)
# plt.title("Visualisation des valeurs manquantes")
# plt.show()

# # Distribution d'une variable clé (ex : taux de paquets envoyés)
# print("\nDistribution d'une variable clé (ex : taux de paquets envoyés):")
# plt.figure(figsize=(6, 4))
# sns.histplot(df['Flow Duration'], kde=True, bins=20, color="blue")
# plt.title("Distribution de la durée des flux")
# plt.xlabel("Durée du flux (ms)")
# plt.ylabel("Fréquence")
# plt.show()

# # Boxplot pour analyser la distribution de certaines variables
# print("\nBoxplot pour analyser la distribution de certaines variables:")
# plt.figure(figsize=(6, 4))
# sns.boxplot(x='Label', y='Flow Duration', data=df)
# plt.title("Boxplot de la durée des flux par type de trafic")
# plt.xlabel("Label (Normal vs. Attaque)")
# plt.ylabel("Durée du flux")
# plt.show()

# # Nettoyage : suppression des NaN
# df_cleaned = df_numeric.dropna()

# # Vérification de la taille avant échantillonnage
# sample_size = min(10000, len(df_cleaned))  # Si moins de 10 000 lignes, on prend tout
# df_cleaned_sample = df_cleaned.sample(n=sample_size, random_state=42)

# # Sélectionner un sous-ensemble de colonnes (éviter surcharge graphique)
# cols_to_plot = df_cleaned_sample.columns[:6]  # Affichage de 6 variables max

# # Vérifier la taille après nettoyage
# print("\nVérifier la taille après nettoyage:")
# print(f"Taille avant nettoyage : {df.shape[0]}")
# print(f"Taille après nettoyage : {df_cleaned.shape[0]}")

# # Affichage du pairplot
# print("\nPairplot pour voir les relations entre les variables numériques:")
# sns.pairplot(df_cleaned_sample[cols_to_plot], diag_kind="kde")
# plt.suptitle("Pairplot des variables UDP Flood (10 000 échantillons)", y=1.02)
# plt.show()

# # Matrice de corrélation
# print("\nMatrice de corrélation:")
# corr_matrix = df_numeric.corr()
# #plt.figure(figsize=(26, 20))
# sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', square=True, linewidths=.5,xticklabels=True, yticklabels=True)
# plt.title("Matrice de Corrélation - UDP Flood")
# fig = plt.gcf()  # or by other means, like plt.subplots
# figsize = fig.get_size_inches()
# print(figsize)
# fig.set_size_inches(figsize * 7)
# plt.tight_layout()
# plt.savefig('correlation.png', dpi=300)
# plt.close()

# # Visualisation des types d'attaques
# print("\nVisualisation des types d'attaques:")
# plt.figure(figsize=(8, 6))
# sns.countplot(x='Label', data=df, hue='Label', palette='viridis', legend=False)
# plt.title("Répartition des types d'attaques")
# plt.xlabel("Type d'attaque")
# plt.ylabel("Nombre d'occurrences")
# plt.xticks(rotation=45)
# plt.show()

# # Détection des outliers avec IQR
# print("\nDétection des outliers avec IQR:")
# Q1 = df_numeric.quantile(0.25)
# Q3 = df_numeric.quantile(0.75)
# IQR = Q3 - Q1
# lower_limit = Q1 - 1.5 * IQR
# upper_limit = Q3 + 1.5 * IQR
# outliers = (df_numeric < lower_limit) | (df_numeric > upper_limit)
# print("\nNombre de valeurs aberrantes détectées par colonne :")
# print(outliers.sum())

# # Créer la boîte à moustaches pour la colonne Flow Duration
# plt.figure(figsize=(8, 5))
# plt.boxplot(df['Flow Duration'], vert=False)
# plt.title("Boîte à moustaches pour Flow Duration")
# plt.xlabel("Flow Duration")
# plt.grid(True)
##################################### K-means ############################################

X = df_numeric  # Prendre uniquement les colonnes numériques
y = df["Label"]  # Remplacez "Label" par le vrai nom de la colonne cible si différent

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(df_numeric)

# Déterminer le nombre optimal de clusters (méthode du coude)
print("\nDéterminer le nombre optimal de clusters (méthode du coude):")
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans2 = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans2.fit(X)
    inertia.append(kmeans2.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title("Méthode du coude pour choisir k")
plt.show()

# Application de K-means
kmeans = KMeans(n_clusters=5, random_state=42)  # (5) clusters pour (5) espèces
kmeans.fit(X)
# Résultats
labels = kmeans.labels_  # Attribuer chaque échantillon à un cluster
centers = kmeans.cluster_centers_  # Centres des clusters

# Réduction de la dimensionnalité (pour visualiser en 2D)
pca = PCA(n_components=2)  # Réduire à 2 dimensions pour la visualisation
X_pca = pca.fit_transform(X)
# Visualisation des clusters
print("\nVisualisation des clusters K-means:")
plt.figure(figsize=(8, 6))
# Tracer les points colorés par cluster
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=100, edgecolors='k', label='Échantillons')
# Tracer les centres des clusters
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centres')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Visualisation des clusters K-means')
plt.legend()
plt.show()

# Clustering hiérarchique avec échantillonnage
print("\nClustering hiérarchique avec échantillonnage:")
# Échantillonnage aléatoire de 10 000 points pour limiter la mémoire
sample_size = min(10000,len(X))  
X_sample = resample(X, n_samples=sample_size, random_state=42)

# Appliquer linkage sur cet échantillon réduit
linked = linkage(X_sample, method='ward')

# Affichage du dendrogramme
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5, color_threshold=100)
plt.axhline(y=100, color='r', linestyle='--', label="Seuil de coupure")
plt.title("Dendrogramme du Clustering Hiérarchique (échantillon)")
plt.xlabel("Échantillons")
plt.ylabel("Distance")
plt.legend()
plt.show()

##################################### Random Forest ############################################

# Préparation des données
# On utilise df_numeric comme features et la colonne 'Label' comme target
X = df_numeric
y = df['Label']

# Séparation train/test (70%/30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisation des données (comme pour K-means)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création et entraînement du modèle Random Forest
print("\nEntraînement du modèle Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, 
                                random_state=42,
                                max_depth=10,
                                class_weight='balanced')  # Utile si classes déséquilibrées

rf_model.fit(X_train_scaled, y_train)

# Prédictions et évaluation
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probabilités pour la classe positive

print("\nPerformance de Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
print("\nMatrice de confusion:")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)
plt.title('Matrice de confusion - Random Forest')
plt.ylabel('Vrai label')
plt.xlabel('Prédiction')
plt.show()

# Feature importance
print("\nImportance des features:")
feature_importance = pd.Series(rf_model.feature_importances_, 
                              index=df_numeric.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.head(15).plot(kind='barh')
plt.title('Top 15 des features les plus importantes')
plt.show()

##################################### Régression Linéaire ############################################

# Pour la régression linéaire, nous devons encoder les labels en valeurs numériques
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Séparation train/test (70%/30%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création et entraînement du modèle de régression linéaire
print("\nEntraînement du modèle de Régression Linéaire...")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Prédictions
y_pred_lin = lin_reg.predict(X_test_scaled)

# Pour la classification, nous pouvons arrondir les prédictions
y_pred_lin_class = np.round(y_pred_lin).astype(int)

# Évaluation
print("\nPerformance de la Régression Linéaire:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lin):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lin):.4f}")
print("\nPerformance en classification (après arrondi):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lin_class):.4f}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred_lin_class))

# Matrice de confusion
print("\nMatrice de confusion (Régression Linéaire):")
conf_matrix_lin = confusion_matrix(y_test, y_pred_lin_class)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lin, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion - Régression Linéaire')
plt.ylabel('Vrai label')
plt.xlabel('Prédiction')
plt.show()

##################################### Régression Logistique ############################################
# La régression logistique est souvent plus adaptée pour la classification

print("\nEntraînement du modèle de Régression Logistique...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Prédictions
y_pred_log = log_reg.predict(X_test_scaled)

# Évaluation
print("\nPerformance de la Régression Logistique:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred_log))

# Matrice de confusion
print("\nMatrice de confusion (Régression Logistique):")
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_log, annot=True, fmt='d', cmap='Greens')
plt.title('Matrice de confusion - Régression Logistique')
plt.ylabel('Vrai label')
plt.xlabel('Prédiction')
plt.show()

##################################### Comparaison des trois méthodes ############################################
# 1. Préparation des données communes
X = df_numeric.values
y = df['Label'].values

# Encodage des labels si nécessaire
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-means (non supervisé)
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters pour normal/attaque
kmeans.fit(X_scaled)
kmeans_labels = kmeans.labels_

# Calcul ARI
ari_score = adjusted_rand_score(y_encoded, kmeans_labels)

# 3. Random Forest (supervisé)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 4. Visualisation PCA pour comparaison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

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

# Subplot 3: Régression Logistique
plt.subplot(1, 3, 3)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_log, cmap='autumn', alpha=0.6)
plt.title(f'Régression Logistique\nAccuracy = {accuracy_score(y_test, y_pred_log):.3f}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Predicted Class')

plt.tight_layout()
plt.show()

# 5. Rapports de performance comparés
print("\n=== Comparaison des Performances ===")
print("K-means (Non supervisé):")
print(f"- ARI Score: {ari_score:.4f}\n")

print("Random Forest (Supervisé):")
print(f"- Accuracy: {accuracy:.4f}")
print(f"- F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}\n")

print("Régression Logistique (Supervisé):")
print(f"- Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"- F1-Score: {f1_score(y_test, y_pred_log, average='weighted'):.4f}\n")

# 7. Analyse comparative
print("\n=== Comparative Analysis ===")
print("K-means (Unsupervised):")
print(f"- ARI Score: {ari_score:.4f}")
print("- Pros: No need for labeled data, finds hidden patterns")
print("- Cons: Hard to interpret, depends on initialization")

print("\nRandom Forest (Supervised):")
print(f"- Accuracy: {accuracy:.4f}")
print("- Pros: High accuracy, interpretable with feature importance")
print("- Cons: Requires labeled data, may overfit")

# 8. Graphique comparative

# 6. Graphique comparatif des performances
metrics = ['Accuracy', 'Précision', 'Rappel', 'F1-Score']
kmeans_values = [ari_score, 0, 0, 0]  # ARI seulement pour K-means
rf_values = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred, average='weighted'),
    recall_score(y_test, y_pred, average='weighted'),
    f1_score(y_test, y_pred, average='weighted')
]
logreg_values = [
    accuracy_score(y_test, y_pred_log),
    precision_score(y_test, y_pred_log, average='weighted'),
    recall_score(y_test, y_pred_log, average='weighted'),
    f1_score(y_test, y_pred_log, average='weighted')
]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, kmeans_values, width, label='K-means', color='royalblue')
rects2 = ax.bar(x, rf_values, width, label='Random Forest', color='orange')
rects3 = ax.bar(x + width, logreg_values, width, label='Régression Logistique', color='green')

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
autolabel(rects3)

# Configuration du graphique
ax.set_ylabel('Scores')
ax.set_title('Comparaison des Performances des Trois Modèles')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 9. Conclusion
print("\n=== Conclusion ===")
print("K-means clustering, while unsupervised, was able to find some structure in the data as indicated by the ARI score.")
print("Random Forest and Logistic Regression, both supervised methods, performed well with high accuracy and F1 scores.")
print("In practice, the choice of model depends on the specific use case, data availability, and the need for interpretability.")