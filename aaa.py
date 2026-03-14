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
 


# Charger le dataset UDP Flood du CIC-DDoS2019 (exemple : fichier CSV)
df = pd.read_csv("UDP.csv", encoding="latin1", low_memory=False)
df_numeric = df.select_dtypes(include=["number"])  # Ne garder que les colonnes numériques

# Remplacer les valeurs infinies par NaN
print("\nRemplacer les valeurs infinies par:")
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.fillna(0, inplace=True)  # Option : remplacer NaN par 0

# Aperçu des premières lignes du dataset
print("Aperçu du dataset :")
print(df.head())

# Résumé statistique
print("\nRésumé statistique :")
print(df_numeric.describe())

# Informations sur le dataset
print("\nInformations sur le dataset :")
print(df.info())

# Analyse des colonnes non numériques
print("\nAnalyse des colonnes non numériques :")
print(df.select_dtypes(exclude=["number"]).nunique())

# Visualisation des valeurs manquantes
print("\nVisualisation des valeurs manquantes:")
msno.matrix(df)
plt.title("Visualisation des valeurs manquantes")
plt.show()

# Distribution d'une variable clé (ex : taux de paquets envoyés)
print("\nDistribution d'une variable clé (ex : taux de paquets envoyés):")
plt.figure(figsize=(6, 4))
sns.histplot(df['Flow Duration'], kde=True, bins=20, color="blue")
plt.title("Distribution de la durée des flux")
plt.xlabel("Durée du flux (ms)")
plt.ylabel("Fréquence")
plt.show()

# Boxplot pour analyser la distribution de certaines variables
print("\nBoxplot pour analyser la distribution de certaines variables:")
plt.figure(figsize=(6, 4))
sns.boxplot(x='Label', y='Flow Duration', data=df)
plt.title("Boxplot de la durée des flux par type de trafic")
plt.xlabel("Label (Normal vs. Attaque)")
plt.ylabel("Durée du flux")
plt.show()

# Nettoyage : suppression des NaN
df_cleaned = df_numeric.dropna()

# Vérification de la taille avant échantillonnage
sample_size = min(10000, len(df_cleaned))  # Si moins de 10 000 lignes, on prend tout
df_cleaned_sample = df_cleaned.sample(n=sample_size, random_state=42)

# Sélectionner un sous-ensemble de colonnes (éviter surcharge graphique)
cols_to_plot = df_cleaned_sample.columns[:6]  # Affichage de 6 variables max

# Vérifier la taille après nettoyage
print("\nVérifier la taille après nettoyage:")
print(f"Taille avant nettoyage : {df.shape[0]}")
print(f"Taille après nettoyage : {df_cleaned.shape[0]}")

# Affichage du pairplot
print("\nPairplot pour voir les relations entre les variables numériques:")
sns.pairplot(df_cleaned_sample[cols_to_plot], diag_kind="kde")
plt.suptitle("Pairplot des variables UDP Flood (10 000 échantillons)", y=1.02)
plt.show()

# Matrice de corrélation
print("\nMatrice de corrélation:")
corr_matrix = df_numeric.corr()
#plt.figure(figsize=(26, 20))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', square=True, linewidths=.5,xticklabels=True, yticklabels=True)
plt.title("Matrice de Corrélation - UDP Flood")
fig = plt.gcf()  # or by other means, like plt.subplots
figsize = fig.get_size_inches()
print(figsize)
fig.set_size_inches(figsize * 7)
plt.tight_layout()
plt.savefig('correlation.png', dpi=300)
plt.close()

# Visualisation des types d'attaques
print("\nVisualisation des types d'attaques:")
plt.figure(figsize=(8, 6))
sns.countplot(x='Label', data=df, hue='Label', palette='viridis', legend=False)
plt.title("Répartition des types d'attaques")
plt.xlabel("Type d'attaque")
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation=45)
plt.show()

# Détection des outliers avec IQR
print("\nDétection des outliers avec IQR:")
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outliers = (df_numeric < lower_limit) | (df_numeric > upper_limit)
print("\nNombre de valeurs aberrantes détectées par colonne :")
print(outliers.sum())

# Créer la boîte à moustaches pour la colonne Flow Duration
plt.figure(figsize=(8, 5))
plt.boxplot(df['Flow Duration'], vert=False)
plt.title("Boîte à moustaches pour Flow Duration")
plt.xlabel("Flow Duration")
plt.grid(True)
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


######################### Simulation ##########################################
print("\nSimulation:")
# Fonction pour simuler le trafic normal
def simulate_normal_traffic(devices, server):
    traffic_data = []
    for device in devices:
        flow_duration = random.randint(100, 1000)  # Durée du flux en ms
        num_packets = random.randint(10, 100)     # Nombre de paquets
        traffic_data.append([device, server, flow_duration, num_packets, "Normal"])
    return pd.DataFrame(traffic_data, columns=["Source", "Destination", "Flow Duration", "Num Packets", "Label"])

# Fonction pour simuler une attaque DoS
def simulate_dos_attack(attacker_ip, server):
    traffic_data = []
    for _ in range(1000):  # Envoi de 1000 paquets
        flow_duration = random.randint(1, 10)  # Durée très courte
        num_packets = random.randint(1000, 10000)  # Nombre de paquets très élevé
        traffic_data.append([attacker_ip, server, flow_duration, num_packets, "Attack"])
    return pd.DataFrame(traffic_data, columns=["Source", "Destination", "Flow Duration", "Num Packets", "Label"])

# Fonction pour lancer la simulation
def launch_simulation():
    global all_traffic
    choice = traffic_choice.get()

    if choice == "Normal":
        all_traffic = simulate_normal_traffic(devices, server)
        show_normal_traffic_results(all_traffic)  # Afficher les résultats dans le terminal
        messagebox.showinfo("Simulation", "Simulation du trafic normal lancée !")
    elif choice == "Attack":
        all_traffic = simulate_dos_attack(attacker_ip, server)
        messagebox.showinfo("Simulation", "Simulation d'une attaque DoS lancée !")
    else:
        messagebox.showerror("Erreur", "Veuillez choisir un type de trafic.")

# Fonction pour afficher les résultats du trafic normal dans le terminal
def show_normal_traffic_results(traffic_data):
    print("\n=== Résultats du Trafic Normal ===")
    print(f"Nombre total de flux : {len(traffic_data)}")
    print(f"Durée moyenne des flux : {traffic_data['Flow Duration'].mean():.2f} ms")
    print(f"Nombre moyen de paquets par flux : {traffic_data['Num Packets'].mean():.2f}")
    print("\nAperçu des données générées :")
    print(traffic_data.head())  # Afficher les 5 premières lignes des données

# Fonction pour analyser le trafic
def analyze_traffic():
    if 'all_traffic' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord lancer une simulation.")
        return

    # Préparation des données pour K-means
    X = all_traffic[["Flow Duration", "Num Packets"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application de K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # Ajout des labels au DataFrame
    all_traffic["Cluster"] = labels

    # Visualisation des clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=100, edgecolors='k')
    plt.xlabel("Flow Duration (scaled)")
    plt.ylabel("Num Packets (scaled)")
    plt.title("Clustering du Trafic Réseau (Normal vs. Attaque)")
    plt.show()

    # Détection de l'attaque
    if "Attack" in all_traffic["Label"].values:
        attack_cluster = all_traffic[all_traffic["Cluster"] == 1]
        print("\nTrafic détecté comme anormal (attaque) :")
        print(attack_cluster)
        messagebox.showinfo("Résultat", "Attaque détectée !")
    else:
        messagebox.showinfo("Résultat", "Aucune attaque détectée.")

# Configuration de l'interface graphique
root = tk.Tk()
root.title("Simulation de Trafic IoT")
root.geometry("400x200")

# Variables globales
devices = [f"Device_{i}" for i in range(1, 13)]  # 10 capteurs et 2 caméras
server = "Server_Central"
attacker_ip = "Attacker_IP"
traffic_choice = tk.StringVar(value="Normal")  # Choix par défaut

# Widgets de l'interface
label = tk.Label(root, text="Choisissez le type de trafic à simuler :")
label.pack(pady=10)

normal_radio = tk.Radiobutton(root, text="Trafic Normal", variable=traffic_choice, value="Normal")
normal_radio.pack()

attack_radio = tk.Radiobutton(root, text="Attaque DoS", variable=traffic_choice, value="Attack")
attack_radio.pack()

simulate_button = tk.Button(root, text="Lancer la Simulation", command=launch_simulation)
simulate_button.pack(pady=10)

analyze_button = tk.Button(root, text="Analyser le Trafic", command=analyze_traffic)
analyze_button.pack(pady=10)

# Lancer l'interface
root.mainloop()