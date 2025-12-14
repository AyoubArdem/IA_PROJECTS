

# 🌦️ Rainfall Prediction Project

##  Description
Ce projet a pour objectif de **prédire la probabilité de pluie** à partir de données météorologiques (pression, température, humidité, nuages, etc.).  
Il utilise des techniques de **prétraitement des données**, de **rééchantillonnage pour équilibrer les classes**, et un **modèle de réseau de neurones (TensorFlow/Keras)** pour effectuer la classification binaire (`rainfall = yes/no`).



##  Structure du projet
- `Rainfall.csv` : dataset brut contenant les observations météorologiques.
- `notebook/code.py` : script principal avec toutes les étapes (prétraitement, entraînement, évaluation).
- `README.md` : documentation du projet.



##  Étapes principales

### 1. Prétraitement des données
- Nettoyage des colonnes (`strip`, gestion des valeurs manquantes).
- Conversion de la variable cible `rainfall` en labels numériques (`yes → 1`, `no → 0`).
- Suppression des colonnes peu pertinentes (`day`, `winddirection`, `mintemp`, `maxtemp`).
- Normalisation des features avec `StandardScaler`.

### 2. Équilibrage des classes
- Utilisation de **resampling** (`sklearn.utils.resample`) pour équilibrer les classes `yes` et `no`.

### 3. Analyse exploratoire
- Visualisation des distributions avec `seaborn` et `matplotlib`.
- Matrice de corrélation et heatmap pour identifier les variables les plus liées à la pluie.

### 4. Modélisation
- Réseau de neurones avec **Keras Sequential** :
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(x_train.shape[1],)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  ```
- Compilation :
  ```python
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  ```
- Callbacks :
  - `EarlyStopping` pour éviter l’overfitting.

### 5. Évaluation
- Courbes `loss` et `accuracy` (train/validation).


##  Résultats attendus
- Une **loss positive** qui diminue progressivement.
- Une **accuracy** qui s’améliore au fil des epochs (>80% selon la qualité des données).
- Visualisation des performances pour interpréter la convergence du modèle.


##  Technologies utilisées
- **Python 3**
- **Pandas / NumPy** : manipulation des données
- **Matplotlib / Seaborn** : visualisation
- **Scikit-learn** : preprocessing, resampling, split
- **TensorFlow / Keras** : modélisation et entraînement
