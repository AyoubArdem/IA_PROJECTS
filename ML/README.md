# 🚢 Titanic Survival Prediction

##  Description
Ce projet explore le célèbre dataset du **Titanic** afin de prédire la survie des passagers.  
Il combine **analyse exploratoire des données (EDA)**, **prétraitement**, et un modèle de **régression logistique** pour effectuer une classification binaire (`Survived = 0/1`).



##  Structure du projet
- `Titanic-Dataset.csv` : dataset brut.
-  notebook : script principal.
- `README.md` : documentation du projet.



##  Étapes principales

### 1. Prétraitement des données
- Suppression de colonnes peu pertinentes (`Ticket`).
- Extraction du **nom de famille** à partir de la colonne `Name`.
- Gestion des valeurs manquantes (`Age` remplacé par la moyenne, `Cabin` réduit à sa première lettre).
- Encodage des variables catégorielles (`Sex`, `Embarked`, `Cabin`, `Pclass`) avec **OneHotEncoder**.
- Normalisation des variables numériques (`Age`, `SibSp`, `Parch`, `Fare`) avec **StandardScaler**.

### 2. Analyse exploratoire (EDA)
- Moyenne de survie par classe (`Pclass`), sexe (`Sex`), port d’embarquement (`Embarked`).
- Visualisation des distributions (`seaborn.violinplot`).
- Corrélations entre les variables numériques et la survie.

### 3. Modélisation
- Séparation des données en **train/test** (`train_test_split`).
- Entraînement d’un modèle de **régression logistique** :
  ```python
  from sklearn.linear_model import LogisticRegression
  reg = LogisticRegression()
  reg.fit(x_train, y_train)
  ```
- Évaluation avec :
  - `classification_report` (précision, rappel, f1-score).
  - `confusion_matrix`.



##  Résultats attendus
- Une **accuracy autour de 75–80%** avec la régression logistique.
- Identification des variables les plus influentes (sexe, classe, âge, cabine).
- Visualisations pour mieux comprendre les facteurs de survie.



## 🛠️ Technologies
- Python 3
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
