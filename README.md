# Projet Spark MS Big Data Télécom : campagnes Kickstarter 

Projet Spark pour le MS Big Data Telecom basé sur des données Kickstarter.


## Exécution du projet
Cloner le repo Git. Les données nécessaires sont déjà présentes, dans le dossier _src/main/resources/_.<br />

Le script build_and_submit.sh pointe vers l'installation de Spark de cette façon:
> path_to_spark="$HOME/spark-2.3.4-bin-hadoop2.7"

Il faut modifier cette ligne si besoin avant d'exécuter le script.


Pour exécuter le Preprocessor (préparation des données, export au format parquet):
> ./build_and_submit.sh Preprocessor

Pour exécuter le Trainer (implémentation du modèle):<br />
(Nécessite d'avoir précédemment exécuté le Preprocessor)
> ./build_and_submit.sh Trainer

Le code source est également disponible sous forme de Jupyter notebook, dans le dossier /notebooks. Les notebooks ont été exportés en format html pour faciliter leur affichage. 


## Résultats
Voici les **F1 score** obtenus avec les différentes configurations du pipeline:
- Modèle de base, données prepared_trainingset: **0.6162**
- Modèle issu de l'optimisation de paramètres, même données: **0.6548**
- Même modèle, données cleanées par le Preprocessor: **0.6576**
- Modèle Random Forest : **0.54**

## Améliorations faites
Voici une liste des améliorations faites, ou des pistes étudiées :
- Lecture du csv d'entrée en **échappant les caractères de quote '\"'**, car sans ça, les virgules dans les textes de description étaient prises pour un séparateur.
- Création de **2 nouvelles colonnes** : mois de lancement de la campagne et nombre de caractères dans la description. Ces paramètres n'amélioraient finalement pas le modèle (F1 score entre 0.64 et 0.65).
- **Ajout de valeurs** pour minDF et elasticNetParam lors de la recherche des meilleurs paramètres du modèle. Ceci a permis d'améliorer le score de prédiction.
- Test d'un **modèle random forest**. Ce modèle donne un résultat très bas (F1 score de 0.54). On pourrait l'optimiser avec une recherche de paramètres, mais le modèle simple prend déjà beaucoup de temps à s'exécuter. Un résultat est disponible sur le notebook associé (dossier notebook).
- **Affichage des coefficients** impactant le plus la régression linéaire, pour voir quels paramètres impactent le plus la régression. Il s'avère que les paramètres impactant le plus la régression sont le pays, de nombreux termes provenant de la transformation en tf-idf et le montant du projet.

## Architecture du code
### Preprocessor
La classe Preprocessor a pour rôle de lire les données d'entrée, de les mettre en forme selon les besoins du modèle, et de les exporter au format parquet.
Voici son fonctionnement :
- Chargement des données dans un dataframe ;
- Cast des colonnes au bon format ;
- Suppression des colonnes inutiles ;
- Retraitement des colonnes ;
- Vérification du contenu ;
- Export au format Parquet.

### Trainer
La classe Trainer a pour rôle de définir des modèles prédisant la réussite d'une campagne Kickstarter.
Voici son fonctionnement :
- Modèle 1 (modèle de base, données prepared_trainingset) ;
- Modèle 2 (modèle issu de l'optimisation de paramètres, même données) ;
- Modèle 3 (même modèle, données cleanées par le Preprocessor) ;
- Affichage des coefficients impactant le plus la régression linéaire ;
- Test du modèle random forest. 

 