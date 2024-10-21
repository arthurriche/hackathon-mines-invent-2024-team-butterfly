Nous avons séparé le travail de la façon suivante afin de nommer des responsables pour chaque étape:
- Arthur: Préparation du pitch entreprise
- Ludovic: Preprocessing
- Raphaël: Implémentation du modèle
- Tanguy: Création de la fonction de training et de la pipeline de cross-validation

Nous avons décidé de partager un github sur lequel nous avions séparé nos branches qui ont été fusionnées à la fin du projet dans le main.

Approche:
1) Nous avons décidé de séparer de le travail, bien que la majorité des tâches aient été discutées en commun.
2) Nous avons analysé les premiers exemples sur le DATA-mini pour comprendre quels étaient les difficultés potentielles du préprocessing.
3) En parallèle de cela nous nous sommes inspirées des modèles de segmentation volumétrique existants dans la littérature en particulier dans le secteur de l'imagerie médicale.
4) Nous avons implémenter le modèle Unet3d sans préprocessing ni cross-validation en première approche dans unet3d/unet3d.py.
5) Nous avons implémenté une classe basée sur BaselineDataset afin de supprimer les observations trop dégradées par les nuages: NoCloudDataset dans baseline/dataset.py.
6) Nous avons par ailleurs implémenté une pipeline de cross-validation afin de faire varier les différents hyper-paramètres existants comme le nombre d'époques, de couches ou le learning_rate, une unbalanced loss function ou non et un préprocessing des nuages ou non.
8) Nous avons finalement implémenté une version finale de l'entraînement.

Les deux derniers points peuvent être trouvés dans baseline/cross_val.py et baseline/train.py, respectivement.

Les limites de nos modèles actuellement sont les suivants:
- Nosu n'avons pas eu le temps de cross-validé tous les modèles dans la mesure où le temps était restreint.
- En conséquence, nous aurions pu avoir un meilleur score avec la version dans laquelle on supprime les observations avec des nuages et où on weight la loss si nous avions augmenté le nombre d'époques et/ou augmenter le learning_rate.
- Il est également possible d'implémenté un modèle basé sur une union d'un transformer et d'un unet, potentiellement plus performant pour réaliser la tâche demandée. 
