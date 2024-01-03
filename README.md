# Project
State shortly what you did during each week. Just a table with the main results is enough. Remind to upload a brief presentation (pptx?) at virtual campus. Do not modify the previous weeks code. If you want to reuse it, just copy it on the corespondig week folder.

#### Team 7
* Marco Cordón
* Iñaki Lacunza
* Cristian Gutiérrez

## Bag of Visual Words (BoVW) classification
Our tuned model with the optimized parameters is **Dense SIFT with PCA and SVM clf** which obtained a maximum Test Accuracy score of 83.02 during an Optuna optimization run.

```python
Step size = 18
Number of Features = 185
Octave Layers = 2
k_codebook = 198
Classifier = SVM (with RBF Kernel)
PCA (with 109 components)
```

We cross-validated every experiment and decided to use it as our metric of choice to decide for an improvement:
|    | Cross-validation Score |
| -- | ---------------------- |
| SIFT | 57.99 |
| SIFT (w/ LDA) | 66.26 |
| Dense SIFT | 76.53 |
| Dense SIFT & Scale x2 | 62.72 |
| Dense SIFT & Scale x4 | 36.76 |
| Dense SIFT (w/ PCA) | 76.01 |
| Dense SIFT (w/ LDA) | 81.69 |
| Dense SIFT (w/ LDA + StandardScaler) | 81.03 |
| Dense SIFT (w/ LDA + MinMax) | 82.10 |
| Dense SIFT (w/ LDA + Normalizer) | 81.58 |
| Dense SIFT (K = 198) | 77.20 |
| Dense SIFT (w/ PCA) & (K = 198) | 76.56 |
| Dense SIFT (w/ LDA) & (K = 198) | 84.71 |
| Dense SIFT (KNN Dist = Euclidean) | 77.2 |
| Dense SIFT (KNN Dist = Cosine) | 75.74 |
| Dense SIFT (KNN Dist = Jaccard) | 73.4 |
| Dense SIFT + SVM (Kernel = Linear) | 77.53 |
| Dense SIFT + SVM (Kernel = RBF) | 82.78 |
| Dense SIFT + SVM (Kernel = Hist Inters) | 80.06 |
| Dense SIFT (w/ PCA) + SVM (Kernel = Linear) | 77.42 |
| Dense SIFT (w/ PCA) + SVM (Kernel = RBF) | 81.32 |
| Dense SIFT (w/ PCA) + SVM (Kernel = Hist Inters) | 78.57 |
| Dense SIFT (w/ LDA) + SVM (Kernel = Linear) | 86.27 |
| Dense SIFT (w/ LDA) + SVM (Kernel = RBF) | 86.42 |


## Task2 work

## Task3 work

## Task4 work

