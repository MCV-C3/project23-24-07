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
| Dense SIFT (Fine-tuned) | 76.53 |
| Dense SIFT (Fine-tuned w/ LDA) | 81.69 |
| Dense SIFT (Fine-tuned w/ LDA + MinMax) | 82.10 |

## Task2 work

## Task3 work

## Task4 work

