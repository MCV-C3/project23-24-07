# Project
State shortly what you did during each week. Just a table with the main results is enough. Remind to upload a brief presentation (pptx?) at virtual campus. Do not modify the previous weeks code. If you want to reuse it, just copy it on the corespondig week folder.

#### Team 7
* Marco Cordón
* Iñaki Lacunza
* Cristian Gutiérrez

## Bag of Visual Words (BoVW) classification
Our tuned model with the optimized parameters is **Dense SIFT (w/ PCA) + SVM (Kernel = RBF)** which obtained a maximum Test Accuracy score of 83.02 during an Optuna optimization run. 

In terms of the Cross-validation score the best model is **Dense SIFT (w/ LDA) + SVM (Kernel = RBF)** which obtained a maximum cross-val score of 86.42.

```
Step size = 18
Number of Features = 185
Octave Layers = 2
k_codebook = 198
Classifier = SVM (with RBF Kernel)
PCA (with 109 components) / LDA (with 7 components)
```

We cross-validated every experiment and decided to use it as our metric of choice to decide for an improvement:
|    | Test Accuracy | Cross-validation Score |
| -- | ------------- | ---------------------- |
| SIFT | 59.11 | 57.99 |
| SIFT (w/ LDA) | 62.95 | 66.26 |
| Dense SIFT | 76.95 | 76.53 |
| Dense SIFT & Scale x2 |63.44| 62.72 |
| Dense SIFT & Scale x4 |36.68| 36.76 |
| Dense SIFT (w/ PCA) |77.08| 76.01 |
| Dense SIFT (w/ LDA) |78.44| 81.69 |
| Dense SIFT (w/ LDA + StandardScaler) |78.07| 81.03 |
| Dense SIFT (w/ LDA + MinMax) |78.93| 82.10 |
| Dense SIFT (w/ LDA + Normalizer) |79.43| 81.58 |
| Dense SIFT (K = 198) |78.81| 77.20 |
| Dense SIFT (w/ PCA) & (K = 198) |76.33| 76.56 |
| Dense SIFT (w/ LDA) & (K = 198) |80.17| 84.71 |
| Dense SIFT (KNN Dist = Euclidean) |78.81| 77.2 |
| Dense SIFT (KNN Dist = Cosine) |75.96| 75.74 |
| Dense SIFT (KNN Dist = Jaccard) |73.98| 73.4 |
| Dense SIFT + SVM (Kernel = Linear) |78.56| 77.53 |
| Dense SIFT + SVM (Kernel = RBF) |82.90| 82.78 |
| Dense SIFT + SVM (Kernel = Hist Inters) |81.41| 80.06 |
| Dense SIFT (w/ PCA) + SVM (Kernel = Linear) |78.93| 77.42 |
| **Dense SIFT (w/ PCA) + SVM (Kernel = RBF)** |**83.02**| **81.32** |
| Dense SIFT (w/ PCA) + SVM (Kernel = Hist Inters) |79.93| 78.57 |
| Dense SIFT (w/ LDA) + SVM (Kernel = Linear) |81.66| 86.27 |
| **Dense SIFT (w/ LDA) + SVM (Kernel = RBF)** |**80.67**| **86.42** |
| Dense SIFT (w/ LDA) + SVM (Kernel = Hist Inters) |80.55| 85.34|
| Dense SIFT + SVM (Spatial Pyramids Level = 2) | 81.29 | - |
| Dense SIFT (w/ PCA) + SVM (Spatial Pyramids Level = 2) | 81.91| - |
| Fisher Vectors (GMM n = 64) | 78.93 | - |
| Fisher Vectors w/ PCA (GMM n = 64) | 79.43 | - |
| Fisher Vectors w/ LDA (GMM n= 64) |  58.24| - |

<img width="840" alt="Screenshot 2024-01-03 at 17 02 37" src="https://github.com/MCV-C3/project23-24-07/assets/57730982/4e7c9a04-e50f-4a38-b7a2-bff3516622c7">


## Deep learning classification with MLP

<u>Best Params</u>

- Patch Based MLP (5 Layers)
- Patch Size = 32
- Features = 5th hidden layer
- BoVW with k = 621
- Standard Scaler
- SVM with kernel = RBF

*Note that all the experiment logs are available under the directory `/ghome/group07/`.*

| Model | Val-Test Accuracy |
| --- | --- |
| MLP (1 Layer) | 0.6456 |
| MLP (2 Layers) | 0.5724 |
| MLP (3 Layers) | 0.6332 |
| MLP (3 Layers) + Image size (256) | 0.5898 |
| MLP (3 Layers) + Adam Optimizer | 0.5712 |
| MLP (3 Layers) + Reduced # Units | 0.5588 |
| MLP (4 Layers) | 0.6121 |
| MLP (5 Layers) | 0.6418 |
| MLP (5 Layers) + L2 Regularization | 0.6034 |
| MLP (1 L) + SVM @ 1st hidden feat | 0.6617 |
| MLP (2 L) + SVM @ 2nd hidden feat | 0.6691 |
| MLP (3 L) + SVM @ 3rd hidden feat | 0.6840 |
| MLP (4 L) + SVM @ 2nd hidden feat | 0.6852 |
| MLP (5 L) + SVM @ 2nd hidden feat | 0.6753 |
| Patch MLP (1 L) + SVM & BoVW @ 1st hidden feat | 0.6741 |
| Patch MLP (2 L) + SVM & BoVW @ 1st hidden feat | 0.7236 |
| Patch MLP (3 L) + SVM & BoVW @ 1st hidden feat | 0.7211 |
| Patch MLP (4 L) + SVM & BoVW @ 1st hidden feat | 0.7261 |
| Patch MLP (5 L) + SVM & BoVW @ 5th hidden feat | 0.7286 |


## Deep learning classification with CNN

* *Disclaimer: During this week we had problems with the server GPUs, as such, they gave very low results and the runs were actually slow. We finally opted to use the CPU of the server.* 

We were assigned the NASNetMobile Convolutional Neural Network. It is one of the nets with less parameters, hence, it is very deep.

<u>Best Params</u>

- 3 Hidden Layers (1024 -> 512 -> 256)
- Drop-out of 50%
- No Data Augmentation
- Batch size of 32
- Adam Optimizer

*Note that all the experiment logs are available under the directory `/ghome/group07/`.*

|          | Best Experiment                                      |  Val-Test Accuracy |
| -------- | ---------------------------------------------------- | ------------------ |
| Task 0   | Original NASNet architecture (Softmax Clf Layer)     | 84%                |
| Task 1   | NASNet + 3 Dense Hidden Layers (Full-Dataset)        | 95%                |
| Task 2   | NASNet + 3 Dense Hidden Layers (Small-Dataset)       | 88%                |
| Task 3   | NASNet + 3 Dense Hidden Layers (Data augmentation)   | 87%                |
| Task 4   | NASNet + 3 Dense Hidden Layers + Drop Out (0.5)      | 88.12%             |
| Task 5   | NASNet + 3 Dense Hidden Layers + Drop Out (Optuna)   | 88.40%             |


## CNN from scratch

* *Disclaimer: We solved the problems with GPU on the server by switching to PyTorch, using **only** the following commands the GPUs work correctly:
```bash
$ conda create --n <name> python=3.10 
$ conda activate <name>
$ pip3 install torch torchvision torchaudio
```

During this week, our most drastic feature was the creation of a new dataset based on the provided small dataset. We applied data augmentation but not in place, meaning that we keep the original 400~ images from the small training dataset and we add augmented images. The augmented small datasets are available on `Task4/MIT_small_augments`.

Also, in order to reduce the parameters, we guided ourselves with the reduction of kernel sizes.

*Note that all the experiment logs are available under the directory `/ghome/group07/`.*

<img width="500" src="https://github.com/MCV-C3/project23-24-07/blob/main/Task4/figures/ball_chart.png" alt="Ball Chart"/>

| Number of Parameters | Accuracy | Normalized distance to ideal ⭐️ |
| -------------------- | -------- | ------------------------------- |
| 141k                 | 70%      | 104.40                          |
| 62k                  | 71%      | 52.67                           |
| 54k                  | 75%      | 45.75                           |
| 30k                  | 70%      | 36.78                           |
| 21k                  | 65%      | 38.04                           |
| 10k                  | 77%      | 24.07                           |
| **10k (Fine-tuned)** | **79%**  | **22.16**                       |



