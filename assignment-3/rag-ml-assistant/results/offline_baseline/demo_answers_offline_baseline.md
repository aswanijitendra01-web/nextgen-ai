# Demo: sample queries and generated answers

- Embeddings: `tfidf` / `tfidf`; store: `sparse`; chunks: 367
- Generator: `extractive` / `extractive-baseline`; k=5; refusal threshold=0.1

## Q: What is overfitting and how does it relate to model complexity?
*type: in-scope*

**Answer:** It's important to note that model complexity is intimately tied to the variation of inputs contained in your training dataset: the larger variety of data points your dataset contains, the more complex a model you can use without overfitting. [1] While real-world plots are rarely very smooth, we can still recognize some of the characteristics of overfitting and underfitting (note that because considering fewer neighbors corresponds to a more complex model, the plot is horizontally flipped relative to the illustration in Figure 2-1). [2] Let's investigate whether we can confirm the connection between model complexity and generalization that we discussed earlier. [2] In other words, using few neighbors corresponds to high model complexity (as shown on the right side of Figure 2-1), and using many neighbors corresponds to low model complexity (as shown on the left side of Figure 2-1 ). [2]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.01s
- retrieved sources:
  - [1] score 0.141 - Chapter 2. Supervised Learning, Generalization, Overfitting, and Underfitting (p. 28-29)
  - [2] score 0.100 - Chapter 2. Supervised Learning, k-Nearest Neighbors (p. 38-39)
  - [3] score 0.093 - Chapter 2. Supervised Learning, Ensembles of Decision Trees (p. 89-90)
  - [4] score 0.087 - Chapter 2. Supervised Learning, Uncertainty in Multiclass Classification (p. 126-127)
  - [5] score 0.075 - Chapter 2. Supervised Learning, Ensembles of Decision Trees (p. 90-92)

> **Evidence (top chunk excerpt):** On the other hand, if your model is too simple—say, "Everybody who owns a house buys a boat"—then you might not be able to capture all the aspects of and variability in the data, and your model will do badly even on the training set. Choosing too simple a model is called underfitting. The more complex we allow our model to be, the better we will be able to predict on the training data. However, if our model becomes t...

## Q: How does the k-nearest neighbors algorithm make a prediction?
*type: in-scope*

**Answer:** To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset—its "nearest neighbors. " k-Neighbors classification In its simplest version, the k-NN algorithm only considers exactly one nearest neighbor, which is the closest training data point to the point we want to make a prediction for. [1] This is where the name of the k-nearest neighbors algorithm comes from. [1] The prediction of the one-nearest-neighbor algorithm is the label of that point (shown by the color of the cross). [1] So, while the nearest k-neighbors algorithm is easy to understand, it is not often used in practice, due to prediction being slow and its inability to handle many features. [2]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.257 - Chapter 2. Supervised Learning, Some Sample Datasets (p. 34-36)
  - [2] score 0.242 - Chapter 2. Supervised Learning, k-Nearest Neighbors (p. 44-45)
  - [3] score 0.237 - Chapter 1. Introduction, Building Your First Model: k-Nearest Neighbors (p. 20-21)
  - [4] score 0.219 - Chapter 2. Supervised Learning, k-Nearest Neighbors (p. 39-42)
  - [5] score 0.191 - Chapter 2. Supervised Learning, k-Nearest Neighbors (p. 41-43)

> **Evidence (top chunk excerpt):** But for now, let's get to the algorithms themselves. First, we will revisit the k-nearest neighbors (k-NN) algorithm that we saw in the previous chapter. k-Nearest Neighbors The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the training datas...

## Q: What is cross-validation and why is it used?
*type: in-scope*

**Answer:** Here, we used the same cross-validation splitter in the inner and the outer loop; however, this is not necessary and you can use any combination of cross-validation strategies in the inner and outer loops. [1] GridSearchCV will use cross-validation in place of the split into a training and validation set that we used before. [2] For this reason, many people use the term cross-validation colloquially to refer to grid search with cross-validation. [2] Because grid search with cross-validation is such a commonly used method to adjust parameters, scikit-learn provides the GridSearchCV class, which implements it in the form of an estimator. [2]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.203 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 273-274)
  - [2] score 0.190 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 265-266)
  - [3] score 0.179 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 273)
  - [4] score 0.177 - Chapter 5. Model Evaluation and Improvement, Cross-Validation in scikit-learn (p. 253-254)
  - [5] score 0.166 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 272-273)

> **Evidence (top chunk excerpt):** Here, we used the same cross-validation splitter in the inner and the outer loop; however, this is not necessary and you can use any combination of cross-validation strategies in the inner and outer loops. It can be a bit tricky to understand what is happening in the single line given above, and it can be helpful to visualize it as for loops, as done in the following simplified implementation: In[35]: def nested_cv(X...

## Q: How does grid search work for tuning model parameters?
*type: in-scope*

**Answer:** Tuning the parameter grid based on the cross-validation scores is perfectly fine, and a good way to explore the importance of different parameters. [2] While running a grid search over many parameters and on large datasets can be computationally challenging, it is also embarrassingly parallel. [3] It is, however, possible to use the IPython parallel framework for parallel grid searches, if you don't mind writing the for loop over parameters as we did in "Simple Grid Search" on page 261. [3] Because grid search with cross-validation is such a commonly used method to adjust parameters, scikit-learn provides the GridSearchCV class, which implements it in the form of an estimator. [4]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.135 - Chapter 6. Algorithm Chains and Pipelines, Grid-Searching Preprocessing Steps and Model Parameters (p. 318-319)
  - [2] score 0.118 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 270-271)
  - [3] score 0.113 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 274-275)
  - [4] score 0.111 - Chapter 5. Model Evaluation and Improvement, Grid Search with Cross-Validation (p. 265-266)
  - [5] score 0.098 - Chapter 5. Model Evaluation and Improvement, Stratified k-Fold Cross-Validation and Other Strategies (p. 259-261)

> **Evidence (top chunk excerpt):** plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha']) plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree']) plt.colorbar() Figure 6-4. Heat map of mean cross-validation score as a function of the degree of the polynomial features and alpha parameter of Ridge Looking at the results produced by the cross-validation, we can see that using poly...

## Q: How does a random forest reduce overfitting?
*type: in-scope*

**Answer:** There are two ways in which the trees in a random forest are randomized: by selecting the data points used to build a tree and by selecting the features in each split test. [1] If we build many trees, all of which work well and overfit in different ways, we can reduce the amount of overfitting by averaging their results. [1] A random forest is essentially a collection of decision trees, where each tree is slightly different from the others. [1] Typically, the feature importances provided by the random forest are more reliable than the ones provided by a single tree. [2]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.172 - Chapter 2. Supervised Learning, Ensembles of Decision Trees (p. 83-84)
  - [2] score 0.143 - Chapter 2. Supervised Learning, Ensembles of Decision Trees (p. 86-88)
  - [3] score 0.142 - Chapter 2. Supervised Learning, Ensembles of Decision Trees (p. 85-86)
  - [4] score 0.142 - Chapter 2. Supervised Learning, Ensembles of Decision Trees (p. 84-85)
  - [5] score 0.116 - Chapter 5. Model Evaluation and Improvement, Metrics for Binary Classification (p. 291-292)

> **Evidence (top chunk excerpt):** max_depth, max_leaf_nodes, or min_samples_leaf—is sufficient to prevent overfitting. Decision trees have two advantages over many of the algorithms we've discussed so far: the resulting model can easily be visualized and understood by nonexperts (at least for smaller trees), and the algorithms are completely invariant to scaling of the data. As each feature is processed separately, and the possible splits of the data...

## Q: What is principal component analysis (PCA)?
*type: in-scope*

**Answer:** second principal component, and then used the class information to color the points. [1] Each row in components_ corresponds to one principal component, and they are sorted by their importance (the first principal component comes first, etc.). [1] Principal component analysis is a method that rotates the dataset in a way such that the rotated features are statistically uncorrelated. [5] One of the simplest and most widely used algorithms for all of these is principal component analysis. [5]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.316 - Chapter 3. Unsupervised Learning and Preprocessing, Principal Component Analysis (PCA) (p. 145-146)
  - [2] score 0.271 - Chapter 3. Unsupervised Learning and Preprocessing, Principal Component Analysis (PCA) (p. 140-142)
  - [3] score 0.261 - Chapter 3. Unsupervised Learning and Preprocessing, Principal Component Analysis (PCA) (p. 146-147)
  - [4] score 0.249 - Chapter 3. Unsupervised Learning and Preprocessing, Principal Component Analysis (PCA) (p. 142)
  - [5] score 0.241 - Chapter 3. Unsupervised Learning and Preprocessing, The Effect of Preprocessing on Supervised Learning (p. 139-140)

> **Evidence (top chunk excerpt):** Out[16]: Original shape: (569, 30) Reduced shape: (569, 2) We can now plot the first two principal components (Figure 3-5): In[17]: # plot first vs. second principal component, colored by class plt.figure(figsize=(8, 8)) mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target) plt.legend(cancer.target_names, loc="best") plt.gca().set_aspect("equal") plt.xlabel("First principal component") plt.ylabel("Second ...

## Q: What is the difference between precision and recall?
*type: in-scope*

**Answer:** There is a trade-off between optimizing recall and optimizing precision. [1] So, while precision and recall are very important measures, looking at only one of them will not provide you with the full picture. [1] Precision and recall are only two of many classification measures derived from TP , FP , TN, and FN. [1] On the other hand, if you find a model that predicts only the single data point it is most sure about as positive and the rest as negative, then precision will be perfect (assuming this data point is in fact positive), but recall will be very bad. [1]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.265 - Chapter 5. Model Evaluation and Improvement, Metrics for Binary Classification (p. 283-284)
  - [2] score 0.222 - Chapter 5. Model Evaluation and Improvement, Metrics for Binary Classification (p. 289)
  - [3] score 0.208 - Chapter 5. Model Evaluation and Improvement, Metrics for Binary Classification (p. 289-290)
  - [4] score 0.205 - Chapter 5. Model Evaluation and Improvement, Metrics for Binary Classification (p. 287-288)
  - [5] score 0.202 - Chapter 5. Model Evaluation and Improvement, Metrics for Binary Classification (p. 291-292)

> **Evidence (top chunk excerpt):** Other names for recall are sensitivity, hit rate, or true positive rate (TPR). There is a trade-off between optimizing recall and optimizing precision. You can trivially obtain a perfect recall if you predict all samples to belong to the positive class— there will be no false negatives, and no true negatives either. However, predicting all samples as positive will result in many false positives, and therefore the pre...

## Q: What is the bag-of-words representation of text?
*type: in-scope*

**Answer:** One of the most simple but effective and commonly used ways to represent text for machine learning is using the bag-of-words representation. [1] The bag-of-words representation is implemented in CountVectorizer, which is a transformer. [1] Note how the order of the words in the original string is completely irrelevant to the bag-of-words feature representation. [1] The bag-of-words representation is stored in a SciPy sparse matrix that only stores the entries that are nonzero (see Chapter 1). [2]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.308 - Chapter 7. Working with Text Data, Representing Text Data as a Bag of Words (p. 327-329)
  - [2] score 0.233 - Chapter 7. Working with Text Data, Representing Text Data as a Bag of Words (p. 328-329)
  - [3] score 0.224 - Chapter 7. Working with Text Data, Bag-of-Words for Movie Reviews (p. 331)
  - [4] score 0.217 - Chapter 7. Working with Text Data, Bag-of-Words for Movie Reviews (p. 331)
  - [5] score 0.187 - Chapter 7. Working with Text Data, Bag-of-Words with More Than One Word (n-Grams) (p. 339-340)

> **Evidence (top chunk excerpt):** The task we want to solve is as follows: given a review, we want to assign the label "positive" or "negative" based on the text content of the review. This is a standard binary classification task. However, the text data is not in a format that a machine learning model can handle. We need to convert the string representation of the text into a numeric representation that we can apply our machine learning algorithms t...

## Q: Who won the 2018 FIFA World Cup?
*type: OUT-OF-SCOPE (should refuse)*

**Answer:** I couldn't find this in the book.

- refused: **True** (best retrieval score 0.090 < threshold 0.100)
- lexical support: n/a
- latency: 0.0s
- retrieved sources:
  - [1] score 0.090 - Chapter 7. Working with Text Data, Example Application: Sentiment Analysis of Movie Reviews (p. 326)
  - [2] score 0.077 - Chapter 2. Supervised Learning, Generalization, Overfitting, and Underfitting (p. 26-27)
  - [3] score 0.063 - Chapter 5. Model Evaluation and Improvement, The Danger of Overfitting the Parameters and the Validation Set (p. 261-262)
  - [4] score 0.055 - Chapter 7. Working with Text Data, Example Application: Sentiment Analysis of Movie Reviews (p. 326-327)
  - [5] score 0.051 - Chapter 2. Supervised Learning, Neural Networks (Deep Learning) (p. 112-113)

> **Evidence (top chunk excerpt):** The pos folder contains all the positive reviews, each as a separate text file, and similarly for the neg folder. There is a helper function in scikit-learn to load files stored in such a folder structure, where each subfolder corresponds to a label, called load_files. We apply the load_files function first to the training data: In[3]: from sklearn.datasets import load_files reviews_train = load_files("data/aclImdb/t...

## Q: How do transformers use self-attention in large language models?
*type: OUT-OF-SCOPE (should refuse)*

**Answer:** They are also often used on very large datasets, simply because it's not feasible to train other models. [2] The fit method of all scikit-learn models returns self. [2] Linear models often perform well when the number of features is large compared to the number of samples. [2] Large values for alpha or small values for C mean simple models. [2]

- refused: **False**
- lexical support of answer in retrieved context: **1.0**
- latency: 0.0s
- retrieved sources:
  - [1] score 0.127 - Chapter 6. Algorithm Chains and Pipelines, The General Pipeline Interface (p. 312-313)
  - [2] score 0.091 - Chapter 2. Supervised Learning, Linear Models (p. 67-68)
  - [3] score 0.085 - Chapter 8. Wrapping Up, Building Your Own Estimator (p. 360-361)
  - [4] score 0.074 - Chapter 6. Algorithm Chains and Pipelines, Summary and Outlook (p. 320-321)
  - [5] score 0.073 - Chapter 4. Representing Data and Engineering Features, One-Hot-Encoding (Dummy Variables) (p. 213-214)

> **Evidence (top chunk excerpt):** For example, you could build a pipeline containing feature extraction, feature selection, scaling, and classification, for a total of four steps. Similarly, the last step could be regression or clustering instead of classification. The only requirement for estimators in a pipeline is that all but the last step need to have a transform method, so they can produce a new representation of the data that can be used in th...
