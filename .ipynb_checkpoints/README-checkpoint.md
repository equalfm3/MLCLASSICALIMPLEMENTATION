# Classical Machine Learning Algorithms Implementation

This project implements a wide range of classical machine learning algorithms from scratch. Each algorithm is accompanied by both a Python implementation file and an example notebook demonstrating its usage.

## Table of Contents
1. [Supervised Learning](#supervised-learning)
   - [Classification](#classification)
   - [Regression](#regression)
2. [Unsupervised Learning](#unsupervised-learning)
   - [Clustering](#clustering)
   - [Dimensionality Reduction](#dimensionality-reduction)
   - [Association Rule Learning](#association-rule-learning)
3. [Ensemble Methods](#ensemble-methods)
4. [Model Selection and Evaluation](#model-selection-and-evaluation)
5. [Optimization Algorithms](#optimization-algorithms)
6. [Anomaly Detection](#anomaly-detection)

## Supervised Learning

### Classification

1. **Logistic Regression** [[notebook]](logistic_regression_example.ipynb)
   
   Logistic regression models the probability of a binary outcome as a function of input features. It uses the logistic function to squash the output of a linear equation between 0 and 1, interpreting the result as a probability. The model is trained by maximizing the likelihood of the observed data, typically using methods like gradient descent to find the optimal parameters.

   Formula: $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x)}}$

2. **Support Vector Machine (SVM)** [[notebook]](svm_example.ipynb)
   
   SVMs find the hyperplane that best separates classes with the maximum margin. They can handle linear and non-linear classification tasks by using the kernel trick to implicitly map inputs to high-dimensional feature spaces. SVMs are particularly effective in high-dimensional spaces and are memory efficient as they use a subset of training points (support vectors) in the decision function.

   Objective: Maximize margin $\frac{2}{||w||}$ subject to $y_i(w^Tx_i + b) \geq 1$

3. **K-Nearest Neighbors (KNN)** [[notebook]](knn_example.ipynb)
   
   KNN is a non-parametric method used for classification and regression. For classification, it assigns the class most common among the k nearest neighbors of a query point. The algorithm's simplicity and effectiveness make it a popular choice, especially when there's little or no prior knowledge about the distribution of the data. However, its performance can degrade with high-dimensional data due to the "curse of dimensionality".

   Algorithm: Find K nearest neighbors, then classify based on majority vote.

4. **Naive Bayes** [[notebook]](naive_bayes_example.ipynb)
   
   Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Despite their simplicity, they often perform surprisingly well and are widely used for text classification and spam filtering. The "naive" assumption often doesn't hold in real-world scenarios, but the algorithm tends to perform well regardless.

   Formula: $P(y|x_1,...,x_n) \propto P(y) \prod_i P(x_i|y)$

5. **Decision Trees** [[notebook]](decision_trees_example.ipynb)
   
   Decision Trees are a non-parametric supervised learning method used for classification and regression. The model is a tree-like graph of decisions and their possible consequences. At each internal node, a test is performed on an attribute; each branch represents the outcome of the test, and each leaf node represents a class label or a probability distribution over the classes. Trees are simple to understand and interpret, but they can easily overfit the training data.

   Splitting criteria: Information Gain or Gini Impurity

### Regression

1. **Linear Regression** [[notebook]](linear_regression_example.ipynb)
   
   Linear regression models the relationship between a scalar dependent variable y and one or more explanatory variables X. It assumes a linear relationship between the variables and finds the best-fitting straight line through the points. The most common method is ordinary least squares, which minimizes the sum of squared residuals. While simple, linear regression is widely used and forms the basis for many other statistical methods.

   Formula: $y = X\beta + \epsilon$

2. **Ridge Regression** [[notebook]](ridge_regression_example.ipynb)
   
   Ridge regression, also known as Tikhonov regularization, is a method of regularizing linear regression to prevent overfitting. It adds a penalty term to the ordinary least squares objective, proportional to the sum of the squares of the coefficient values (L2 regularization). This encourages the coefficients to be small but doesn't force them to zero, helping to handle multicollinearity in the data.

   Objective: $\min(||y - X\beta||^2 + \lambda||\beta||_2^2)$

3. **Lasso Regression** [[notebook]](lasso_regression_example.ipynb)
   
   Lasso (Least Absolute Shrinkage and Selection Operator) regression is another regularized version of linear regression. It uses L1 regularization, adding a penalty term proportional to the absolute value of the coefficient magnitudes. This can force some coefficients to be exactly zero, effectively performing feature selection. Lasso is particularly useful when we believe only a few input variables are relevant.

   Objective: $\min(||y - X\beta||^2 + \lambda||\beta||_1)$

4. **Elastic Net** [[notebook]](elastic_net_example.ipynb)
   
   Elastic Net combines the penalties of Lasso and Ridge regression. It adds both L1 and L2 penalty terms to the ordinary least squares objective. This method is particularly useful when there are multiple features correlated with each other. Elastic Net can perform feature selection like Lasso while still maintaining the regularization properties of Ridge.

   Objective: $\min(||y - X\beta||^2 + \lambda_1||\beta||_1 + \lambda_2||\beta||_2^2)$

5. **Gaussian Process** [[notebook]](gaussian_process_example.ipynb)
   
   Gaussian Process regression is a non-parametric approach that defines a distribution over functions. It's a powerful method for non-linear regression that provides not just predictions but also uncertainty estimates. The key idea is to assume that function values at different points are random variables with a joint Gaussian distribution. The covariance between function values is given by a kernel function, which encodes our assumptions about the function's properties (smoothness, periodicity, etc.).

   Predictions based on $f(x) \sim GP(m(x), k(x,x'))$

## Unsupervised Learning

### Clustering

1. **K-Means** [[notebook]](kmeans_example.ipynb)
   
   K-Means is one of the simplest and most popular unsupervised machine learning algorithms. It partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid). The algorithm proceeds by alternating between two steps: assigning points to clusters based on the current centroids, and recomputing centroids based on the current cluster assignments. K-Means is fast and works well on many practical problems, but it assumes spherical clusters and is sensitive to the initial placement of centroids.

   Objective: $\min \sum_{i=1}^k \sum_{x \in S_i} ||x - \mu_i||^2$

2. **Hierarchical Clustering** [[notebook]](hierarchical_clustering_example.ipynb)
   
   Hierarchical clustering builds a hierarchy of clusters, represented as a tree (dendrogram). There are two main approaches: agglomerative (bottom-up) starts with each observation in its own cluster and iteratively merges the closest clusters, while divisive (top-down) starts with all observations in one cluster and recursively splits them. The result allows for exploring different numbers of clusters by cutting the dendrogram at different levels. Hierarchical clustering doesn't require specifying the number of clusters in advance, but it can be computationally expensive for large datasets.

   Methods: Agglomerative (bottom-up) or Divisive (top-down)

3. **DBSCAN** [[notebook]](dbscan_example.ipynb)
   
   DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. It groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions. DBSCAN is particularly effective when the clusters have irregular shapes and when there's noise in the data. Unlike K-Means, it doesn't require specifying the number of clusters beforehand, but it does require setting two parameters: the maximum distance between two samples for them to be considered as in the same neighborhood (ε), and the minimum number of samples in a neighborhood for a point to be considered as a core point (MinPts).

   Parameters: $\epsilon$ (neighborhood distance), MinPts (minimum points in neighborhood)

4. **Gaussian Mixture Model** [[notebook]](gaussian_mixture_example.ipynb)
   
   Gaussian Mixture Models (GMMs) are probabilistic models that assume the data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters. GMMs can be viewed as a generalization of k-means clustering that incorporates information about the covariance structure of the data. They are more flexible than k-means as they allow for elliptical clusters of different sizes and orientations. GMMs are typically fitted using the Expectation-Maximization (EM) algorithm, which iteratively estimates the model parameters and the probabilities of each point belonging to each cluster.

   Probability Density: $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$

### Dimensionality Reduction

1. **Principal Component Analysis (PCA)** [[notebook]](pca_example.ipynb)
   
   PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. The first principal component accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible. PCA is commonly used for dimensionality reduction by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible.

   Objective: Maximize variance $\text{Var}(X^Tw)$ subject to $||w|| = 1$

2. **t-SNE** [[notebook]](tsne_example.ipynb)
   
   t-Distributed Stochastic Neighbor Embedding (t-SNE) is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. It models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability. t-SNE is particularly good at creating a single map that reveals structure at many different scales, which is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds.

   Objective: Minimize KL divergence between high and low-dimensional distributions

3. **Custom UMAP** [[notebook]](custom_umap_example.ipynb)
   
   Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualization similarly to t-SNE, but also for general non-linear dimension reduction. It is founded on three assumptions about the data: The data is uniformly distributed on a Riemannian manifold; the Riemannian metric is locally constant (or can be approximated as such); and the manifold is locally connected. UMAP uses ideas from topological data analysis and manifold learning theory to construct a topological representation of the high dimensional data, then optimizes a low-dimensional graph to have a similar topological representation.

   Objective: Optimize low-dimensional representation to match high-dimensional fuzzy topological representation

### Association Rule Learning

1. **Apriori Algorithm** [[notebook]](apriori_example.ipynb)
   
   The Apriori algorithm is used for mining frequent itemsets and generating association rules from transactional databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database. The algorithm uses a "bottom up" approach, where frequent subsets are extended one item at a time (a step known as candidate generation), and groups of candidates are tested against the data. The algorithm terminates when no further successful extensions are found. Apriori uses breadth-first search and a tree structure to count candidate item sets efficiently.

   Metrics: Support, Confidence, Lift

## Ensemble Methods

1. **Random Forests** [[notebook]](random_forests_example.ipynb)
   
   Random Forests are an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests correct for decision trees' habit of overfitting to their training set. The algorithm adds an additional layer of randomness to bagging. In addition to constructing each tree using a different bootstrap sample of the data, random forests change how the classification or regression trees are constructed. In standard trees, each node is split using the best split among all variables. In a random forest, each node is split using the best among a subset of predictors randomly chosen at that node.

   Method: Combines multiple decision trees using bagging

2. **AdaBoost** [[notebook]](adaboost_example.ipynb)
   
   AdaBoost, short for Adaptive Boosting, is a boosting technique used as an ensemble method in machine learning. It works by building a strong classifier as a linear combination of weak classifiers. The algorithm starts by training a decision tree on the original dataset. The subsequent trees are trained on repeatedly modified versions of the data, where the modification consists of applying weights to each of the training samples. In each iteration, the weights of misclassified samples are increased, while the weights of correctly classified samples are decreased. This forces subsequent learners to focus on the examples that previous weak learners misclassified.

   Boosting formula: $F_T(x) = \sum_{t=1}^T \alpha_t h_t(x)$

3. **Gradient Boosting** [[notebook]](gradient_boosting_example.ipynb)
   
   Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function. The idea of gradient boosting originated in the observation that boosting can be interpreted as an optimization algorithm on a suitable cost function. In each stage, a regression tree is fit on the negative gradient of the loss function with respect to the model values at each training data point.

   Update rule: $F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$

4. **XGBoost** [[notebook]](xgboost_example.ipynb)
   
   XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting that solve many data science problems in a fast and accurate way. It supports various objective functions, including regression, classification and ranking. The system is available as an open source package. The impact of the system has been widely recognized in a number of machine learning and data mining challenges.

   Objective: $\text{Obj} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$

5. **Voting Classifier** [[notebook]](voting_example.ipynb)
   
   A Voting Classifier is an ensemble machine learning model that combines the predictions from multiple other models. It can be used for classification or regression problems and works by combining the predictions from multiple other models. There are two main types of voting classifiers: hard voting and soft voting. In hard voting, the predicted output class is the one that receives the highest majority of votes. Each classifier votes for a particular class, and the class with the highest number of votes wins. In soft voting, the output class is the prediction based on the average of probability estimates. This is only available for classifiers that can provide probability estimates for each class.

   Method: Combines multiple models through voting (hard or soft)

6. **Bagging** [[notebook]](bagging_example.ipynb) (continued)
   
   Bagging, short for Bootstrap Aggregating, is an ensemble learning technique that combines multiple models to improve prediction accuracy and reduce overfitting. The process involves creating several subsets of the original training data through random sampling with replacement (bootstrap sampling). Each subset is then used to train a separate model, typically of the same type.

    For classification tasks, the final prediction is typically made by majority voting, while for regression tasks, it's usually the average of all predictions. Bagging helps to reduce overfitting and variance in the model predictions. It's particularly effective for algorithms that have high variance (like decision trees), as it helps to reduce this variance by creating multiple models from different subsamples of the training data.

   Method: Trains multiple models on bootstrap samples of the data

## Model Selection and Evaluation

1. **Grid Search** [[notebook]](grid_search_example.ipynb)
   
   Grid Search is a tuning technique that attempts to compute the optimum values of hyperparameters. It is an exhaustive search that is performed on a manually specified subset of the hyperparameter space of a learning algorithm. The grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set. Grid search can be computationally expensive, especially if you are searching over a large hyperparameter space, as it tries every single combination of the parameter values specified.

   Method: Exhaustive search over specified parameter values

## Optimization Algorithms

1. **Gradient Descent** [[notebook]](gradient_descent_example.ipynb)
   
   Gradient Descent is a first-order iterative optimization algorithm used to find a local minimum of a differentiable function. It's commonly used in machine learning to minimize the cost function of a model. The algorithm starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the cost function. This iterative minimization is achieved by taking steps in the negative direction of the function gradient. The size of these steps is determined by the learning rate. There are several variants of gradient descent, including batch gradient descent, stochastic gradient descent, and mini-batch gradient descent, each with its own trade-offs between computation cost and convergence speed.

   Update rule: $\theta = \theta - \alpha \nabla J(\theta)$

## Anomaly Detection

1. **Isolation Forest** [[notebook]](isolation_forest_example.ipynb)
   
   Isolation Forest is an unsupervised learning algorithm for anomaly detection that works on the principle of isolating anomalies rather than profiling normal points. It's based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation. This method is highly effective for high-dimensional datasets and is one of the few anomaly detection algorithms that scale to large datasets without extensive parameter tuning. The algorithm builds an ensemble of isolation trees for the dataset, and anomalies are those instances that have short average path lengths on the isolation trees.

   Method: Anomaly score based on average path length in isolation trees

## Usage

 Each algorithm is implemented in its own Python file (e.g., `adaboost.py`) and has a corresponding example notebook (e.g., `adaboost_example.ipynb`) demonstrating its usage. To use an algorithm:

1. Import the necessary class from the corresponding Python file.
2. Create an instance of the class with desired parameters.
3. Fit the model to your training data.
4. Use the model to make predictions or transform data as needed.

    Refer to the example notebooks (linked next to each algorithm title) for detailed usage instructions for each algorithm.

## Conclusion

This project provides implementations and examples of a wide range of classical machine learning algorithms. These algorithms form the foundation of many modern machine learning applications and are crucial for understanding more advanced techniques. By studying and experimenting with these implementations, you can gain a deeper understanding of how these algorithms work, their strengths and limitations, and when to apply them to different types of problems.

Remember that while these implementations are educational, for production use, it's often better to use well-optimized libraries like scikit-learn, XGBoost, or TensorFlow, which have been extensively tested and optimized for performance and scalability.

## Contributing

Contributions to this project are welcome! If you find a bug, have a suggestion for improvement, or want to add a new algorithm implementation, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.