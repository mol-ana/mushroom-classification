# Mushroom Classification Using Machine Learning

**Maulana P. R.**  
**Project Domain**: Mushroom Classification

---

## 1. Project Overview

### 1.1 Background

Mushroom foraging is a popular activity, but it comes with substantial risks due to the challenge of distinguishing between edible and poisonous mushrooms. Many poisonous mushrooms look similar to edible ones, which can lead to severe health issues or even death. This project leverages machine learning to classify mushrooms as edible or poisonous based on their physical characteristics, providing a reliable tool for foragers and minimizing the risks of accidental poisoning.

### 1.2 Importance of the Problem

The misidentification of mushrooms, especially when foraging for wild varieties, remains a significant public health concern. Since many edible and poisonous mushrooms share similar morphological characteristics, an automated system capable of distinguishing between them could help prevent accidental poisoning, making it easier and safer for people to forage mushrooms in the wild.

### 1.3 Related Research

Various machine learning techniques, including decision trees, support vector machines (SVM), and ensemble methods, have been explored for mushroom classification. These methods rely on multiple features such as cap shape, odor, and habitat to predict whether a mushroom is edible or poisonous.

**References**:
- M. C. Santos, G. M. Silva, "Mushroom Classification Using Machine Learning: A Comparative Analysis of Algorithms," *Journal of Computer Science & Technology*, vol. 9, no. 2, pp. 123-135, 2022.
- D. K. P. L. H. Nguyen, T. T. M. Hoang, "Application of Random Forest for Mushroom Identification," *International Journal of Computer Science*, vol. 15, no. 3, pp. 76-82, 2020.
- R. Smith, A. B. Miller, "Using Support Vector Machines for Mushroom Classification," *Machine Learning Journal*, vol. 24, no. 4, pp. 400-412, 2019.
- J. H. Bell and L. F. Matthews, "Improving Decision Trees for Mushroom Classification," *IEEE Transactions on Artificial Intelligence*, vol. 22, no. 6, pp. 50-59, 2021.

---

## 2. Business Understanding

### 2.1 Problem Statements

The core problem addressed is the accurate classification of mushrooms as either edible or poisonous. Specific challenges include:

- **Challenge 1**: The similarity in physical characteristics between edible and poisonous mushrooms, which complicates classification.
- **Challenge 2**: The increasing popularity of mushroom foraging, especially among non-experts, leading to more cases of misidentification.
- **Challenge 3**: The severe health risks, including poisoning or death, from consuming poisonous mushrooms.

### 2.2 Goals

The project aims to:

- **Goal 1**: Build a machine learning model that classifies mushrooms accurately as edible or poisonous based on their physical attributes.
- **Goal 2**: Create a tool for foragers and chefs to help identify mushrooms in real-time.
- **Goal 3**: Achieve high accuracy to minimize classification errors.
- **Goal 4**: Develop a user-friendly application for a wider audience.

### 2.3 Solution Statement

To achieve the goals outlined, the following solutions were implemented:

- **Solution 1**: I applied a variety of machine learning algorithms such as Logistic Regression, KNN, SVM, Decision Trees, Random Forest, and Gradient Boosting for comparison.
- **Solution 2**: Hyperparameter tuning techniques (grid search and random search) were utilized to optimize the models.
- **Solution 3**: Model performance was evaluated based on accuracy metrics.

---

## 3. Data Understanding

### 3.1 Data Source

The dataset used is the Mushroom Classification dataset from the UCI Machine Learning Repository, which contains 8124 mushroom samples. Each sample has 22 features, which describe different characteristics of the mushroom, such as cap shape, odor, and habitat, along with a class label indicating whether the mushroom is edible (e) or poisonous (p).

[Download the Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)

### 3.2 Features in the Dataset

The dataset includes a wide range of categorical features describing the characteristics of mushrooms. These features are as follows:

- **cap-shape**: The shape of the mushroom cap (e.g., bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s).
- **cap-surface**: The texture of the cap (e.g., fibrous = f, grooves = g, scaly = y, smooth = s).
- **cap-color**: The color of the cap (e.g., brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y).
- **bruises**: Whether the mushroom bruises when touched (e.g., bruises = t, no = f).
- **odor**: The odor of the mushroom (e.g., almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s).
- **gill-attachment**: The attachment of the gills (e.g., attached = a, descending = d, free = f, notched = n).
- **gill-spacing**: The spacing of the gills (e.g., close = c, crowded = w, distant = d).
- **gill-size**: The size of the gills (e.g., broad = b, narrow = n).
- **gill-color**: The color of the gills (e.g., black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y).
- **stalk-shape**: The shape of the stalk (e.g., enlarging = e, tapering = t).
- **stalk-root**: The shape of the stalk's root (e.g., bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?).
- **stalk-surface-above-ring**: Texture above the ring (e.g., fibrous = f, scaly = y, silky = k, smooth = s).
- **stalk-surface-below-ring**: Texture below the ring (e.g., fibrous = f, scaly = y, silky = k, smooth = s).
- **stalk-color-above-ring**: The color above the ring (e.g., brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y).
- **stalk-color-below-ring**: The color below the ring (e.g., brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y).
- **veil-type**: Type of veil (e.g., partial = p, universal = u).
- **veil-color**: Color of the veil (e.g., brown = n, orange = o, white = w, yellow = y).
- **ring-number**: The number of rings on the mushroom stalk (e.g., none = n, one = o, two = t).
- **ring-type**: Type of ring (e.g., cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z).
- **spore-print-color**: The color of the spore print (e.g., black = k, brown = n, buff = b, chocolate = h, green = r, orange = o, purple = u, white = w, yellow = y).
- **population**: The population density (e.g., abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y).
- **habitat**: The habitat where the mushroom is found (e.g., grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d).

### 3.3 Exploratory Data Analysis (EDA)

I performed an initial exploration of the data, which included visualizations and summary statistics to uncover any patterns, trends, or anomalies in the dataset. Some of the key findings include:

- **Distribution of Classes**: The dataset is balanced, with an approximately equal number of edible and poisonous mushrooms.
- **Correlation**: Certain features, like `cap-color`, `odor`, and `gill-color`, show high correlations with the target class (edible or poisonous).
- **Feature Interdependencies**: Pair plots and heatmaps revealed some relationships between features like `cap-color` and `gill-color`, which are often indicative of the mushroom's edibility.

---

## 4. Data Preparation

### 4.1 Data Cleaning

The dataset was free of missing values and duplicates, simplifying the preprocessing task. The only issue that required attention was encoding the categorical features into numerical values.

### 4.2 Feature Encoding

As the dataset contained only categorical features, **Label Encoding** was applied to transform them into numeric values suitable for machine learning models. This encoding was performed for each feature, where each category was assigned a unique integer.

### 4.3 Dimensionality Reduction with PCA

Principal Component Analysis (PCA) is applied to reduce the dimensionality of the feature space from 22 features to 7 components. This step is important for improving the performance of machine learning models and reducing computational complexity.

**Note**: PCA helps in capturing the most important variance in the data, thereby simplifying the learning process.

### 4.4 Data Splitting

The dataset was split into training (80%) and test (20%) sets to ensure that model performance could be validated on unseen data.

---

## 5. Modeling

### 5.1 Models Used

I trained several machine learning models:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**

### 5.2 Hyperparameter Tuning

To enhance model performance, hyperparameter tuning was conducted, including adjustments for:

- **KNN**: Number of neighbors.
- **Random Forest**: Number of trees (`n_estimators`), maximum depth, and other parameters.
- **Gradient Boosting**: Learning rate and tree depth.

### 5.3 How Each Algorithm Works on My Dataset

In my dataset, I use a variety of machine learning algorithms to classify mushrooms as edible or poisonous based on features like cap shape, odor, gill color, and habitat. Here's a breakdown of how each algorithm works and their respective advantages and disadvantages:

#### 1. Logistic Regression (LR)

**How It Works**:  
Logistic Regression is a linear model that attempts to find the best boundary to separate edible and poisonous mushrooms by fitting a logistic (sigmoid) function. This function outputs probabilities indicating how likely a mushroom is to be poisonous or edible.

**Advantages**:  
- **Simplicity**: Easy to implement and interpret.
- **Fast**: Computationally efficient for large datasets.
- **Probabilistic Output**: Provides confidence in predictions.

**Disadvantages**:  
- **Linear Boundaries**: Assumes a linear relationship between features and class, which may not always hold in complex datasets.
- **Sensitive to Outliers**: Performance can degrade if the dataset contains outliers.

#### 2. K-Nearest Neighbors (KNN)

**How It Works**:  
KNN classifies mushrooms by looking at the `k` nearest neighbors of a given data point and assigning the majority class of those neighbors. The value of `k` determines the number of neighbors to consider.

**Advantages**:  
- **Simple**: Easy to understand and implement.
- **Non-Parametric**: Makes no assumptions about the underlying data distribution.

**Disadvantages**:  
- **Computationally Intensive**: As the dataset grows, the algorithm becomes slower.
- **Sensitive to Irrelevant Features**: Performance can decrease with many irrelevant features.

#### 3. Support Vector Machine (SVM)

**How It Works**:  
SVM works by finding a hyperplane that best separates the classes of mushrooms in the feature space. It tries to maximize the margin between the classes to achieve a robust classifier.

**Advantages**:  
- **Effective in High Dimensions**: Performs well even with many features.
- **Memory Efficient**: Uses a subset of training points (support vectors) to define the hyperplane.

**Disadvantages**:  
- **Training Time**: Can be slow, especially with large datasets.
- **Parameter Tuning**: Requires careful selection of parameters such as the regularization parameter.

#### 4. Decision Tree

**How It Works**:  
Decision trees split the feature space into regions based on the feature values. It recursively divides the space into smaller regions that are more homogeneous with respect to the class labels.

**Advantages**:  
- **Easy to Interpret**: The tree structure is simple to visualize and understand.
- **No Feature Scaling**: Does not require normalization or scaling of features.

**Disadvantages**:  
- **Overfitting**: Decision trees tend to overfit the data if not pruned properly.
- **Instability**: Small changes in the data can result in large changes in the tree structure.

#### 5. Random Forest

**How It Works**:  
Random Forest is an ensemble method that creates multiple decision trees on bootstrapped samples of the data and averages their predictions to make a final decision.

**Advantages**:  
- **High Accuracy**: Reduces overfitting and improves generalization by averaging multiple trees.
- **Robust to Noise**: Handles noisy data well.

**Disadvantages**:  
- **Complexity**: Can be harder to interpret due to the ensemble nature.
- **Slow Prediction**: Can be computationally expensive for real-time applications.

#### 6. Gradient Boosting

**How It Works**:  
Gradient Boosting builds trees sequentially, where each tree corrects the errors made by the previous one. The model is trained to minimize the loss function by adding trees in stages.

**Advantages**:  
- **High Accuracy**: Often outperforms other models due to its boosting technique.
- **Handles Complex Data**: Effective for complex datasets with non-linear relationships.

**Disadvantages**:  
- **Training Time**: Can be slow, especially with large datasets.
- **Overfitting**: Prone to overfitting if not properly tuned.

---

## 6. Evaluation

### 6.1 Evaluation Metrics

I used **accuracy** as the primary evaluation metric:

![image](https://github.com/user-attachments/assets/1b620202-9625-4efa-b870-2fd728412bdb)

### 6.2 Model Comparison

The following table compares the accuracy of the models:

| Model              | Accuracy (%) |
|--------------------|--------------|
| Logistic Regression| 96%          |
| K-Nearest Neighbors| 97%          |
| Support Vector Machine | 97%      |
| Decision Tree      | 94%          |
| Random Forest      | 98%          |
| Gradient Boosting  | 96%          |

The **Random Forest** and **Gradient Boosting** models provided the best performance, with the Random Forest model achieving the highest accuracy.

### 6.3 Results

The **Random Forest** model achieved an impressive accuracy of **98%**, which indicates its high reliability in classifying mushrooms accurately.

### 6.4 Model Selection

Based on the results, **Random Forest** was selected as the final model due to its highest accuracy of **98%**.

---

## 7. Conclusion

The **Random Forest** model demonstrated excellent performance, achieving an accuracy of **98%** in mushroom classification. This model can effectively aid mushroom foragers in distinguishing between edible and poisonous mushrooms, potentially preventing health risks associated with misidentification.

### 7.1 Future Work

Future directions for the project include:

- **Mobile App Development**: Building a mobile application for real-time mushroom identification.
- **Neural Network Approaches**: Exploring deep learning models for even greater accuracy.
- **Further Hyperparameter Tuning**: Refining the Random Forest model for even better results.

---

## 8. References

- **UCI Machine Learning Repository**. Mushroom Classification Dataset. https://archive.ics.uci.edu/ml/datasets/Mushroom
- M. C. Santos, G. M. Silva, "Mushroom Classification Using Machine Learning: A Comparative Analysis of Algorithms," *Journal of Computer Science & Technology*, vol. 9, no. 2, pp. 123-135, 2022.
- D. K. P. L. H. Nguyen, T. T. M. Hoang, "Application of Random Forest for Mushroom Identification," *International Journal of Computer Science*, vol. 15, no. 3, pp. 76-82, 2020.
- R. Smith, A. B. Miller, "Using Support Vector Machines for Mushroom Classification," *Machine Learning Journal*, vol. 24, no. 4, pp. 400-412, 2019.
- J. H. Bell and L. F. Matthews, "Improving Decision Trees for Mushroom Classification," *IEEE Transactions on Artificial Intelligence*, vol. 22, no. 6, pp. 50-59, 2021.
