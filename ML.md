
---

## 🔵 2. Variance

### 🟢 Meaning
Variance measures how much predictions **change with different training data**.

High variance means the model is **too sensitive** to training data.

### 🟡 Characteristics
- Overfitting  
- Very low training error  
- High test error  
- Learns noise instead of patterns  

### 🔷 Example
A deep neural network trained on a small dataset.

---

## ⚖️ 3. Bias–Variance Tradeoff

As model complexity increases:

| Model Complexity | Bias   | Variance |
|------------------|--------|----------|
| Low              | High   | Low      |
| Medium           | Balanced | Balanced |
| High             | Low    | High     |

👉 Goal: find the **optimal balance**

---

## 📉 4. Error Decomposition

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

### 🟢 Components
- **Bias²** → error due to wrong assumptions  
- **Variance** → sensitivity to data  
- **Irreducible Error** → noise in data  

---

## 🧩 5. Visual Intuition

### 🔻 Underfitting (High Bias)
- Model too simple  
- Fails to learn patterns  
- Poor training and test performance  

### ⚖️ Good Fit
- Balanced bias and variance  
- Good generalization  

### 🔺 Overfitting (High Variance)
- Model too complex  
- Memorizes training data  
- Poor test performance  

---

## 🛠️ 6. How to Reduce Bias

Increase model capacity:

- Use more complex models  
- Add more features  
- Reduce regularization  
- Use deeper networks  

---

## 🛠️ 7. How to Reduce Variance

Reduce model sensitivity:

- Increase training data  
- Apply regularization (L1 / L2)  
- Use dropout  
- Use ensemble methods (Random Forest, Bagging)  
- Data augmentation  

---

## 📊 8. Practical Examples

| Model                | Bias   | Variance |
|---------------------|--------|----------|
| Linear Regression   | High   | Low      |
| Decision Tree       | Low    | High     |
| Random Forest       | Medium | Low      |
| Neural Network      | Low    | High     |

---

## 🎯 9. Interview Explanation

Bias–variance tradeoff describes the balance between **underfitting and overfitting**.

- High bias → model too simple → underfitting  
- High variance → model too complex → overfitting  

👉 The goal is to choose a model that **minimizes total prediction error** and generalizes well.

# 🧠 📊 Regression Techniques — Complete Guide

---

## 🔵 1. What is Regression?

### 🟢 Definition
Regression is used to predict **continuous values**

### 🟡 Goal
Given input X → predict output Y

### 🔷 Example
- 🏠 Input: House size  
- 💰 Output: Price  

---

## 🔵 2. Linear Regression

### 🟢 Model
$$
y = m x + b
$$

### 🟡 Matrix Form
$$
y = X w + b
$$

### 🟡 Loss Function (MSE)
$$
\text{Loss} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

### 🟡 Gradients
$$
\frac{\partial L}{\partial w} = -\frac{2}{n} X^T (y - \hat{y})
$$

$$
\frac{\partial L}{\partial b} = -\frac{2}{n} \sum (y - \hat{y})
$$

---

## 🔵 3. Normal Equation (Closed Form)

$$
w = (X^T X)^{-1} X^T y
$$

---

## 🔵 4. Multiple Linear Regression

$$
y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

---

## 🔵 5. Polynomial Regression

### 🟢 Model
$$
y = a x^2 + b x + c
$$

### 🟡 General Form
$$
y = a_0 + a_1 x + a_2 x^2 + \dots + a_n x^n
$$

---

## 🔵 6. Ridge Regression (L2 Regularization)

### 🟢 Loss Function
$$
\text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2 + \lambda \sum w_j^2
$$

### 🟡 Closed Form
$$
w = (X^T X + \lambda I)^{-1} X^T y
$$

---

## 🔵 7. Lasso Regression (L1 Regularization)

### 🟢 Loss Function
$$
\text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2 + \lambda \sum |w_j|
$$

---

## 🔵 8. Elastic Net

$$
\text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2 
+ \lambda_1 \sum |w| 
+ \lambda_2 \sum w^2
$$

---

## 🌳 9. Decision Tree Regression

### 🟢 Prediction (Mean of Region)
$$
\hat{y} = \frac{1}{N_{\text{region}}} \sum y_i
$$

### 🟡 Split Criterion (MSE)
$$
\text{MSE} = \frac{1}{n} \sum (y_i - \bar{y})^2
$$

---

## 🌲 10. Random Forest Regression

### 🟢 Prediction (Average of Trees)
$$
\hat{y} = \frac{1}{T} \sum \hat{y}^{(t)}
$$

Where:
- $T$ = number of trees  

---

## 📏 11. Support Vector Regression (SVR)

### 🟢 Objective
$$
\min \frac{1}{2} \|w\|^2
$$

### 🟡 Constraint
$$
|y_i - (w x_i + b)| \leq \epsilon
$$

### 🟡 With Slack Variables
$$
\min \frac{1}{2} \|w\|^2 + C \sum (\xi_i + \xi_i^*)
$$

---

## 🤖 12. Neural Network Regression

### 🟢 Single Layer
$$
y = f(Wx + b)
$$

### 🟡 Multi-layer
$$
y = f_L \big( W_L \, f_{L-1} ( \dots f_1(W_1 x + b_1) \dots ) + b_L \big)
$$

---

## ⚖️ 13. Bias-Variance Decomposition

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

---

## 🔥 14. Gradient Descent

### 🟢 Update Rule
$$
w = w - \eta \frac{\partial L}{\partial w}
$$

$$
b = b - \eta \frac{\partial L}{\partial b}
$$

---

## ⚖️ 15. Bias-Variance Tradeoff

| Model           | Bias   | Variance |
|----------------|--------|----------|
| Linear         | High   | Low      |
| Polynomial     | Medium | Medium   |
| Decision Tree  | Low    | High     |
| Random Forest  | Medium | Low      |
| Neural Network | Low    | High     |

---

## 🔥 16. When to Use What

| Situation               | Best Model                     |
|------------------------|-------------------------------|
| Simple linear data     | Linear Regression             |
| Slight non-linearity   | Polynomial                    |
| Too many features      | Lasso                         |
| Overfitting issue      | Ridge                         |
| Complex patterns       | Random Forest / Neural Network|
| Small dataset          | SVR                           |

---

## 🎯 17. Key Insights

- Regression = function approximation  
- Goal = minimize prediction error  
- Regularization = control overfitting  

---

## 💥 Pro Tip

**Bias-Variance Tradeoff + Regularization = Core of Regression**




# 🧠 📊 Classification Problems — Complete Guide

---

## 🔵 1. What is Classification?

🟢 **Definition**  
Classification is used to predict **discrete labels (categories)**

🟡 **Goal**
Predict class label y ∈ {0,1,...,K}

🔷 **Examples**
- 📧 Spam Detection → {Spam, Not Spam}  
- 🖼️ Image Classification → {Cat, Dog, Car}  
- 🏥 Disease Prediction → {Positive, Negative}  

---

## 🔵 2. Types of Classification

### 🟢 Binary Classification
- Two classes  
- Example: {0,1}

### 🟡 Multi-class Classification
- More than 2 classes  
- Example: {0,1,2,...,K}

### 🔵 Multi-label Classification
- Multiple labels per sample  
- Example: {Dog, Brown, Running}

---

## 🔵 3. Logistic Regression

### 🟢 Model (Sigmoid Function)
$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b
$$

### 🟡 Decision Boundary
$$
\hat{y} \geq 0.5 \Rightarrow \text{Class 1}, \quad \hat{y} < 0.5 \Rightarrow \text{Class 0}
$$

---

## 🔵 4. Cross Entropy Loss (Binary)

$$
\text{Loss} = - \frac{1}{n} \sum \left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]
$$

---

## 🔵 5. Softmax (Multi-class)

### 🟢 Probability Distribution
$$
P(y = k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

---

## 🔵 6. Categorical Cross Entropy

$$
\text{Loss} = - \sum_{k=1}^{K} y_k \log(\hat{y}_k)
$$

---

## 🔵 7. Decision Boundary

$$
w^T x + b = 0
$$

---

## 🔵 8. Evaluation Metrics

### 🟢 Accuracy
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

### 🟡 Precision
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### 🔵 Recall
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### 🔴 F1 Score
$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## 🔵 9. Confusion Matrix

|               | Predicted Positive | Predicted Negative |
|--------------|-------------------|-------------------|
| Actual Positive | TP              | FN                |
| Actual Negative | FP              | TN                |

---

## 🔵 10. ROC Curve & AUC

### 🟢 True Positive Rate (TPR)
$$
TPR = \frac{TP}{TP + FN}
$$

### 🟡 False Positive Rate (FPR)
$$
FPR = \frac{FP}{FP + TN}
$$

---

## 🔵 11. Regularization (Classification)

### 🟢 L2 (Ridge)
$$
\text{Loss} = \text{CrossEntropy} + \lambda \sum w^2
$$

### 🟡 L1 (Lasso)
$$
\text{Loss} = \text{CrossEntropy} + \lambda \sum |w|
$$

---

## 🔵 12. Gradient Descent

$$
w = w - \eta \frac{\partial L}{\partial w}
$$

$$
b = b - \eta \frac{\partial L}{\partial b}
$$

---

## 🔵 13. Key Insights

- Classification predicts categories, not continuous values  
- Probabilities are mapped using sigmoid or softmax  
- Cross-entropy is the main loss function  
- Evaluation requires multiple metrics (not just accuracy)  

---

## 💥 Pro Tip (Interview Gold)

Classification = Probability Estimation + Decision Boundary
