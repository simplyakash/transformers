# Bias–Variance Tradeoff

The bias–variance tradeoff explains how model complexity affects prediction error in machine learning.

A model must balance:

- **Bias** (error from wrong assumptions)
- **Variance** (error from sensitivity to training data)

The goal is to find a model that generalizes well to unseen data.

---

# Bias

## Meaning

Bias is the error caused by **oversimplified assumptions in the learning algorithm**.

High bias means the model is **too simple** and cannot capture the true relationship in data.

## Characteristics

- Model underfits the data
- High training error
- High test error
- Predictions are overly simplified

## Example

Using **linear regression** to model a highly nonlinear relationship.

```
True relation: curve
Model: straight line
```

---

# Variance

## Meaning

Variance measures how much the model output **changes when trained on different datasets**.

High variance means the model is **too sensitive to training data**.

## Characteristics

- Model overfits the data
- Very low training error
- High test error
- Learns noise instead of patterns

## Example

A very deep neural network trained on a small dataset.

---

# Bias–Variance Tradeoff

If we increase model complexity:

| Model Complexity | Bias | Variance |
|------------------|------|----------|
| Low complexity | High | Low |
| Medium complexity | Balanced | Balanced |
| High complexity | Low | High |

So we must find a **balance between bias and variance**.

---

# Error Decomposition

Total prediction error can be written as:

```
Total Error = Bias² + Variance + Irreducible Error
```

Where:

- **Bias²** → error due to wrong model assumptions  
- **Variance** → error due to sensitivity to training data  
- **Irreducible Error** → noise inherent in data

---

# Visual Intuition

```
Underfitting (High Bias)

Model too simple
Fails to learn patterns
Poor training and test performance


Good Fit

Balanced bias and variance
Good generalization


Overfitting (High Variance)

Model too complex
Memorizes training data
Poor test performance
```

---

# Methods to Reduce Bias

Increase model capacity:

- Use more complex models
- Add more features
- Reduce regularization
- Use deeper networks

---

# Methods to Reduce Variance

Reduce model sensitivity:

- Increase training data
- Apply regularization (L1 / L2)
- Use dropout
- Use ensemble methods (Random Forest, Bagging)
- Data augmentation

---

# Practical Examples

| Model | Bias | Variance |
|------|------|----------|
| Linear Regression | High | Low |
| Decision Tree | Low | High |
| Random Forest | Medium | Low |
| Deep Neural Network | Low | High |

---

# Simple Interview Explanation

Bias–variance tradeoff describes the balance between **underfitting and overfitting**.  
High bias models are too simple and underfit the data, while high variance models are too complex and overfit the training data.  
The goal is to choose a model complexity that minimizes total prediction error.

Regression tehcniques:
🧠 📊 Regression Techniques — Complete Guide (Plain Text)

🔵 1. What is Regression?

🟢 Definition
Regression is used to predict continuous values

🟡 Goal

Given input X → predict output Y

🔷 Example

🏠 Input: House size
💰 Output: Price

🔵 2. Linear Regression

🟢 Model

y = m*x + b

🟡 Matrix Form

y = X*w + b

🟡 Loss Function (Mean Squared Error)

Loss = (1/n) * sum( (y_i - y_pred_i)^2 )

🟡 Gradients

dL/dw = -(2/n) * X^T * (y - y_pred)

dL/db = -(2/n) * sum(y - y_pred)

🔵 3. Normal Equation (Closed Form Solution)
w = (X^T * X)^(-1) * X^T * y

🔵 4. Multiple Linear Regression
y = w1*x1 + w2*x2 + ... + wn*xn + b

🔵 5. Polynomial Regression

🟢 Quadratic Example

y = a*x^2 + b*x + c

🟡 General Form

y = a0 + a1*x + a2*x^2 + ... + an*x^n

🔵 6. Ridge Regression (L2 Regularization)

🟢 Loss Function

Loss = (1/n) * sum( (y - y_pred)^2 ) + lambda * sum(w_j^2)

🟡 Closed Form

w = (X^T * X + lambda * I)^(-1) * X^T * y

🔵 7. Lasso Regression (L1 Regularization)

🟢 Loss Function

Loss = (1/n) * sum( (y - y_pred)^2 ) + lambda * sum(|w_j|)

🔵 8. Elastic Net
Loss = (1/n) * sum( (y - y_pred)^2 ) 
       + lambda1 * sum(|w|) 
       + lambda2 * sum(w^2)

🌳 9. Decision Tree Regression

🟢 Prediction (mean of region)

y_pred = (1 / N_region) * sum(y_i in region)

🟡 Split Criterion (MSE)

MSE = (1/n) * sum( (y_i - y_mean)^2 )
🌲 10. Random Forest Regression

🟢 Prediction (average of trees)

y_pred = (1 / T) * sum(prediction_from_each_tree)

Where:

T = number of trees
📏 11. Support Vector Regression (SVR)

🟢 Objective

Minimize: (1/2) * ||w||^2

🟡 Constraint

|y_i - (w*x_i + b)| <= epsilon

🟡 With Slack Variables

Minimize: (1/2)*||w||^2 + C * sum(xi_i + xi_i*)

🤖 12. Neural Network Regression

🟢 Single Layer

y = f(W*x + b)

🟡 Multi-layer

y = fL( W_L * f_{L-1}( ... f1(W1*x + b1) ... ) + bL )

⚖️ 13. Bias-Variance Decomposition
Total Error = Bias^2 + Variance + Noise
🔥 14. Gradient Descent (Optimization)

🟢 Update Rule

w = w - learning_rate * (dL/dw)

b = b - learning_rate * (dL/db)

⚖️ 15. Bias-Variance Tradeoff
| Model           | Bias   | Variance |
|----------------|--------|----------|
| Linear         | High   | Low      |
| Polynomial     | Medium | Medium   |
| Decision Tree  | Low    | High     |
| Random Forest  | Medium | Low      |
| Neural Network | Low    | High     |

🔥 16. **When to Use What**
| Situation               | Best Model                     |
|------------------------|-------------------------------|
| Simple linear data     | Linear Regression             |
| Slight non-linearity   | Polynomial                    |
| Too many features      | Lasso                         |
| Overfitting issue      | Ridge                         |
| Complex patterns       | Random Forest / Neural Network|
| Small dataset          | SVR                           |

🎯 17. Key Insights

Regression = function approximation

Goal = minimize prediction error

Regularization = control overfitting

💥 Pro Tip (Interview Gold)
Bias-Variance Tradeoff + Regularization = Core of Regression
