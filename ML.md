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