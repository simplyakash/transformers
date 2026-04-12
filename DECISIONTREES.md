## 🧠 1. Intuition

A Decision Tree for regression splits the data into **regions** and predicts a **constant value (mean)** in each region.

👉 Think of it as a sequence of **if-else rules**

---

## 🔵 2. How It Works

### 🟢 Step-by-step

1. Choose best feature and threshold  
2. Split data into subsets  
3. Repeat recursively  
4. Assign prediction at leaf nodes  

---

## 🌿 3. Tree Structure

            [X1 < 5]
           /       \
    [X2 < 3]       Leaf: 80
     /    \

Leaf: 50 Leaf: 60


---

## 📉 4. Prediction Rule

Each leaf predicts:

$$
\hat{y} = \text{mean of target values in that region}
$$

---

## 📊 5. Split Criterion (MSE)

$$
\text{MSE} = \frac{1}{n} \sum (y_i - \bar{y})^2
$$

---

## ⚙️ 6. Optimization Objective

$$
\text{Weighted MSE} =
\frac{n_{left}}{n} \cdot \text{MSE}_{left} +
\frac{n_{right}}{n} \cdot \text{MSE}_{right}
$$

---

## 📐 7. 1D Decision Boundary (Step Function)


Input (X)

|------|-------------|
52.5 85

Prediction:

| | | |
| 52 | | 85 |
|_| |__|


👉 Piecewise constant function

---

## 📐 8. 2D Decision Boundary (Feature Space)

### 🟢 Axis-Aligned Splits

      X2
      ↑
      |
  80  |---------|---------|
      |         |         |
      |   60    |   80    |
      |         |         |
  60  |---------|---------|
      |         |
      |   50    |
      |         |
  40  |---------|--------------→ X1
          5         10

👉 Splits are **axis-aligned (vertical & horizontal lines)**

---

### 🟡 Region-wise Predictions

- Region 1: X1 < 5, X2 < 3 → 50  
- Region 2: X1 < 5, X2 ≥ 3 → 60  
- Region 3: X1 ≥ 5 → 80  

---

### 🔷 Interpretation

- Feature space is divided into **rectangular regions**  
- Each region has a **constant prediction**  
- Output surface is **piecewise constant (non-smooth)**  

---

## ⚖️ 9. Bias-Variance Behavior

| Property | Value |
|----------|------|
| Bias     | Low  |
| Variance | High |

---

## ⚠️ 10. Overfitting

- Deep trees memorize training data  
- Poor generalization  

---

## 🛠️ 11. Controlling Overfitting

### 🔹 Pre-pruning
- max_depth  
- min_samples_split  
- min_samples_leaf  

### 🔹 Post-pruning
- Cost complexity pruning  

---

## 🌲 12. Advantages

- Handles non-linearity  
- No feature scaling required  
- Works with mixed data types  
- Easy to interpret  

---

## ❌ 13. Limitations

- High variance  
- Unstable (small data changes → different tree)  
- Piecewise constant (not smooth)  

---

## 🧪 14. Example


Data:
X1: [2, 3, 7, 9]
X2: [1, 4, 2, 5]
Y : [50, 60, 80, 90]

if X1 < 5:
if X2 < 3 → 50
else → 60
else:
→ 85


---

## 🎯 15. Key Interview Insights

- Axis-aligned splits  
- Piecewise constant approximation  
- Greedy algorithm (locally optimal splits)  
- High variance model  

---

## 💥 16. 1-Min Interview Answer

Decision tree regression works by recursively splitting the feature space using axis-aligned thresholds to minimize mean squared error. Each region corresponds to a leaf node that predicts the mean of target values in that region. This results in a piecewise constant function. While trees capture non-linearity well, they have high variance and can overfit, which is why pruning or ensemble methods like Random Forest are used.



# 🌳 📊 Decision Tree Regression — Algorithms & Implementations (Python)

---

## 🧠 1. Core Algorithm Used in Practice

### 🔵 CART (Classification And Regression Trees)

👉 This is the **primary algorithm used for regression trees**

### 🟢 Key Idea
- Binary splits only  
- Greedy top-down approach  
- Minimizes Mean Squared Error (MSE)

### 🟡 Objective

$$
\text{MSE}_{split} =
\frac{n_L}{n} \cdot \text{MSE}_L +
\frac{n_R}{n} \cdot \text{MSE}_R
$$

👉 Used internally in most Python libraries

---

## 🐍 2. Python Implementations (What to Cite in Interview)

### 🔹 1. Scikit-learn — DecisionTreeRegressor

👉 Most commonly used

**Algorithm Used:** CART

**Criteria options:**
- "squared_error" (default → MSE)
- "friedman_mse"
- "absolute_error" (MAE)
- "poisson"

---

### 🔹 2. XGBoost (Single Tree Mode)

👉 Normally boosting, but can behave like a single tree

**Algorithm:** Regularized gradient boosting tree

**Split Criterion:** Gradient + Hessian based

---

### 🔹 3. LightGBM

👉 Histogram-based decision tree

**Algorithm:** Gradient-based One-Side Sampling (GOSS)

- Faster than traditional CART
- Uses binning (histograms)

---

### 🔹 4. CatBoost

👉 Handles categorical data well

**Algorithm:** Symmetric (oblivious) trees

- Same split applied at each level
- Balanced tree structure

---

## ⚙️ 3. Variants of Split Criteria

Even within CART, different objectives exist:

### 🟢 MSE (Default)
$$
\text{MSE} = \frac{1}{n} \sum (y_i - \bar{y})^2
$$

### 🟡 MAE
$$
\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

👉 Leads to median prediction instead of mean

---

### 🔵 Friedman MSE (used in boosting)
- Improves split quality for gradient boosting

---

## 🌿 4. Internal Tree Growing Strategy

All libraries follow similar pattern:

1. Start at root  
2. Try all features and thresholds  
3. Compute split score  
4. Choose best split  
5. Recurse  

👉 This is called a **greedy recursive partitioning algorithm**

---

## ⚖️ 5. Key Differences Across Libraries

| Library        | Tree Type              | Key Feature                  |
|----------------|----------------------|-----------------------------|
| Scikit-learn   | CART                 | Simple, interpretable       |
| XGBoost        | Boosted Trees        | Regularization + accuracy   |
| LightGBM       | Histogram-based      | Fast, scalable              |
| CatBoost       | Symmetric Trees      | Handles categorical data    |

---

## 🎯 6. What to Say in Interview

👉 Best concise answer:

“Most regression trees in Python use the CART algorithm, which performs greedy binary splitting to minimize mean squared error. In practice, libraries like scikit-learn implement CART directly, while advanced libraries like XGBoost, LightGBM, and CatBoost use optimized or modified versions of decision trees with additional improvements like regularization, histogram binning, or symmetric trees.”

---

## 💥 7. Pro Tip

👉 If interviewer asks “Which algorithm is used?”

Say:

- **CART (main answer)**  
- Then optionally mention:
  - Histogram-based trees (LightGBM)
  - Gradient boosting trees (XGBoost)

✔️ This shows depth beyond basics
