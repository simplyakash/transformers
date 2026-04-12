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
