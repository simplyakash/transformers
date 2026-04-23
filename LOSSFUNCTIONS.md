Loss Function in Vision-Language Models (VLMs)

🔹 Total Loss

$L_{total} = L_{contrastive} + \lambda_1 L_{ce} + \lambda_2 L_{match}$

Where:

$L_{contrastive}$ → Contrastive alignment loss,
$L_{ce}$ → Cross-entropy (captioning) loss,
$L_{match}$ → Image-text matching loss,
$\lambda_1, \lambda_2$ → Weighting coefficients

🔹 1. Contrastive Loss ($L_{contrastive}$)

**InfoNCE Loss = Information Noise Contrastive Estimation**

Used in CLIP-style models for aligning image and text embeddings.

Formula:

$L_{contrastive} = \frac{1}{2}(L_{image} + L_{text})$

Image-to-Text Loss:

$L_{image} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(sim(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(sim(I_i, T_j)/\tau)}$

Text-to-Image Loss:

$L_{text} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(sim(T_i, I_i)/\tau)}{\sum_{j=1}^{N} \exp(sim(T_i, I_j)/\tau)}$

Where:

$I_i$ → Image embedding
$T_i$ → Text embedding
$sim(\cdot)$ → Similarity function (cosine similarity)
$\tau$ → Temperature parameter
$N$ → Batch size

Role of Temperature ($\tau$)
Controls sharpness of softmax
Lower $\tau$ → harder separation
Higher $\tau$ → smoother distribution

What it does:

Takes a text $T_i$
Compares it with all images in the batch
Tries to:
Maximize similarity with the correct image $I_i$
Minimize similarity with incorrect images $I_j$

👉 Using both ensures:

Stronger alignment,
Symmetric learning,
Better retrieval performance

🔹 Short Answer
It is NOT plain Cross-Entropy
It is a form of Contrastive Loss
More specifically:
👉 InfoNCE Loss InfoNCE = Information Noise Contrastive Estimation

🔹 2. Cross-Entropy Loss ($L_{ce}$)

Used in captioning and generative VLMs.

Formula:

$L_{ce} = - \sum_{t=1}^{T} y_t \log(p_t)$

Where:

$y_t$ → Ground truth label at time step t,Usually a one-hot vector (only the correct token = 1)
$p_t$ → the model’s predicted probability for the correct token at step t
$T$ → length of the sequence (number of tokens)

### ✅ If the model is confident and correct


p_t ≈ 1 → log(p_t) ≈ 0 → low loss


---

### ❌ If the model is wrong


p_t ≈ 0 → log(p_t) → -∞ → high loss


---
### 🔢 Example Calculation (with Ground Truth)

We use the formula:

$$
L_{ce} = - \sum_{t=1}^{T} y_t \log(p_t)
$$


### 🧾 Setup

Assume sequence length T = 3, and for each step the vocabulary has 3 tokens.

#### Step 1
- True label: y₁ = [1, 0, 0]  
- Predicted: p₁ = [0.9, 0.05, 0.05]  

#### Step 2
- True label: y₂ = [0, 1, 0]  
- Predicted: p₂ = [0.2, 0.6, 0.2]  

#### Step 3
- True label: y₃ = [0, 0, 1]  
- Predicted: p₃ = [0.7, 0.2, 0.1]  

---

### 🧮 Calculation

Apply element-wise multiplication:

#### Step 1
$$
y_1 \cdot \log(p_1) = 1 \cdot \log(0.9) + 0 + 0 = \log(0.9)
$$

#### Step 2
$$
y_2 \cdot \log(p_2) = 0 + 1 \cdot \log(0.6) + 0 = \log(0.6)
$$

#### Step 3
$$
y_3 \cdot \log(p_3) = 0 + 0 + 1 \cdot \log(0.1) = \log(0.1)
$$

---

### ➕ Total Loss

$$
L_{ce} = - [\log(0.9) + \log(0.6) + \log(0.1)]
$$

$$
L_{ce} \approx - [(-0.105) + (-0.511) + (-2.303)] = 2.919
$$

---

### 💡 Key Insight

Because \( y_t \) is **one-hot**, it “selects” only the correct token’s probability:

- All incorrect tokens get multiplied by 0  
- Only the correct token contributes to the loss  

👉 That’s why it simplifies to:
L = -∑ log(p_correct)
---

### 💡 Summary

Cross-entropy loss measures how "surprised" the model is about the correct answer:

- Less surprise → lower loss  
- More surprise → higher loss 




🔹 3. Image-Text Matching Loss ($L_{match}$)

Used for binary classification of matching pairs.

Formula:

$L_{match} = - \left[y \log(p) + (1 - y)\log(1 - p)\right]$

Where:

$y$ → Ground truth label (1 = match, 0 = non-match)
$p$ → Predicted probability

---
### 🔢 Example: Binary Cross-Entropy (Matching Loss)

We use the formula:

$$
L_{match} = - \left[ y \log(p) + (1 - y)\log(1 - p) \right]
$$

---

### 🧾 Setup (same idea, but now binary)

Instead of predicting over many tokens, we now ask:

👉 “Does this pair match?” (e.g., image–text pair)

- y = 1 → correct match  
- y = 0 → incorrect match  
- p = model’s predicted probability that it’s a match  

---

### 🧪 Example Cases

#### ✅ Case 1: Correct match, high confidence
- y = 1  
- p = 0.9  

$$
L = - [1 \cdot \log(0.9) + (1 - 1)\cdot \log(1 - 0.9)]
$$

$$
L = - \log(0.9) \approx 0.105
$$

👉 Low loss (good prediction)

---

#### ⚠️ Case 2: Correct match, medium confidence
- y = 1  
- p = 0.6  

$$
L = - \log(0.6) \approx 0.511
$$

👉 Moderate loss

---

#### ❌ Case 3: Correct match, wrong prediction
- y = 1  
- p = 0.1  

$$
L = - \log(0.1) \approx 2.303
$$

👉 Very high loss

---

### 🔁 Negative Case (important!)

#### ❌ Case 4: Not a match, but model is wrong
- y = 0  
- p = 0.9  

$$
L = - [0 + 1 \cdot \log(1 - 0.9)]
$$

$$
L = - \log(0.1) \approx 2.303
$$

👉 High loss (model is confidently wrong)

---

### 💡 Key Insight

- The first term: \( y \log(p) \) → used when it's a **true match**  
- The second term: \( (1 - y)\log(1 - p) \) → used when it's **not a match**  

👉 The loss:
- Rewards correct confidence  
- Penalizes confident mistakes (very strongly)

---

### 🔗 Connection to Previous Example

- Multi-class cross-entropy → choose correct token from many  
- Binary cross-entropy → decide **match vs no match**

👉 Same principle, simpler output space (just 0 or 1)
---

📊 Summary
Loss Component	Type	Purpose
$L_{contrastive}$	Contrastive Loss	Align image and text embeddings
$L_{ce}$	Cross-Entropy Loss	Generate text from images
$L_{match}$	Binary Cross Entropy	Image-text matching classification

👉 Combined together, they enable powerful multimodal understanding in VLMs.
