# Loss Functions in Vision-Language Models (VLMs)

---

# 🔹 Total Loss

```text
L_total = L_contrastive + λ₁L_ce + λ₂L_match
```

Where:

- L_contrastive → Contrastive alignment loss
- L_ce → Cross-entropy (captioning) loss
- L_match → Image-text matching loss
- λ₁, λ₂ → Weighting coefficients

---

# 🔹 1. Contrastive Loss (L_contrastive)

InfoNCE Loss

```text
InfoNCE = Information Noise Contrastive Estimation
```

Used in CLIP-style models for aligning image and text embeddings.

---

# 📌 Formula

```text
L_contrastive = 1/2 (L_image + L_text)
```

---

# 📌 Image-to-Text Loss

```text
L_image = -(1/N) ∑ log( exp(sim(Iᵢ, Tᵢ)/τ) / ∑ exp(sim(Iᵢ, Tⱼ)/τ) )
```

---

# 📌 Text-to-Image Loss

```text
L_text = -(1/N) ∑ log( exp(sim(Tᵢ, Iᵢ)/τ) / ∑ exp(sim(Tᵢ, Iⱼ)/τ) )
```

---

# 📌 Where

- Iᵢ → Image embedding
- Tᵢ → Text embedding
- sim(.) → Similarity function (usually cosine similarity)
- τ → Temperature parameter
- N → Batch size

---

# 🔹 Role of Temperature (τ)

- Lower τ → harder separation
- Higher τ → smoother distribution

It controls the sharpness of the softmax distribution.

---

# 🔹 What Contrastive Loss Does

For a text embedding Tᵢ:

- Compare it with all image embeddings in the batch
- Maximize similarity with correct image Iᵢ
- Minimize similarity with incorrect images Iⱼ

Using both image→text and text→image objectives gives:

- Stronger alignment
- Symmetric learning
- Better retrieval performance

---

# 🔹 Key Point

This is NOT plain Cross-Entropy.

It is a form of Contrastive Loss, specifically:

```text
InfoNCE Loss
```

---

# 🔹 2. Cross-Entropy Loss (L_ce)

Used in captioning and generative VLMs.

---

# 📌 Formula

```text
L_ce = - ∑ yₜ log(pₜ)
```

---

# 📌 Where

- yₜ → Ground truth label at timestep t
- pₜ → Predicted probability for correct token
- T → Sequence length

---

# 🔹 Intuition

✅ Correct and confident prediction

```text
pₜ ≈ 1
log(pₜ) ≈ 0
```

➡ Low loss

---

❌ Wrong prediction

```text
pₜ ≈ 0
log(pₜ) → -∞
```

➡ High loss

---

# 🔹 Example Calculation

Assume sequence length:

```text
T = 3
```

---

# 📌 Step 1

- True label:

```text
y₁ = [1,0,0]
```

- Prediction:

```text
p₁ = [0.9,0.05,0.05]
```

Contribution:

```text
log(0.9)
```

---

# 📌 Step 2

- True label:

```text
y₂ = [0,1,0]
```

- Prediction:

```text
p₂ = [0.2,0.6,0.2]
```

Contribution:

```text
log(0.6)
```

---

# 📌 Step 3

- True label:

```text
y₃ = [0,0,1]
```

- Prediction:

```text
p₃ = [0.7,0.2,0.1]
```

Contribution:

```text
log(0.1)
```

---

# 🔹 Total Loss

```text
L_ce = - [log(0.9) + log(0.6) + log(0.1)]
```

Approximation:

```text
L_ce ≈ 2.919
```

---

# 🔹 Key Insight

Because yₜ is one-hot encoded:

- Incorrect tokens get multiplied by 0
- Only the correct token contributes to the loss

So it simplifies to:

```text
L = - ∑ log(p_correct)
```

---

# 🔹 Summary

Cross-entropy measures how “surprised” the model is about the correct answer.

- Less surprise → lower loss
- More surprise → higher loss

---

# 🔹 3. Image-Text Matching Loss (L_match)

Used for binary classification of matching image-text pairs.

---

# 📌 Formula

```text
L_match = - [ y log(p) + (1-y)log(1-p) ]
```

---

# 📌 Where

- y → Ground truth (1 = match, 0 = non-match)
- p → Predicted probability

---

# 🔹 Binary Cross-Entropy Examples

---

## ✅ Case 1: Correct match, high confidence

- y = 1
- p = 0.9

```text
L = -log(0.9) ≈ 0.105
```

➡ Low loss

---

## ⚠️ Case 2: Correct match, medium confidence

- y = 1
- p = 0.6

```text
L = -log(0.6) ≈ 0.511
```

➡ Moderate loss

---

## ❌ Case 3: Correct match, wrong prediction

- y = 1
- p = 0.1

```text
L = -log(0.1) ≈ 2.303
```

➡ Very high loss

---

## ❌ Case 4: Non-match but model predicts match

- y = 0
- p = 0.9

```text
L = -log(0.1) ≈ 2.303
```

➡ High loss

---

# 🔹 Key Insight

- y log(p) → active for true matches
- (1-y)log(1-p) → active for non-matches

The loss:

- Rewards correct confidence
- Strongly penalizes confident mistakes

---

# 🔹 Connection to Cross-Entropy

- Multi-class cross-entropy → choose one token from many
- Binary cross-entropy → match vs non-match

Same principle, simpler output space.

---

# 📊 Summary Table

| Loss Component | Type | Purpose |
|---|---|---|
| L_contrastive | Contrastive Loss | Align image and text embeddings |
| L_ce | Cross-Entropy Loss | Generate captions/text |
| L_match | Binary Cross-Entropy | Image-text matching |

---