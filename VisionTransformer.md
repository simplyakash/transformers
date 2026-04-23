# 🧠 Vision Transformer (ViT) — Step-by-Step Explanation

---

## 🔄 1. Full Pipeline

    ┌──────────────┐
    │   Image      │  (224 × 224 × 3)
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Split into   │
    │ 16×16 patches│
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Flatten      │
    │ patches      │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Linear       │
    │ Embedding    │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ + Positional │
    │ Embeddings   │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Transformer  │
    │ Encoder ×12  │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ CLS Token    │
    │ Extraction   │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Classification│
    └──────────────┘

    
---

## 🧩 2. Patch Extraction

Input Image: 224 × 224 × 3

Divide into grid:

14 × 14 patches
each patch = 16 × 16

┌──┬──┬──┬──┐
│■■│■■│■■│■■│
├──┼──┼──┼──┤
│■■│■■│■■│■■│
├──┼──┼──┼──┤
│■■│■■│■■│■■│
└──┴──┴──┴──┘

Total patches = 196


---

## 📦 3. Flatten + Embedding

Each patch: 16 × 16 × 3 = 768

Flatten:
[16×16×3] → [768]

Stack:
196 patches → 196 × 768

Linear Projection:
(196 × 768) → (196 × 768)


---

## ➕ 4. CLS Token + Positional Encoding

Before:
[196 × 768]

Add CLS token:
[CLS]
↓
[197 × 768]

Add positional embeddings:
[197 × 768] + [197 × 768]


---

## 🧠 5. Transformer Encoder (Single Block)

    Input X
       │
 ┌─────▼─────┐
 │ LayerNorm │
 └─────┬─────┘
       ↓
 ┌───────────────┐
 │ Multi-Head    │
 │ Attention     │
 └─────┬─────────┘
       ↓
 Add Residual
       ↓
 ┌─────▼─────┐
 │ LayerNorm │
 └─────┬─────┘
       ↓
 ┌───────────────┐
 │ MLP (FFN)     │
 └─────┬─────────┘
       ↓
 Add Residual
       ↓
    Output

    
---

## 🔍 6. Multi-Head Attention

Input: X (197 × 768)

Linear projections:
Q = XWq
K = XWk
V = XWv

Shapes:
Q, K, V → (197 × 768)

Split into heads:
→ (197 × 12 × 64)

Per head:

 Q (197×64)
      │
      ▼
 QKᵀ → (197×197)
      │
 Scale (÷√64)
      │
 Softmax
      │
      ▼
 × V (197×64)
      │
      ▼
 Output (197×64)

 Concatenate heads:
→ (197 × 768)

Final projection:
→ (197 × 768)



---

## 🔁 7. MLP Block
Input: (197 × 768)


  │
  ▼

  Linear (768 → 3072)
│
▼
GELU
│
▼
Linear (3072 → 768)
│
▼
Output (197 × 768)


---

## 🎯 8. Classification Head

Final output: (197 × 768)

Take CLS token:
→ (1 × 768)

Linear layer:
(768 × num_classes)

Example:
(768 × 10) → (1 × 10)

Softmax → probabilities


---

## ⚙️ 9. Training Flow

Image
↓
ViT Encoder
↓
Prediction
↓
Cross-Entropy Loss
↓
Backpropagation
↓
Update Weights (AdamW)


---

## 💡 Final Intuition

Image → Patches → Tokens → Transformer → Prediction



👉 Just like NLP:
- Words → tokens  
- Here: patches → tokens  

The model learns relationships between image regions.

The model learns relationships between image regions.



































