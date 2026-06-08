## 🧩 2. Patch Extraction

Input Image:

```text
224 × 224 × 3
```

Patch Size:

```text
16 × 16
```

Divide image into:

```text
224 / 16 = 14 patches per side
```

Grid:

```text
14 × 14
```

Visualization:

```text
┌──┬──┬──┬──┐
│■■│■■│■■│■■│
├──┼──┼──┼──┤
│■■│■■│■■│■■│
├──┼──┼──┼──┤
│■■│■■│■■│■■│
└──┴──┴──┴──┘
```

Total Number of Patches:

```text
14 × 14 = 196
```

At this stage:

```text
Context Length = 196 Tokens
```

Each patch will become one token.

---

## 📦 3. Flatten + Patch Embedding

Each patch size:

```text
16 × 16 × 3
```

Flatten:

```text
16 × 16 × 3 = 768
```

Patch Representation:

```text
[16×16×3]
        ↓
      [768]
```

Stack all patches:

```text
196 patches
      ↓
(196 × 768)
```

Apply Linear Projection:

```text
(196 × 768)
      ↓
Linear Layer
      ↓
(196 × 768)
```

Output:

```text
196 Tokens
Embedding Dimension = 768
```

Shape:

```text
Sequence Length × Embedding Dimension

(196 × 768)
```

---

## ➕ 4. CLS Token + Positional Encoding

Before:

```text
(196 × 768)
```

Add Learnable CLS Token:

```text
[CLS]
```

Shape becomes:

```text
(197 × 768)
```

Now:

```text
Context Length = 197
```

where:

```text
196 Patch Tokens
+
1 CLS Token
=
197 Tokens
```

Add Positional Embeddings:

```text
(197 × 768)
+
(197 × 768)
```

Output:

```text
(197 × 768)
```

---

## 🧠 5. Transformer Encoder Block

Input:

```text
X = (197 × 768)
```

```text
    Input X
       │
 ┌─────▼─────┐
 │ LayerNorm │
 └─────┬─────┘
       │
       ▼
 ┌───────────────┐
 │ Multi-Head    │
 │ Attention     │
 └─────┬─────────┘
       │
       ▼
 Add Residual
       │
       ▼
 ┌─────▼─────┐
 │ LayerNorm │
 └─────┬─────┘
       │
       ▼
 ┌───────────────┐
 │ MLP / FFN     │
 └─────┬─────────┘
       │
       ▼
 Add Residual
       │
       ▼
    Output
```

Output Shape:

```text
(197 × 768)
```

Shape remains unchanged throughout the encoder.

---

## 🔍 6. Multi-Head Self-Attention

Input:

```text
X = (197 × 768)
```

Context Length:

```text
197 Tokens
```

Generate:

```text
Q = XWq
K = XWk
V = XWv
```

Shapes:

```text
Q = (197 × 768)
K = (197 × 768)
V = (197 × 768)
```

---

### Split Into 12 Heads

Embedding Dimension:

```text
768
```

Number of Heads:

```text
12
```

Head Dimension:

```text
768 / 12 = 64
```

Reshape:

```text
(197 × 768)

      ↓

(197 × 12 × 64)
```

Per Head:

```text
Q = (197 × 64)
K = (197 × 64)
V = (197 × 64)
```

---

### Attention Computation

```text
Q (197×64)

      │

      ▼

Kᵀ (64×197)

      │

      ▼

QKᵀ
```

Output:

```text
(197 × 197)
```

Important:

```text
197 = Context Length
```

Therefore:

```text
Attention Matrix Shape

(Context Length × Context Length)

(197 × 197)
```

Every token attends to every other token.

---

### Scale

```text
(QKᵀ) / √64
```

---

### Softmax

```text
Softmax(QKᵀ / √64)

Shape: (197 × 197)
```

---

### Multiply By Values

```text
(197 × 197)

      ×

(197 × 64)

      ↓

(197 × 64)
```

Output for One Head:

```text
(197 × 64)
```

---

### Concatenate All Heads

```text
12 Heads

12 × 64 = 768
```

Output:

```text
(197 × 768)
```

---

### Final Projection

```text
(197 × 768)
      ↓
Linear Layer
      ↓
(197 × 768)
```

Final Attention Output:

```text
(197 × 768)
```

---

## 🎯 Key Concept: Context Length in ViT

For ViT-Base:

```text
Image Size = 224 × 224
Patch Size = 16 × 16

Number of Patches = 196

CLS Token = 1

Context Length = 197
```

This is directly analogous to LLMs:

```text
LLM:
Context Length = Number of Text Tokens

ViT:
Context Length = Number of Patch Tokens + CLS Token
```

For ViT-Base:

```text
Context Length = 197
Embedding Dimension = 768
```

Therefore the Transformer processes:

```text
197 Tokens × 768 Features
```

through every encoder block.
