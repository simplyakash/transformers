# ChatGPT Architecture (Interview Notes)

> ChatGPT is built on a **decoder-only Transformer** architecture with additional training stages such as instruction tuning and human preference optimization.

---

# Overall Pipeline

```
User Prompt
      │
      ▼
Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Token Embeddings
      │
      ▼
Positional Information
      │
      ▼
┌──────────────────────────────┐
│ Transformer Decoder Block ×N │
└──────────────────────────────┘
      │
      ▼
Linear Layer
      │
      ▼
Softmax
      │
      ▼
Next Token
      │
      ▼
Append Token
      │
      ▼
Repeat...
```

---

# Example

Prompt

```
"I love AI"
```

Assume

- Batch Size = 1
- Sequence Length = 3
- Hidden Dimension = 768
- Number of Heads = 12

---

# 1. Tokenizer

```
"I"      → 101
" love"  → 456
" AI"    → 789
```

Output

```
Shape

(1,3)
```

Meaning

```
Batch = 1
Sequence = 3
```

---

# 2. Embedding Layer

Embedding Matrix

```
(Vocab Size, Hidden Size)

(50000,768)
```

Each token becomes a 768-dimensional vector.

Output

```
(1,3,768)
```

Meaning

```
1 sentence

3 tokens

768 features per token
```

---

# 3. Positional Information

Since transformers don't know token order, positional information is added.

```
Embedding

+

Position Embedding
```

Shape remains

```
(1,3,768)
```

---

# 4. Transformer Decoder Block

Each decoder block contains

```
Masked Multi-Head Attention

↓

Add & LayerNorm

↓

Feed Forward Network

↓

Add & LayerNorm
```

This block is repeated N times (e.g., dozens of layers in modern GPT models; the exact number for ChatGPT is not public).

---

# 5. Multi-Head Self-Attention

Input

```
(1,3,768)
```

Three linear layers create

```
Q
K
V
```

Each has shape

```
(1,3,768)
```

---

# 6. Split into Multiple Heads

Assume

```
Hidden Size = 768

Heads = 12
```

Head Dimension

```
768 / 12 = 64
```

Reshape

```
(1,3,768)

↓

(1,12,3,64)
```

Meaning

```
Batch = 1

12 heads

3 tokens

64 features per head
```

---

# 7. Attention Score

```
Q @ Kᵀ
```

Dimensions

```
Q

(1,12,3,64)

Kᵀ

(1,12,64,3)

↓

Output

(1,12,3,3)
```

Each token now has an attention score for every other token.

---

# 8. Scale

```
QKᵀ / √64
```

This prevents the softmax from becoming too peaked and stabilizes training.

---

# 9. Causal Mask

GPT cannot see future tokens.

```
I

↓

love

↓

AI
```

Mask

```
✓ ✗ ✗

✓ ✓ ✗

✓ ✓ ✓
```

This ensures token 1 cannot attend to tokens that come after it.

---

# 10. Softmax

Apply

```
Softmax(dim=-1)
```

Each row becomes a probability distribution over previous/current tokens.

Shape remains

```
(1,12,3,3)
```

---

# 11. Multiply by Value

```
Attention

@

V
```

Dimensions

```
(1,12,3,3)

@

(1,12,3,64)

↓

(1,12,3,64)
```

---

# 12. Merge Heads

Concatenate

```
12 heads

↓

768 dimensions
```

Shape

```
(1,12,3,64)

↓

(1,3,768)
```

---

# 13. Output Projection

Linear layer

```
(768 → 768)
```

Output

```
(1,3,768)
```

---

# 14. Residual Connection

```
Output

+

Input
```

Shape

```
(1,3,768)
```

Residual connections help preserve information and improve gradient flow.

---

# 15. LayerNorm

Normalize each token's hidden representation.

Shape

```
(1,3,768)
```

---

# 16. Feed Forward Network (MLP)

Each token is processed independently.

Typical dimensions

```
768

↓

3072

↓

768
```

Pipeline

```
Linear

↓

GELU

↓

Linear
```

Output

```
(1,3,768)
```

---

# 17. Repeat N Times

The same pattern repeats:

```
Attention

↓

LayerNorm

↓

MLP

↓

LayerNorm
```

Each block keeps the shape

```
(1,3,768)
```

---

# 18. Output Layer

Final hidden state

```
(1,3,768)
```

Projection

```
768

↓

50000
```

Output

```
(1,3,50000)
```

Each token now has a score for every vocabulary word.

---

# 19. Softmax

```
(1,3,50000)

↓

Probability Distribution
```

The last token's distribution is used to choose the next token.

---

# 20. Autoregressive Generation

Example

```
Input

"I love"

↓

Predict

"AI"

↓

Append

"I love AI"

↓

Predict Next Token

↓

Repeat
```

---

# Training Stages

```
Internet Text
      │
      ▼
Pretraining
      │
      ▼
Instruction Tuning (SFT)
      │
      ▼
Preference Optimization (using human preference data)
      │
      ▼
ChatGPT
```

---

# Important Interview Questions

### Why only a Decoder?

GPT generates text one token at a time. It doesn't require an encoder because it's not performing sequence-to-sequence tasks like machine translation.

---

### Why Masked Attention?

To prevent the model from seeing future tokens during training and generation.

---

### Why Multi-Head Attention?

Different heads can learn different relationships simultaneously, such as syntax, long-range dependencies, or coreference.

---

### Why Residual Connections?

They improve gradient flow and make training deep networks more stable.

---

### Why LayerNorm?

It stabilizes activations and accelerates training.

---

### Why Feed Forward Network?

Attention mixes information across tokens, while the MLP transforms each token's features independently, increasing model capacity.

---

### Typical Tensor Shapes

| Layer | Shape |
|--------|-------|
| Token IDs | (B, S) |
| Embeddings | (B, S, D) |
| Q, K, V | (B, S, D) |
| Split Heads | (B, H, S, D/H) |
| Attention Scores | (B, H, S, S) |
| Attention Output | (B, H, S, D/H) |
| Merge Heads | (B, S, D) |
| MLP Output | (B, S, D) |
| Vocabulary Logits | (B, S, Vocab) |

Where:

- **B** = Batch Size
- **S** = Sequence Length
- **D** = Hidden Dimension
- **H** = Number of Attention Heads
