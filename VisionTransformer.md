Vision Transformer (ViT) — Complete Step-by-Step Explanation

This document explains the Vision Transformer (ViT) architecture step by step using an example configuration similar to ViT-Base.

1. Core Idea of Vision Transformer

Vision Transformer comes from the paper:

An Image is Worth 16x16 Words

The key idea is:

Treat image patches as tokens (words) and process them using a Transformer encoder, similar to NLP models.

Pipeline
Image
  ↓
Split into patches
  ↓
Flatten patches
  ↓
Linear embedding
  ↓
Add positional embedding
  ↓
Transformer Encoder
  ↓
Classification head
2. Example Configuration (ViT-Base)
Parameter	Value
Image size	224 × 224
Patch size	16 × 16
Embedding dimension	768
Transformer blocks	12
Attention heads	12
Head dimension	64

Because:

768 / 12 = 64
3. Example Input Image

Input image size:

224 × 224 × 3

Typical tensor format (PyTorch):

[batch, channels, height, width]

[1, 3, 224, 224]
4. Split Image into Patches

Patch size:

16 × 16

Number of patches per dimension:

224 / 16 = 14

Total patches:

14 × 14 = 196 patches

Each patch shape:

16 × 16 × 3

Values per patch:

16 × 16 × 3 = 768

Patch extraction is commonly implemented as:

Conv2D(kernel_size=16, stride=16)

Learnable parameters:

W_patch
b_patch
5. Flatten Patches

Each patch:

16 × 16 × 3

Flattened vector:

768

So the patch matrix becomes:

196 × 768

Where:

[patches, features]
6. Linear Projection (Patch Embedding)

Each patch is projected into the embedding space.

Embedding dimension:

768

Weight matrix:

768 × 768

Computation:

Patch_vector (1 × 768)
        ×
Weight (768 × 768)

Result:

1 × 768

All patches become:

196 × 768
7. Add CLS Token

A CLS token is a learnable vector used for classification.

Shape:

1 × 768

After adding CLS:

197 × 768
8. Add Positional Embeddings

Transformers do not understand spatial order, so positional embeddings are added.

Learnable positional embedding:

197 × 768

Final transformer input:

X = patch_embeddings + positional_embeddings

Shape:

197 × 768
9. Transformer Encoder

Each transformer block contains:

LayerNorm
Multi-Head Self Attention
Residual Connection
LayerNorm
MLP
Residual Connection
10. Transformer Block Structure
Input X
 │
LayerNorm
 │
Multi-Head Self Attention
 │
Add Residual (X + Attention_Output)
 │
LayerNorm
 │
MLP
 │
Add Residual
 │
Output

This is known as the Pre-LayerNorm Transformer architecture.

1. Layer Normalization

Learnable parameters:

γ (scale)
β (shift)

Formula:

x_norm = (x − μ) / sqrt(σ² + ε)
y = γ * x_norm + β

Purpose:

stabilizes training
prevents exploding activations
12. Self-Attention Example

Assume:

tokens = 197
embedding = 768
heads = 12

Head dimension:

768 / 12 = 64

Projection matrices:

Wq = 768 × 768
Wk = 768 × 768
Wv = 768 × 768

Compute:

Q = XWq → 197 × 768
K = XWk → 197 × 768
V = XWv → 197 × 768

Split into heads:

197 × 12 × 64
Attention Formula
Attention(Q, K, V) = softmax((QK^T) / √d_k) V
where 
Q = Query matrix
K = Key matrix
V = Value matrix
d_k = dimension of key vectors 

For one head:

Q = 197 × 64
K = 197 × 64

Compute attention scores:

QKᵀ → 197 × 197

Scale:

√64 = 8

Then apply softmax.

Multiply with V:

(197 × 197) × (197 × 64)

Output per head:

197 × 64

Concatenate heads:

197 × 768

Apply output projection:

Wo = 768 × 768
13. MLP Layer

The MLP block performs nonlinear transformations.

Two linear layers:

768 → 3072 → 768

Computation:

h = GELU(XW1 + b1)
output = hW2 + b2
14. Final Representation

After 12 transformer blocks:

197 × 768

The CLS token is extracted:

1 × 768
15. Classification Head

Final linear layer:

768 × num_classes

Example (10 classes):

768 × 10

Output:

1 × 10

Softmax converts logits to probabilities.

1. Training Process

Training pipeline:

Forward pass
Image → patches → embeddings → transformer → prediction

Loss function:

Cross Entropy Loss

Example:

Predicted: [0.1, 0.6, 0.3]
True label: class 2

Loss:

−log(0.6)

Backpropagation updates all parameters.

Optimizer commonly used:

AdamW
17. Why Vision Transformers Need Large Data

Unlike CNNs, ViT has very weak inductive bias.

CNN assumptions:

locality
translation invariance

ViT must learn these from data.

Therefore large datasets are required, such as:

ImageNet-21k  
JFT-300M  
18. Learnable Components Summary  
Component	Learnable  
Patch embedding	✅  
Positional embedding	✅  
CLS token	✅:w

  
Attention matrices (Q,K,V,O)	✅  
MLP layers	✅  
LayerNorm γ, β	✅  
Residual connections	❌  
Softmax	❌  
GELU activation	❌  
Final Intuition

Think of ViT like this:

Image → Words → Transformer → Language Model

Image patches become tokens, and the transformer learns relationships between them.