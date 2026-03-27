Residual connections in the full ViT pipeline

Full flow:

Image
 ↓

Patch embedding
 ↓

Add positional embedding
 ↓

Transformer Block 1
   ├─ Attention + residual
   └─ MLP + residual
 ↓

Transformer Block 2
 ↓
...
 ↓

Transformer Block L
 ↓

CLS token
 ↓

Classification head
#######################
One transformer block summary
X
 │
LN
 │
MSA
 │
+ X
 │
LN
 │
MLP
 │
+ previous
 │
Output

1️⃣ Idea of Vision Transformer

The key idea from the paper An Image is Worth 16x16 Words:
Treat image patches like words in a sentence and feed them to a transformer.
Pipeline:
Image -> Split into patches -> Flatten patches -> Linear embedding  -> Add positional embedding -> Transformer Encoder -> Classification head

2️⃣ Example Image
Suppose we have: Image size = 32 × 32  Channels = 3 (RGB)
So input tensor: 32 × 32 × 3

3️⃣ Step 1 — Split image into patches
Patch size: 16 × 16  ; Number of patches:  32/16 = 2
So total patches:  2 × 2 = 4 patches  ; Patch shapes: 16 × 16 × 3
Example patches:
Patch1
Patch2
Patch3
Patch4

4️⃣ Step 2 — Flatten patches
Each patch:  16 × 16 × 3
Flatten: 16 × 16 × 3 = 768
So each patch becomes a vector:
Patch1 → 768
Patch2 → 768
Patch3 → 768
Patch4 → 768

Matrix: 4 × 768

5️⃣ Step 3 — Linear Projection (Patch Embedding)
ViT converts patches into embedding dimension.
Suppose: embedding dimension = 128

Weight matrix:

768 × 128

Calculation:

Patch_vector (1×768) × Weight (768×128)

Result:

1 × 128

So all patches become:

4 × 128

Example:

[patch1_embedding]
[patch2_embedding]
[patch3_embedding]
[patch4_embedding]

6️⃣ Step 4 — Add CLS token

A classification token is added. CLS token → 1 × 128
Now total tokens: 5 tokens
Matrix: 5 × 128

7️⃣ Step 5 — Add positional embeddings
Transformers don't know spatial order, so we add positions.
Position embeddings: 5 × 128
Final input to transformer:
X = patch_embeddings + position_embeddings
Shape: 5 × 128

8️⃣ Step 6 — Transformer Encoder
ViT uses standard transformer blocks.

Each block:
LayerNorm
Multi-head Attention
MLP
Residual connections

9️⃣ Self Attention Calculation (Example)
Suppose: tokens = 5, embedding = 128, heads = 4

Head dimension: 128 / 4 = 32
Compute Q, K, V
Weight matrices:
Wq = 128 × 128
Wk = 128 × 128
Wv = 128 × 128
Input: X = 5 × 128
Compute:
Q = XWq → 5 × 128
K = XWk → 5 × 128
V = XWv → 5 × 128
Split into heads 5 × 4 × 32

Attention score
Formula:
Attention(Q,K,V) = softmax(QKᵀ / √d) V
For one head:
Q = 5 × 32
K = 5 × 32
Compute: QKᵀ
Result: 5 × 5

Example matrix:

[[2.1 1.3 0.9 0.4 0.7]
 [1.8 2.5 1.1 0.6 0.3]
 ...]
Divide by: √32 ≈ 5.65

Then:
softmax

This gives attention weights.

Multiply by V
(5×5) × (5×32)
Output: 5 × 32

All heads combined:
5 × 128

🔟 MLP Layer
Two linear layers:
128 → 512 → 128

Example:
h = GELU(XW1 + b1)
output = hW2 + b2

1️⃣1️⃣ Final representation
After several transformer blocks: 5 × 128
We take the CLS token: 1 × 128

1️⃣2️⃣ Classification Head

Linear layer: 128 × num_classes

Example: 128 × 10
Output: 1 × 10
Softmax:
class probabilities

1️⃣3️⃣ Training Process

Training steps:

Forward pass
image → patches → embeddings → transformer → prediction
Loss
Usually: Cross Entropy Loss

Example:
Predicted: [0.1,0.6,0.3]
True label: class 2

Loss:
−log(0.6)

Backpropagation

Gradients computed for:

patch embedding weights,attention weights,MLP weights,classifier weights

Optimizer:

AdamW

1️⃣4️⃣ Example Training Code (simplified)
images = batch["image"]
labels = batch["label"]

pred = model(images)

loss = cross_entropy(pred, labels)

loss.backward()

optimizer.step()

1️⃣5️⃣ Why ViT needs large data

Vision Transformer has very weak inductive bias.

Unlike CNNs:

CNN assumes locality
ViT learns everything from data

So it requires:

ImageNet-21k
JFT-300M
⭐ Final intuition

Think of ViT like this:

Image → words
Transformer → language model
