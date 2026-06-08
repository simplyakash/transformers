# 🧠 Vision Transformer (ViT) 

## 🎯 What Vision Transformer is Used For

Vision Transformer (ViT) is used for a wide range of computer vision tasks, especially those involving image understanding.

---

### 🔹 1. Image Classification

The primary use case.

- Predicts the class of an image  
- Example: cat vs dog, object categories  

---

### 🔹 2. Object Detection

- Identifies and localizes multiple objects in an image  
- Often used with extensions like DETR  

---

### 🔹 3. Image Segmentation

- Assigns a label to each pixel in an image  
- Used in medical imaging and autonomous driving  

---

### 🔹 4. Image Captioning (Multimodal)

- Generates text descriptions for images  
- Combines vision and language understanding  

---

### 🔹 5. Visual Question Answering (VQA)

- Answers questions about an image  
- Example: “What color is the car?”  

---

### 🔹 6. Feature Extraction / Backbone

- Used as a feature encoder in larger systems  
- Often replaces CNNs in modern pipelines  

---

## 💡 Key Insight

Vision Transformer is a general-purpose vision backbone, similar to how Transformers are used in NLP.

👉 It learns rich representations of images that can be adapted to many different tasks.

---

# 🧠 Vision Transformer (ViT) — Complete Explanation

Vision Transformer (ViT) is a model that applies the Transformer architecture, originally designed for natural language processing, to image data.

The core idea of ViT is to treat an image as a sequence of smaller patches, similar to how a sentence is treated as a sequence of words.

---

## 🔄 Pipeline Overview

Image → Patches → Embeddings → Transformer → Classification

---

## ⚙️ Example Configuration (ViT-Base)

* Image size: 224 × 224
* Patch size: 16 × 16
* Embedding dimension: 768 [
 0.12,
 -0.45,
 1.23,
 ...
 768 values total
]
* Transformer blocks: 12
* Attention heads: 12
* Head dimension: 64 (since 768 / 12 = 64)

---

## 🖼️ Input Representation

An input image of size 224 × 224 × 3 is divided into fixed-size patches of 16 × 16, resulting in 14 patches along each dimension and a total of 196 patches.

Each patch has dimensions 16 × 16 × 3, which when flattened becomes a vector of size 768.

These flattened patches are then linearly projected into an embedding space of dimension 768 using a learnable weight matrix.

---

## ➕ CLS Token and Positional Embedding

A special classification token, called the CLS token, is added to the sequence of patch embeddings. This token is used to aggregate information from all patches for the final prediction.

Since Transformers do not inherently capture spatial relationships, learnable positional embeddings are added to each patch embedding to encode positional information.

The final input to the transformer encoder is a sequence of 197 tokens (196 patches + 1 CLS token), each of dimension 768.

---

## 🧠 Transformer Encoder

The transformer encoder consists of multiple identical blocks, typically 12 in the ViT-Base configuration.

Each transformer block includes:

* Layer Normalization
* Multi-Head Self-Attention
* Residual connection
* Layer Normalization
* MLP (feed-forward network)
* Residual connection

---

## 🔍 Self-Attention Mechanism

In multi-head self-attention, the input embeddings are projected into query (Q), key (K), and value (V) matrices using learnable weights.

For an input of shape 197 × 768 and 12 attention heads, each head operates on a subspace of dimension 64.

Attention scores are computed using the scaled dot-product attention formula:

softmax((QKᵀ) / √dₖ)

where dₖ is the dimension of the key vectors.

These attention scores are used to compute a weighted sum of the value vectors, producing the output of each attention head.

The outputs from all heads are concatenated and passed through a final linear projection layer.

---

## 🔁 MLP Block

The MLP block consists of two linear layers with a GELU activation in between.

Typical transformation:
768 → 3072 → 768

This allows the model to learn nonlinear feature transformations.

---

## 📌 Final Representation

After passing through all transformer blocks, the output corresponding to the CLS token is extracted as the final representation of the image.

---

## 🎯 Classification Head

This representation is passed through a linear classification head to produce logits for each class.

A softmax function is applied to convert logits into probabilities.

---

## 🏋️ Training

During training, the model uses cross-entropy loss, which penalizes incorrect predictions based on the predicted probability of the true class.

For example, if the predicted probability for the correct class is 0.6, the loss is computed as:

−log(0.6)

All parameters in the model, including patch embeddings, positional embeddings, attention weights, and MLP layers, are learned through backpropagation.

The AdamW optimizer is commonly used to update the model parameters.

---

## 📊 Why ViT Needs Large Data

Unlike convolutional neural networks (CNNs), Vision Transformers do not have strong inductive biases such as locality and translation invariance.

As a result, ViTs require large-scale datasets to learn these patterns effectively.

Common datasets used to train ViTs include:

* ImageNet-21k
* JFT-300M

---

## 🧩 Learnable Components

* Patch embedding ✅
* Positional embedding ✅
* CLS token ✅
* Attention matrices (Q, K, V, O) ✅
* MLP layers ✅
* LayerNorm parameters (γ, β) ✅

Non-learnable:

* Residual connections ❌
* Softmax ❌
* GELU ❌

---

## 💡 Final Intuition

Vision Transformer processes images by converting them into a sequence of patches, treating them like tokens, and learning relationships between these patches using self-attention mechanisms.

In simple terms:

Image → Words → Transformer → Prediction


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

# 🧠 Difference Between Normalization in CNNs vs Transformers

Normalization helps:
- stabilize training
- improve gradient flow
- speed up convergence

But:

```text
CNNs and Transformers use different normalization methods
```

because:
- their architectures are very different.

---

# 📌 Main Difference

| Architecture | Common Normalization |
|---|---|
| CNNs | Batch Normalization (BatchNorm) |
| Transformers | Layer Normalization (LayerNorm) |

---

# 🧠 Why Different?

CNNs process:
- spatial image features

Transformers process:
- sequential/token embeddings

Thus normalization dimensions differ.

---

# 🥇 1️⃣ Batch Normalization (CNNs)

Used heavily in:
- CNNs
- ResNet
- YOLO
- EfficientNet

---

# 📐 BatchNorm Formula

```text
x_normalized = (x - μ_batch) / sqrt(σ_batch² + ε)
```

Then:

```text
y = γx_normalized + β
```

Where:

| Symbol | Meaning |
|---|---|
| μ_batch | Batch mean |
| σ_batch² | Batch variance |
| γ | Learnable scale |
| β | Learnable shift |

---

# 📌 How BatchNorm Works

Normalization happens:

```text
across batch dimension
```

for each feature channel.

---

# 🏗️ CNN Tensor Shape

```text
(B, C, H, W)
```

Where:

| Symbol | Meaning |
|---|---|
| B | Batch size |
| C | Channels |
| H | Height |
| W | Width |

---

# 📌 BatchNorm Computes

For each channel:

```text
mean and variance across:
(B × H × W)
```

---

# 🧠 Why It Works Well for CNNs

CNNs usually use:
- large batch sizes
- stable spatial statistics

Thus:
- batch statistics become reliable.

---

# 🥈 2️⃣ Layer Normalization (Transformers)

Used in:
- Transformers
- BERT
- GPT
- ViTs
- LLMs

---

# 📐 LayerNorm Formula

```text
x_normalized = (x - μ_layer) / sqrt(σ_layer² + ε)
```

Then:

```text
y = γx_normalized + β
```

---

# 📌 How LayerNorm Works

Normalization happens:

```text
within each token embedding
```

NOT across batch.

---

# 🏗️ Transformer Tensor Shape

```text
(B, Seq_Length, Embedding_Dim)
```

Example:

```text
(32, 128, 768)
```

---

# 📌 LayerNorm Computes

For each token:

```text
mean and variance across embedding dimensions
```

Example:

```text
768 embedding values normalized together
```

---

# 🧠 Why Transformers Use LayerNorm

Transformers often:
- use varying sequence lengths
- use autoregressive generation
- use small batch sizes

Batch statistics become unstable.

LayerNorm avoids this issue.

---

# 📊 Key Difference Table

| Property | BatchNorm | LayerNorm |
|---|---|---|
| Used In | CNNs | Transformers |
| Normalization Across | Batch | Features |
| Depends on Batch Size | Yes | No |
| Stable for Small Batch? | No | Yes |
| Works in Inference Alone? | Uses running stats | Yes |
| Good for Sequential Models? | Poor | Excellent |

---

# 🏗️ Visualization

---

# CNN + BatchNorm

```text
Across multiple images
```

```text
Image1 Channel1
Image2 Channel1
Image3 Channel1
        ↓
Normalize Together
```

---

# Transformer + LayerNorm

```text
Inside one token embedding
```

```text
[0.2, -1.1, 0.5, ...]
         ↓
Normalize Internally
```

---

# 📌 Why BatchNorm Fails in Transformers

Transformers often train with:
- variable sequence lengths
- tiny batches
- autoregressive decoding

Batch statistics become:
- noisy
- unstable

Thus LayerNorm works better.

---

# 📌 Pre-LN vs Post-LN Transformers

Modern transformers often use:

| Type | Description |
|---|---|
| Pre-LN | LayerNorm before attention/FFN |
| Post-LN | LayerNorm after residual |

Most modern LLMs use:
- Pre-LN

because:
- better gradient stability

---

# 📌 RMSNorm (Modern LLMs)

Recent models use:

```text
RMSNorm
```

instead of LayerNorm.

Used in:
- LLaMA
- Mistral

Reason:
- computationally simpler
- faster training

---

# 📐 RMSNorm Idea

Uses:
- root mean square normalization

without:
- subtracting mean

---

# 📊 CNN vs Transformer Pipeline

---

# CNN

```text
Conv
 ↓
BatchNorm
 ↓
ReLU
```

---

# Transformer

```text
Attention
 ↓
Add
 ↓
LayerNorm
```

---

# 🚘 Real Model Examples

| Model | Normalization |
|---|---|
| ResNet | BatchNorm |
| YOLOv5 | BatchNorm |
| EfficientNet | BatchNorm |
| BERT | LayerNorm |
| GPT | LayerNorm |
| LLaMA | RMSNorm |

---

# 🎤 Interview-Friendly Explanation

> “CNNs typically use Batch Normalization, which normalizes activations across the batch dimension for each feature channel. Transformers instead use Layer Normalization, which normalizes across embedding dimensions within each token independently. LayerNorm works better for sequential architectures because it does not depend on batch statistics and remains stable for variable sequence lengths and small batch sizes.”



# 🧠 RMSNorm Formula

RMSNorm stands for:

```text
Root Mean Square Normalization
```

Used in modern LLMs like:
- LLaMA
- Mistral
- Gemma

---

# 📐 RMSNorm Formula

```text
RMSNorm(x) = x / RMS(x) × γ
```

Where:

```text
RMS(x) = sqrt((1/n) × Σ(xᵢ²) + ε)
```

---

# 📌 Full Expanded Formula

```text
RMSNorm(x) = x / sqrt((1/n) × Σ(xᵢ²) + ε) × γ
```

---

# 📌 Symbols Meaning

| Symbol | Meaning |
|---|---|
| x | Input vector |
| n | Number of features |
| Σ | Summation |
| ε | Small stability constant |
| γ | Learnable scaling parameter |

---

# 🧠 Key Difference from LayerNorm

---

# LayerNorm

Uses:

```text
(x - mean) / std
```

It:
- subtracts mean
- divides by standard deviation

---

# RMSNorm

Uses only:

```text
root mean square
```

It:
- DOES NOT subtract mean
- only scales magnitude

---

# 📊 RMSNorm vs LayerNorm

| Property | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | Yes | No |
| Variance normalization | Yes | Partial |
| Computational cost | Higher | Lower |
| Faster | No | Yes |
| Used in modern LLMs | Sometimes | Very common |

---

# 🏗️ RMS Calculation Example

Suppose:

```text
x = [2, 4]
```

---

# Step 1️⃣ Square Values

```text
[4, 16]
```

---

# Step 2️⃣ Mean of Squares

```text
(4 + 16) / 2 = 10
```

---

# Step 3️⃣ Square Root

```text
sqrt(10) ≈ 3.16
```

---

# Step 4️⃣ Normalize

```text
[2/3.16, 4/3.16]
≈ [0.63, 1.26]
```

---

# 📌 Why RMSNorm is Faster

LayerNorm computes:
- mean
- variance
- subtraction
- normalization

RMSNorm removes:
- mean subtraction

Thus:
- fewer operations
- faster training/inference

---

# 📌 Why Modern LLMs Prefer RMSNorm

Large models require:
- memory efficiency
- faster computation
- stable training

RMSNorm provides:
- similar performance
- lower computational overhead

---

# 📌 Used In

| Model | Normalization |
|---|---|
| GPT-2 | LayerNorm |
| BERT | LayerNorm |
| LLaMA | RMSNorm |
| Mistral | RMSNorm |
| Gemma | RMSNorm |

---

# 🏗️ Transformer Block with RMSNorm

```text
Input
 ↓
RMSNorm
 ↓
Attention
 ↓
Residual Add
 ↓
RMSNorm
 ↓
Feed Forward Network
```

---

# 📌 Important Intuition

RMSNorm mainly stabilizes:

```text
vector magnitude
```

instead of:
- centering distribution around zero

---

# 📌 Why Mean Subtraction May Not Be Necessary

In transformers:
- residual connections already stabilize activations
- strict centering often unnecessary

Thus RMSNorm works surprisingly well.

---

# 🎤 Interview-Friendly Explanation

> “RMSNorm, or Root Mean Square Normalization, normalizes activations using only the root mean square of the input without subtracting the mean. Compared to LayerNorm, it is computationally simpler and faster, which is why many modern LLMs such as LLaMA and Mistral use RMSNorm instead of LayerNorm.”


































