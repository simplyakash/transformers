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
* Embedding dimension: 768
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

The model learns relationships between image regions.



































