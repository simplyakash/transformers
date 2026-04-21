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

$y_t$ → Ground truth token
$p_t$ → Predicted probability
$T$ → Sequence length

🔹 3. Image-Text Matching Loss ($L_{match}$)

Used for binary classification of matching pairs.

Formula:

$L_{match} = - \left[y \log(p) + (1 - y)\log(1 - p)\right]$

Where:

$y$ → Ground truth label (1 = match, 0 = non-match)
$p$ → Predicted probability


📊 Summary
Loss Component	Type	Purpose
$L_{contrastive}$	Contrastive Loss	Align image and text embeddings
$L_{ce}$	Cross-Entropy Loss	Generate text from images
$L_{match}$	Binary Cross Entropy	Image-text matching classification

👉 Combined together, they enable powerful multimodal understanding in VLMs.
