1️⃣ Idea of Vision Transformer

The key idea from the paper An Image is Worth 16x16 Words:

Treat image patches like words in a sentence and feed them to a transformer.

Pipeline:

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
