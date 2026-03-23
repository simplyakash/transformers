# transformers
simple examples for transformers
DETR (Detection Transformer) is an end-to-end object detection model that replaces traditional detection pipelines (anchors, region proposals, NMS) with a Transformer-based architecture.

It was introduced by **Nicolas Carion and team at Meta AI in the paper
End-to-End Object Detection with Transformers (2020).

1. Why DETR was introduced

Traditional detectors like:

YOLO
Faster R-CNN
SSD

require many hand-designed components:

anchor boxes
region proposal networks
NMS (Non-Max Suppression)
IoU thresholds
heuristic post-processing

DETR removes all of this.

It treats detection as a direct set prediction problem.

2. High-Level Architecture

Pipeline:

Input Image
     │
CNN Backbone (ResNet)
     │
Feature Map
     │
Transformer Encoder
     │
Transformer Decoder
     │
Object Queries
     │
Prediction Heads
     ├── Class label
     └── Bounding box

Typical backbone:

ResNet-50
3. Step-by-Step Working
Step 1 — Image → CNN backbone

Image:

3 × H × W

Backbone produces feature map:

C × H/32 × W/32

Example:

3×800×800 → ResNet → 2048×25×25
Step 2 — Flatten features

Feature map reshaped to sequence:

2048 × 25 × 25
      ↓
625 tokens

Each token = one spatial location.

Step 3 — Positional Encoding

Since transformers have no spatial awareness, positional encoding is added:

token = feature + positional_encoding

Similar idea as used in Vision Transformer.

Step 4 — Transformer Encoder

Encoder processes the sequence using self-attention.

It learns global relationships between pixels/regions.

Example:

A car pixel can attend to all other pixels in the image.

Step 5 — Object Queries

DETR introduces learned embeddings called object queries.

Example:

100 object queries

Each query asks:

"Is there an object here?"

These go into the Transformer Decoder.

Step 6 — Transformer Decoder

Decoder performs:

Cross Attention:
object_query ↔ image features

Each query tries to detect one object.

Output:

100 predictions

Each prediction contains:

Class probabilities
Bounding box (x, y, w, h)
4. Prediction Head

For each query:

Class head

Predicts:

N classes + 1 "no object"
Box head

Predicts normalized coordinates:

(cx, cy, w, h)

using an MLP.

5. Loss Function (Key Innovation)

DETR uses bipartite matching with Hungarian Algorithm.

Problem

Model predicts 100 boxes, but image may have only 3 objects.

Solution:

Match predicted boxes to ground truth one-to-one.

Cost function includes:

Classification loss
+
Bounding box L1 loss
+
GIoU loss
6. Loss Equation

Total loss:

Loss = Classification + λ1 * L1 bbox loss + λ2 * GIoU loss
7. Why DETR does NOT need NMS

Traditional detectors output many duplicate boxes → need NMS.

DETR produces unique predictions because of Hungarian matching.

Therefore:

No anchors
No NMS
No proposals

Very clean architecture.

8. Advantages

✅ End-to-end training
✅ Global context via attention
✅ No hand-crafted components
✅ Simpler pipeline

9. Limitations

❌ Slow convergence

Training may require:

300–500 epochs

❌ Poor small object detection

Because feature resolution is low.

10. Important DETR Variants
