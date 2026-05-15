# VLMs for Video Understanding Tasks

Video understanding extends Vision-Language Models (VLMs) from single images to temporal sequences of frames.

Instead of understanding:

```text
one image
```

the model must understand:

```text
motion + events + temporal relationships
```

across many frames.

---

# 🎯 Goal of Video Understanding

A video model should understand:

- objects
- actions
- motion
- interactions
- temporal order
- events over time

Example tasks:

- action recognition
- video captioning
- temporal localization
- video QA
- video summarization
- robotics perception

---

# 📊 Difference Between Image VLM and Video VLM

| Feature | Image VLM | Video VLM |
|---|---|---|
| Input | Single image | Sequence of frames |
| Temporal reasoning | No | Yes |
| Motion understanding | No | Yes |
| Complexity | Lower | Higher |
| Memory usage | Lower | Much higher |

---

# 📌 Basic Video Pipeline

Video:

```text
Frames → Encoder → Temporal Fusion → LLM → Output
```

---

# 📌 Step 1: Frame Extraction

Video is split into frames.

Example:

```text
30 FPS video
```

may sample:

- 1 FPS
- 2 FPS
- key frames only

to reduce computation.

---

# 📌 Step 2: Visual Encoding

Each frame is encoded using a vision encoder.

Common encoders:

- ViT
- CLIP
- EVA
- SigLIP

Each frame becomes feature tokens.

---

# 📌 Step 3: Temporal Modeling

This is the most important part.

The model must connect information across time.

---

# 📊 Temporal Modeling Methods

| Method | Idea |
|---|---|
| Frame averaging | Simple pooling |
| RNN/LSTM | Sequential memory |
| Temporal CNN | Temporal convolutions |
| Transformer | Attention across frames |
| Memory modules | Long-context storage |

---

# 📌 Transformer-Based Video Understanding

Most modern systems use Transformers.

Input:

F₁, F₂, F₃, ..., Fₜ

Attention learns:

- motion
- object changes
- event progression

across frames.

---

# 📌 Temporal Attention

Self-attention can operate:

- spatially
- temporally
- jointly

---

# 📊 Spatial vs Temporal Attention

| Type | Learns |
|---|---|
| Spatial attention | Relations inside frame |
| Temporal attention | Relations across time |

---

# 📌 Common Architectures

## 1. Frame-wise Encoder + LLM

Pipeline:

```text
Frames → Vision Encoder → Temporal Pooling → LLM
```

Simple and efficient.

---

## 2. Video Transformer

Uses attention directly across video patches.

Example:

```text
Time × Height × Width patches
```

---

## 3. Two-Stream Networks

Separate:

- appearance stream
- motion stream

Often uses optical flow.

---

## 4. Memory-Augmented Models

Store long-term temporal context.

Useful for:

- long videos
- agentic systems
- robotics

---

# 📌 Video Tokens

Video tokens may represent:

- frame patches
- temporal chunks
- object tracks

---

# 📊 Example Token Flow

```text
Video
↓
Frames
↓
Patch Embeddings
↓
Temporal Transformer
↓
LLM
↓
Caption / QA / Action
```

---

# 📌 Key Challenge: Temporal Reasoning

Image models cannot answer:

```text
"What happened before the person fell?"
```

Video understanding requires:

- ordering
- causality
- motion tracking
- event memory

---

# 📌 Optical Flow

Classical motion representation.

Represents pixel movement between frames.

Often used before transformers became dominant.

---

# 📌 3D CNNs

Older approach:

- convolution across:
  - height
  - width
  - time

Example:

```text
3 × 3 × 3 kernels
```

Used in:

- C3D
- I3D

---

# 📌 Modern Video VLMs

## Examples

| Model | Specialty |
|---|---|
| Video-LLaVA | Video conversation |
| VideoChatGPT | Video QA |
| LLaVA-NeXT-Video | Strong temporal reasoning |
| GPT-4o | Native multimodal/video |
| Gemini | Long-context video |
| Flamingo | Few-shot multimodal |
| InternVideo | Large-scale video understanding |

---

# 📌 Video Captioning

Input:

```text
video clip
```

Output:

```text
"A dog runs across the field and jumps into water."
```

---

# 📌 Video Question Answering (Video QA)

Example:

Question:

```text
"What color was the car before turning left?"
```

Requires temporal memory.

---

# 📌 Temporal Action Localization

Find when an action occurs.

Example:

```text
start time → end time
```

for:

```text
"person opens door"
```

---

# 📌 Video Summarization

Generate compact summaries of long videos.

Useful for:

- surveillance
- meetings
- lectures

---

# 📌 Robotics and Embodied AI

Video VLMs help robots understand:

- motion
- manipulation
- trajectories
- human interaction

---

# 📌 Training Challenges

## ❌ Huge Compute Cost

Videos contain many frames.

---

## ❌ Long Context Length

Thousands of visual tokens.

---

## ❌ Temporal Alignment

Need frame consistency.

---

## ❌ Memory Explosion

Attention complexity grows rapidly.

---

# 📊 Attention Complexity

Transformer attention:

O(n²)

For video:

```text
tokens = frames × patches
```

becomes extremely large.

---

# 📌 Common Optimizations

| Method | Purpose |
|---|---|
| Frame sampling | Reduce frames |
| Token pruning | Reduce tokens |
| Sliding window attention | Local temporal reasoning |
| Memory compression | Reduce context |
| Hierarchical transformers | Multi-scale reasoning |

---

# 📌 Video + Audio + Text

Modern multimodal systems combine:

- video
- speech
- subtitles
- audio events
- text prompts

---

# 📌 Example End-to-End Pipeline

```text
Video
↓
Frame Sampling
↓
Vision Encoder
↓
Temporal Transformer
↓
Projection Layer
↓
LLM
↓
Text Output
```

---

# 📌 Relation to LLMs

Video VLMs often connect visual features to LLM token space using:

- projector layers
- Q-formers
- cross-attention

---

# 📊 Video Understanding Tasks

| Task | Output |
|---|---|
| Action recognition | Label |
| Video captioning | Sentence |
| Video QA | Answer |
| Temporal localization | Time interval |
| Summarization | Condensed video/text |

---

# 🧠 Core Intuition

Image understanding asks:

```text
"What is in this image?"
```

Video understanding asks:

```text
"What changes over time?"
```

---

# 🚀 One-Line Summary

Video VLMs extend image-language models with temporal reasoning mechanisms to understand motion, events, and relationships across video frames.









# Small Video Datasets for Learning Video Transformers

If you are starting with:

- Video Transformers
- VideoMAE
- TimeSformer
- ViViT
- Action Recognition
- Video Classification

these are the best beginner-friendly datasets.

---

# 🥇 Best Beginner Dataset → UCF101

UCF101

[Official UCF101 Dataset](https://www.crcv.ucf.edu/research/data-sets/ucf101/?utm_source=chatgpt.com)

---

# 📊 Why UCF101 Is Great

| Feature | Value |
|---|---|
| Videos | 13,320 |
| Classes | 101 |
| Avg video length | ~7 sec |
| Resolution | 320×240 |
| Difficulty | Beginner → Intermediate |
| Used for | Action recognition |

---

# 📌 Example Classes

- Basketball
- PushUps
- PlayingGuitar
- Typing
- Archery
- Diving
- WalkingWithDog

---

# 📌 Why Researchers Love It

UCF101 is the classic "ImageNet for video learning".

Used in papers for:

- TimeSformer
- VideoMAE
- SlowFast
- I3D
- ViViT

---

# 🥈 Smaller Alternative → HMDB51

HMDB51

[HMDB51 on Hugging Face](https://huggingface.co/datasets/CVML-TueAI/HMDB51?utm_source=chatgpt.com)

---

# 📊 HMDB51 Stats

| Feature | Value |
|---|---|
| Videos | 6,849 |
| Classes | 51 |
| Avg length | ~5 sec |
| Difficulty | Harder than UCF101 |

---

# 📌 Why HMDB51 Is Useful

Smaller dataset:

- faster experimentation
- easier debugging
- less storage

Good for:

- sanity checking
- architecture experiments

---

# 🥉 Tiny Dataset for Very Fast Experiments

## UCF11

Very small action dataset.

Good for:

- quick debugging
- learning dataloaders
- testing augmentations

---

# 🚀 Recommended Beginner Path

| Stage | Dataset |
|---|---|
| Learn pipeline | UCF11 |
| Learn transformers | HMDB51 |
| Serious experiments | UCF101 |
| Large-scale training | Kinetics |

---

# 📌 Recommended Models to Try

| Model | Difficulty |
|---|---|
| R3D-18 | Easy |
| MC3-18 | Easy |
| TimeSformer | Medium |
| ViViT | Medium |
| VideoMAE | Medium |
| InternVideo | Advanced |

---

# 📌 Simplest Pipeline

```text
Video
↓
Frame Sampling
↓
Vision Transformer
↓
Temporal Attention
↓
Classification Head
↓
Action Label
```

---

# 📌 Easy Starter Project

## Task

Action classification on UCF101.

Example:

```text
Input Video → "BasketballDunk"
```

---

# 📌 Suggested Starter Stack

| Component | Recommendation |
|---|---|
| Framework | PyTorch |
| Video Loader | torchvision |
| Model | torchvision R3D-18 |
| GPU | 8GB+ |
| Epochs | 5–10 |
| Frames/sample | 8 or 16 |

---

# 📌 Beginner-Friendly Libraries

## PyTorchVideo

[PyTorchVideo GitHub](https://github.com/facebookresearch/pytorchvideo?utm_source=chatgpt.com)

---

## Hugging Face Transformers

[Hugging Face Video Models](https://huggingface.co/models?pipeline_tag=video-classification&utm_source=chatgpt.com)

---

## MMAction2

[MMAction2 GitHub](https://github.com/open-mmlab/mmaction2?utm_source=chatgpt.com)

Very popular for research.

---

# 📌 Recommended First Experiment

Train:

- R3D-18
or
- MC3-18

on:

- HMDB51
or
- small subset of UCF101

before trying transformers.

---

# 📌 Why Video Transformers Are Hard

Video attention complexity grows fast:

```text
tokens = frames × patches
```

So training is expensive.

---

# 📌 Practical Advice

Start with:

- 8 frames
- 112×112 resolution
- batch size 2–4

Otherwise VRAM usage explodes.

---

# 📊 Typical Video Input Shape

Video tensor:

```text
(B, C, T, H, W)
```

Where:

- B = batch
- C = channels
- T = frames/time
- H = height
- W = width

Example:

```text
(4, 3, 16, 112, 112)
```

---

# 📌 What You’ll Learn

Working with these datasets teaches:

- temporal attention
- frame sampling
- video tokenization
- temporal augmentations
- 3D convolutions
- spatiotemporal transformers

---

# 🚀 My Recommendation

For learning:

1. HMDB51 first
2. Small subset of UCF101 second
3. Then TimeSformer or VideoMAE

That gives the smoothest learning curve.

| **Dataset**            | **# Videos**       | **# Classes**        | **Avg Video Length**   | **Resolution** | **Dataset Size** | **Task Type**                 | **Difficulty**          | **Best Use Case**           | **Pros**                               | **Cons**                                  |
| ---------------------- | ------------------ | -------------------- | ---------------------- | -------------- | ---------------- | ----------------------------- | ----------------------- | --------------------------- | -------------------------------------- | ----------------------------------------- |
| UCF11                  | ~1,600             | 11                   | 2–10 sec               | Low            | Very Small       | Action Recognition            | Beginner                | Learning pipelines          | Very fast training, easy debugging     | Too small for strong transformer learning |
| HMDB51                 | 6,849              | 51                   | ~5 sec                 | Low–Medium     | Small            | Action Recognition            | Beginner → Intermediate | Transformer experimentation | Faster than UCF101, manageable size    | More noisy labels                         |
| UCF101                 | 13,320             | 101                  | ~7 sec                 | 320×240        | Medium           | Action Recognition            | Intermediate            | Standard academic benchmark | Huge community usage, balanced dataset | Limited modern-scale diversity            |
| Kinetics-400           | ~240K              | 400                  | ~10 sec                | Variable       | Very Large       | Action Recognition            | Advanced                | Pretraining transformers    | Industry-standard large dataset        | Heavy compute/storage                     |
| Kinetics-600           | ~500K              | 600                  | ~10 sec                | Variable       | Very Large       | Action Recognition            | Advanced                | Large-scale research        | Better diversity than K400             | Very expensive training                   |
| Kinetics-700           | ~650K              | 700                  | ~10 sec                | Variable       | Huge             | Action Recognition            | Advanced                | SOTA benchmarking           | Massive scale                          | Requires multi-GPU setups                 |
| Something-Something V2 | ~220K              | 174                  | 2–6 sec                | Variable       | Large            | Temporal Reasoning            | Advanced                | Motion understanding        | Strong temporal dependency learning    | Harder than UCF/Kinetics                  |
| Epic-Kitchens          | ~100K clips        | 97 verbs + 300 nouns | Long egocentric videos | Variable       | Large            | Egocentric Action Recognition | Advanced                | Robotics / embodied AI      | Real-world first-person actions        | Complex annotations                       |
| AVA                    | ~430 15-min videos | 80 actions           | Long continuous videos | Variable       | Large            | Spatiotemporal Detection      | Advanced                | Action localization         | Dense annotations                      | Annotation complexity                     |
| Charades               | ~10K               | 157                  | ~30 sec                | Variable       | Medium           | Multi-action Recognition      | Intermediate → Advanced | Multi-action understanding  | Indoor realistic scenes                | Temporal overlap complexity               |
| ActivityNet            | ~20K               | 200                  | ~2 min                 | Variable       | Large            | Temporal Localization         | Advanced                | Event detection             | Long activity understanding            | Longer training times                     |
| MSR-VTT                | 10K                | Captioning-oriented  | 10–30 sec              | Variable       | Medium           | Video Captioning              | Intermediate            | Video-text alignment        | Great for multimodal tasks             | Smaller action diversity                  |
| YouCook2               | 2K long videos     | Cooking activities   | 5–10 min               | Variable       | Medium           | Procedure Understanding       | Intermediate            | Sequential reasoning        | Strong temporal structure              | Domain-specific                           |
| TVQA                   | 152K QA pairs      | TV show scenes       | 60–90 sec              | Variable       | Large            | Video Question Answering      | Advanced                | Multimodal QA               | Temporal-language reasoning            | Complex preprocessing                     |
| Ego4D                  | 3,000+ hrs         | Multiple tasks       | Very long              | Variable       | Massive          | Egocentric Understanding      | Research-level          | Embodied AI                 | Extremely rich annotations             | Huge compute requirements                 |

# 📊 Recommended Dataset by Learning Stage\

| **Stage**                | **Dataset**            | **Why**                    |
| ------------------------ | ---------------------- | -------------------------- |
| Absolute beginner        | UCF11                  | Tiny and fast              |
| Beginner                 | HMDB51                 | Small but meaningful       |
| Intermediate             | UCF101                 | Standard benchmark         |
| Temporal reasoning       | Something-Something V2 | Motion-focused             |
| Video-language learning  | MSR-VTT                | Text-video alignment       |
| Large-scale transformers | Kinetics-400           | Industry standard          |
| Robotics/embodied AI     | Ego4D / Epic-Kitchens  | First-person understanding |


# 📊 Storage + Compute Requirements

| **Dataset**            | **Approx Storage** | **GPU Need** |
| ---------------------- | ------------------ | ------------ |
| UCF11                  | <1 GB              | Very Low     |
| HMDB51                 | ~3–5 GB            | Low          |
| UCF101                 | ~7 GB              | Medium       |
| Kinetics-400           | 450+ GB            | High         |
| Something-Something V2 | ~250 GB            | High         |
| Ego4D                  | Multiple TBs       | Very High    |



# 📊 Best Dataset for Specific Goals



| **Goal**                 | **Best Dataset**    |
| ------------------------ | ------------------- |
| Learn dataloaders        | UCF11               |
| Learn video transformers | HMDB51              |
| Benchmark models         | UCF101              |
| Motion reasoning         | Something-Something |
| Video captioning         | MSR-VTT             |
| Robotics perception      | Ego4D               |
| Temporal localization    | ActivityNet         |
| Video QA                 | TVQA                |
