# 🚀 VLM Interview Handbook
# Part 1 — Multi-GPU Training Fundamentals (DDP, Data Parallelism, AllReduce, NCCL & DistributedSampler)

This section covers the most frequently asked interview questions on **distributed training**, especially for Vision-Language Models (VLMs) such as **LLaVA**, **BLIP-2**, **Flamingo**, **Qwen2.5-VL**, **InternVL**, and other multimodal Transformer models.

---

# ❓ Q1. Why can't modern Vision-Language Models (VLMs) be trained on a single GPU?

## ✅ Answer

Modern VLMs combine:

- A Vision Encoder (ViT, CLIP, EVA, etc.)
- A Multimodal Projector
- A Large Language Model (7B–70B+ parameters)

During training, GPU memory is consumed by much more than just the model parameters.

GPU memory stores:

- Model Parameters
- Gradients
- Optimizer States
- Intermediate Activations
- Attention Matrices
- Temporary Buffers

For example, a 7B parameter model in FP16 requires approximately:

| Component | Approximate Memory |
|------------|-------------------:|
| Parameters | ~14 GB |
| Gradients | ~14 GB |
| Adam Optimizer States | ~28 GB |
| Activations | 10–30+ GB |

Total memory can easily exceed **60 GB**, which is larger than the memory available on a single consumer GPU.

Therefore, training must be distributed across multiple GPUs.

---

# ❓ Q2. What is Data Parallelism?

## ✅ Answer

Data Parallelism is a distributed training strategy where:

- Every GPU stores an identical copy of the model.
- Each GPU processes a different mini-batch of data.
- Each GPU computes gradients independently.
- Gradients are synchronized before updating the model.

## Workflow

```text
Dataset
   │
   ├────────────┬────────────┬────────────┬────────────
   ▼            ▼            ▼            ▼
 GPU0         GPU1         GPU2         GPU3
Batch1       Batch2       Batch3       Batch4
   │            │            │            │
Forward      Forward      Forward      Forward
   │            │            │            │
Backward     Backward     Backward     Backward
   └────────────┬────────────┬────────────┘
                ▼
      Average Gradients (AllReduce)
                │
         Optimizer Update
                │
 All GPUs now have identical weights
```

### Advantages

- Easy to implement
- Excellent GPU utilization
- Scales well for medium-sized models

### Disadvantages

- Every GPU stores the full model.
- Memory usage grows with model size.

---

# ❓ Q3. What is Distributed Data Parallel (DDP)?

## ✅ Answer

Distributed Data Parallel (DDP) is PyTorch's recommended implementation of Data Parallelism.

Unlike `torch.nn.DataParallel`, DDP launches **one process per GPU**.

Each process:

- Owns one GPU
- Has its own model replica
- Computes gradients locally
- Synchronizes gradients using NCCL AllReduce

Unlike DataParallel, GPUs communicate directly with each other instead of routing everything through GPU0.

### DDP Architecture

```text
Process 0 ─── GPU0 ─── Model Copy
Process 1 ─── GPU1 ─── Model Copy
Process 2 ─── GPU2 ─── Model Copy
Process 3 ─── GPU3 ─── Model Copy

          │
          ▼

    NCCL AllReduce

          │

Average Gradients

          │

Optimizer Step
```

### Why DDP is Faster

- Parallel communication
- No GPU0 bottleneck
- Better scaling
- Independent processes
- Lower CPU overhead

---

# ❓ Q4. Why is Distributed Data Parallel (DDP) faster than DataParallel?

## ✅ Answer

The primary reason is communication efficiency.

### DataParallel

```text
GPU0

↓

Scatter batches

↓

GPU1
GPU2
GPU3

↓

Gather outputs back to GPU0

↓

Backward

↓

GPU0 computes gradients

↓

Broadcast updated model
```

Problems:

- GPU0 becomes overloaded.
- Communication passes through GPU0.
- High synchronization overhead.
- Poor scalability.

---

### Distributed Data Parallel

```text
GPU0

Forward

Backward

↓

GPU1

Forward

Backward

↓

GPU2

Forward

Backward

↓

GPU3

Forward

Backward

↓

AllReduce

↓

Optimizer Step
```

Every GPU participates equally.

There is no master GPU.

---

# ❓ Q5. What is AllReduce?

## ✅ Answer

AllReduce is the distributed communication operation used to synchronize gradients across GPUs.

Each GPU computes gradients using its local mini-batch.

Example:

```text
GPU0 Gradient = 2

GPU1 Gradient = 6

GPU2 Gradient = 4

GPU3 Gradient = 8
```

AllReduce computes

```text
Average Gradient

= (2 + 6 + 4 + 8) / 4

= 5
```

Every GPU receives the averaged gradient.

Then every GPU performs the optimizer update.

After the update:

```text
GPU0 Weights

=

GPU1 Weights

=

GPU2 Weights

=

GPU3 Weights
```

All model replicas remain identical.

---

# ❓ Q6. Why are gradients averaged instead of summed?

## ✅ Answer

Suppose

```text
GPU0 Batch Size = 32

GPU1 Batch Size = 32

GPU2 Batch Size = 32

GPU3 Batch Size = 32
```

Effective batch size

```text
32 × 4 = 128
```

If gradients were summed,

the gradient magnitude would increase with the number of GPUs.

Training would become unstable unless the learning rate were adjusted.

Averaging gradients keeps the gradient scale approximately the same as single-GPU training.

Mathematically,

```text
Average Gradient

=

(g₁ + g₂ + ... + gₙ)

/

n
```

This allows the same learning rate to be used across different GPU counts (though large-scale training may still benefit from learning-rate scaling strategies).

---

# ❓ Q7. What is NCCL?

## ✅ Answer

NCCL stands for **NVIDIA Collective Communications Library**.

It provides highly optimized communication primitives for multi-GPU and multi-node training.

Operations supported include:

- AllReduce
- Broadcast
- Reduce
- Gather
- Scatter
- AllGather
- ReduceScatter

DDP uses NCCL as its backend to synchronize gradients efficiently.

Advantages:

- Optimized for NVIDIA GPUs
- Supports NVLink and InfiniBand
- High bandwidth
- Low latency

---

# ❓ Q8. What is a DistributedSampler?

## ✅ Answer

If every GPU reads the same training samples,

training becomes inefficient because all GPUs compute identical gradients.

A `DistributedSampler` ensures that each GPU receives a different portion of the dataset.

Without DistributedSampler

```text
GPU0

Images 1–100

GPU1

Images 1–100

GPU2

Images 1–100

GPU3

Images 1–100
```

Every GPU performs duplicate work.

---

With DistributedSampler

```text
GPU0

Images 1–100

GPU1

Images 101–200

GPU2

Images 201–300

GPU3

Images 301–400
```

Each GPU processes unique samples.

Benefits:

- No duplicated computation
- Better GPU utilization
- Faster convergence
- Correct gradient estimation

---

# ❓ Q9. What happens if DistributedSampler is not used?

## ✅ Answer

If every GPU processes the same mini-batch:

- Computation is duplicated.
- Effective batch size does not increase.
- GPUs waste compute resources.
- Training throughput decreases.
- Gradient diversity is reduced.

As a result, distributed training provides little or no speedup.

---

# ❓ Q10. Explain the complete DDP training pipeline.

## ✅ Answer

The complete training process is:

```text
Initialize Process Group

↓

Launch One Process per GPU

↓

Load Dataset

↓

DistributedSampler splits the dataset

↓

Each GPU loads a different mini-batch

↓

Forward Pass

↓

Loss Computation

↓

Backward Pass

↓

Compute Local Gradients

↓

NCCL AllReduce

↓

Average Gradients Across GPUs

↓

Optimizer Step

↓

Synchronize Updated Weights

↓

Repeat Until Training Completes
```

---

# ⭐ Interview Tips

### Common Follow-Up Questions

- Why is DDP preferred over DataParallel?
- What is gradient synchronization?
- Why are gradients averaged?
- Explain AllReduce.
- Why does every GPU need identical weights?
- What backend does PyTorch DDP use?
- Why is NCCL faster than Gloo for GPU training?
- What happens if one GPU receives more samples than others?
- Why is DistributedSampler required?
- How does DDP scale to multiple nodes?

---

# 📝 Key Takeaways

- **Data Parallelism** replicates the entire model on every GPU while distributing the data.
- **Distributed Data Parallel (DDP)** is the preferred PyTorch implementation because it uses one process per GPU and synchronizes gradients efficiently.
- **AllReduce** averages gradients across GPUs so that every model replica remains identical.
- **NCCL** provides optimized GPU-to-GPU communication for distributed training.
- **DistributedSampler** ensures each GPU processes a unique subset of the dataset, avoiding redundant computation and improving scalability.
# Part 2 — Model Parallelism, Tensor Parallelism & Pipeline Parallelism

This section covers the techniques used when a model is **too large to fit into the memory of a single GPU**.

Unlike Data Parallelism (DDP), where every GPU stores a complete model, these techniques split the **model itself** across multiple GPUs.

---

# ❓ Q1. Why is Data Parallelism not sufficient for very large models?

## ✅ Answer

Data Parallelism (DDP) assumes that **every GPU stores a complete copy of the model**.

For example, suppose we have a 70B parameter VLM.

```text
GPU0
Entire 70B Model

GPU1
Entire 70B Model

GPU2
Entire 70B Model

GPU3
Entire 70B Model
```

If the model requires **140 GB** of memory in FP16, and each GPU has only **80 GB**, then even a single copy cannot fit on one GPU.

In this case, DDP cannot be used.

Instead, we split the model itself across GPUs using:

- Model Parallelism
- Tensor Parallelism
- Pipeline Parallelism

---

# ❓ Q2. What is Model Parallelism?

## ✅ Answer

Model Parallelism divides the neural network into different parts and places each part on a different GPU.

Instead of copying the model,

the model is partitioned.

Example:

```text
               Input Image

                    │

                    ▼

GPU0

Vision Encoder

                    │

                    ▼

GPU1

Multimodal Projector

                    │

                    ▼

GPU2

LLM Layers 1-20

                    │

                    ▼

GPU3

LLM Layers 21-40

                    │

                    ▼

Output
```

Each GPU computes only its assigned layers.

The output of one GPU becomes the input of the next GPU.

---

### Advantages

- Very large models fit into memory.
- Memory usage is divided across GPUs.

### Disadvantages

- GPUs often wait for previous GPUs.
- Communication overhead increases.
- Lower utilization than DDP.

---

# ❓ Q3. How does the forward pass work in Model Parallelism?

## ✅ Answer

Suppose the model is divided across four GPUs.

Forward pass:

```text
Input

↓

GPU0

Vision Encoder

↓

Hidden Representation

↓

GPU1

Projector

↓

Projected Embedding

↓

GPU2

LLM Layers 1-20

↓

Intermediate Features

↓

GPU3

LLM Layers 21-40

↓

Prediction
```

Each GPU waits for the previous GPU to finish before continuing.

---

# ❓ Q4. How does the backward pass work in Model Parallelism?

## ✅ Answer

The backward pass proceeds in the reverse direction.

```text
Loss

↓

GPU3

Compute Gradients

↓

GPU2

Compute Gradients

↓

GPU1

Compute Gradients

↓

GPU0

Compute Gradients
```

Gradients flow backward through the GPUs using the chain rule.

Each GPU updates only the parameters that it owns.

---

# ❓ Q5. What are the drawbacks of Model Parallelism?

## ✅ Answer

Although Model Parallelism allows larger models to fit into memory, it introduces several challenges.

### 1. Sequential Execution

GPU1 cannot start until GPU0 finishes.

```text
GPU0

██████████

GPU1

          ██████████

GPU2

                    ██████████
```

This reduces GPU utilization.

---

### 2. Communication Overhead

Hidden activations must be transferred between GPUs after every stage.

```text
GPU0

↓

Transfer Tensor

↓

GPU1

↓

Transfer Tensor

↓

GPU2
```

Communication can become a bottleneck.

---

### 3. Poor Scalability

Adding more GPUs increases communication and synchronization costs.

---

# ❓ Q6. What is Tensor Parallelism?

## ✅ Answer

Tensor Parallelism splits **individual operations inside a layer** across multiple GPUs.

Instead of assigning whole layers to different GPUs,

one layer is divided among several GPUs.

Suppose we have a linear layer

```text
Input

↓

Linear Layer

↓

4096 Output Neurons
```

Instead of computing all 4096 outputs on one GPU,

split them.

```text
GPU0

Computes Output Neurons 1-2048

GPU1

Computes Output Neurons 2049-4096
```

The outputs are then combined.

---

### Workflow

```text
Input

↓

GPU0 → Half of Matrix Multiplication

GPU1 → Other Half

↓

Concatenate Results

↓

Next Layer
```

---

### Advantages

- Handles extremely large Transformer layers.
- Excellent scaling.
- Widely used for LLMs.

---

# ❓ Q7. How is Tensor Parallelism different from Model Parallelism?

## ✅ Answer

| Model Parallelism | Tensor Parallelism |
|-------------------|-------------------|
| Splits layers across GPUs | Splits computations inside a layer |
| One GPU owns entire layers | Multiple GPUs work on the same layer |
| Less synchronization inside a layer | Requires frequent synchronization |
| Easier to understand | More complex implementation |

---

# ❓ Q8. Why is Tensor Parallelism widely used for Transformers?

## ✅ Answer

Transformer models contain extremely large matrix multiplications.

Example

```text
Linear Layer

Input Dimension

16384

Output Dimension

16384
```

A single weight matrix contains

```text
16384 × 16384
```

which may not fit into one GPU.

Tensor Parallelism divides this matrix across GPUs.

Each GPU computes only part of the matrix multiplication.

This allows training of models with hundreds of billions of parameters.

---

# ❓ Q9. What is Pipeline Parallelism?

## ✅ Answer

Pipeline Parallelism divides the model into stages and keeps multiple GPUs busy by processing **different micro-batches simultaneously**.

Instead of waiting,

GPUs work on different batches.

Example

```text
GPU0

Vision Encoder

GPU1

Projector

GPU2

LLM Layers 1-20

GPU3

LLM Layers 21-40
```

Micro-batch scheduling

```text
Time

T1

GPU0 → Batch1

GPU1 → Idle

GPU2 → Idle

GPU3 → Idle

--------------------------------

T2

GPU0 → Batch2

GPU1 → Batch1

GPU2 → Idle

GPU3 → Idle

--------------------------------

T3

GPU0 → Batch3

GPU1 → Batch2

GPU2 → Batch1

GPU3 → Idle

--------------------------------

T4

GPU0 → Batch4

GPU1 → Batch3

GPU2 → Batch2

GPU3 → Batch1
```

Eventually every GPU is busy.

---

# ❓ Q10. What are Pipeline Bubbles?

## ✅ Answer

At the beginning and end of a pipeline,

some GPUs remain idle.

Example

```text
GPU0

████████████████

GPU1

    ████████████

GPU2

        ████████

GPU3

            ████
```

The empty regions are called **pipeline bubbles**.

They reduce GPU utilization.

---

# ❓ Q11. How do Micro-batches reduce Pipeline Bubbles?

## ✅ Answer

Instead of sending one large batch,

divide it into several smaller batches.

Example

Batch Size

```text
128
```

Instead of

```text
128
```

Use

```text
32

32

32

32
```

While GPU0 processes Micro-batch 4,

GPU1 processes Micro-batch 3,

GPU2 processes Micro-batch 2,

GPU3 processes Micro-batch 1.

This keeps nearly every GPU busy.

---

# ❓ Q12. Can Tensor Parallelism and Pipeline Parallelism be combined?

## ✅ Answer

Yes.

Large LLMs often combine multiple parallelism strategies.

Example

```text
Pipeline Stage 1

GPU0 + GPU1

Tensor Parallel

↓

Pipeline Stage 2

GPU2 + GPU3

Tensor Parallel

↓

Pipeline Stage 3

GPU4 + GPU5

Tensor Parallel

↓

Pipeline Stage 4

GPU6 + GPU7

Tensor Parallel
```

This approach enables training of models with hundreds of billions of parameters.

---

# ❓ Q13. Which parallelism strategy is used in modern VLMs?

## ✅ Answer

Most production VLMs use a combination of techniques.

Typical setup

```text
Data Parallelism (Across Nodes)

↓

Tensor Parallelism (Inside LLM Layers)

↓

Pipeline Parallelism (Across Transformer Blocks)

↓

Activation Checkpointing

↓

Mixed Precision

↓

FSDP or ZeRO
```

Large-scale training almost never relies on a single parallelism strategy.

---

# ❓ Q14. How would you train a 70B Vision-Language Model on 8 GPUs?

## ✅ Answer

A practical strategy would be:

- Use **Tensor Parallelism** to split large Transformer layers.
- Use **Pipeline Parallelism** to divide Transformer blocks across GPUs.
- Use **FSDP** or **DeepSpeed ZeRO-3** to shard parameters, gradients, and optimizer states.
- Use **BF16 mixed precision** to reduce memory usage.
- Enable **Activation Checkpointing** to save activation memory.
- Use **Gradient Accumulation** to achieve a larger effective batch size.
- Use **DistributedSampler** for efficient data loading.

This combination balances memory usage, communication overhead, and training throughput.

---

# ⭐ Interview Tips

### Common Follow-Up Questions

- Why can't DDP train a 70B model?
- Which is better: Model Parallelism or Tensor Parallelism?
- Why do Transformers benefit from Tensor Parallelism?
- What causes pipeline bubbles?
- Why are micro-batches necessary?
- Can Tensor Parallelism and DDP be used together?
- Why do companies combine multiple parallelism techniques?

---

# 📝 Key Takeaways

- **Model Parallelism** splits the model into different layers across GPUs.
- **Tensor Parallelism** splits the computations within a single layer across GPUs.
- **Pipeline Parallelism** processes multiple micro-batches simultaneously to improve GPU utilization.
- **Pipeline bubbles** are idle periods at the start and end of a pipeline; using **micro-batches** helps reduce them.
- Modern VLM training typically combines **Data Parallelism**, **Tensor Parallelism**, **Pipeline Parallelism**, **FSDP/ZeRO**, **Mixed Precision**, and **Activation Checkpointing** to efficiently train very large models.
# Part 3 — FSDP, DeepSpeed ZeRO, Mixed Precision & Gradient Accumulation

This chapter covers the memory optimization techniques used to train billion-parameter Vision-Language Models (VLMs) such as LLaVA, Qwen2.5-VL, InternVL, Flamingo, and BLIP-2.

---

# ❓ Q1. What is Fully Sharded Data Parallel (FSDP)?

## ✅ Answer

Fully Sharded Data Parallel (FSDP) is a distributed training strategy that reduces GPU memory usage by **sharding** (splitting) the model across multiple GPUs.

Unlike Distributed Data Parallel (DDP), where every GPU stores a complete copy of the model, FSDP stores only a fraction of the model on each GPU.

Example:

### DDP

```text
GPU0 → Entire Model

GPU1 → Entire Model

GPU2 → Entire Model

GPU3 → Entire Model
```

Every GPU stores 100% of the model.

---

### FSDP

```text
GPU0 → Parameters 1–25%

GPU1 → Parameters 26–50%

GPU2 → Parameters 51–75%

GPU3 → Parameters 76–100%
```

Each GPU stores only 25% of the parameters.

---

This allows much larger models to fit into GPU memory.

---

# ❓ Q2. How does FSDP work?

## ✅ Answer

FSDP works in four stages.

### Step 1

Each GPU stores only its assigned parameter shard.

```text
GPU0

P1

GPU1

P2

GPU2

P3

GPU3

P4
```

---

### Step 2

Before computing a layer,

FSDP performs an **AllGather** operation.

```text
GPU0

↓

AllGather

↓

Complete Layer Parameters
```

The layer is reconstructed temporarily.

---

### Step 3

Forward and backward passes are executed.

---

### Step 4

After gradients are computed,

the parameters are discarded again.

Only the assigned shard remains.

Memory usage stays low throughout training.

---

# ❓ Q3. Why is FSDP better than DDP?

## ✅ Answer

DDP replicates the entire model on every GPU.

Memory requirement

```text
GPU0

100%

GPU1

100%

GPU2

100%

GPU3

100%
```

FSDP

```text
GPU0

25%

GPU1

25%

GPU2

25%

GPU3

25%
```

Advantages:

- Lower GPU memory
- Enables larger batch sizes
- Allows training larger models
- Better scalability

---

# ❓ Q4. What does FSDP shard?

## ✅ Answer

FSDP shards

- Model Parameters
- Gradients
- Optimizer States

instead of replicating them.

Memory savings come from distributing all three across GPUs.

---

# ❓ Q5. What is DeepSpeed ZeRO?

## ✅ Answer

ZeRO stands for

**Zero Redundancy Optimizer**

Its goal is the same as FSDP:

Reduce GPU memory by eliminating redundant copies.

Instead of every GPU storing

- Parameters
- Gradients
- Optimizer States

ZeRO distributes them.

---

# ❓ Q6. Explain ZeRO Stage 1.

## ✅ Answer

Stage 1 shards only the optimizer states.

### DDP

```text
GPU0

Parameters

Gradients

Optimizer

GPU1

Parameters

Gradients

Optimizer
```

Optimizer memory is duplicated.

---

### ZeRO Stage 1

```text
GPU0

Optimizer Part A

GPU1

Optimizer Part B

GPU2

Optimizer Part C

GPU3

Optimizer Part D
```

Parameters and gradients are still replicated.

Only optimizer states are sharded.

---

# ❓ Q7. Explain ZeRO Stage 2.

## ✅ Answer

Stage 2 shards

- Optimizer States
- Gradients

Parameters remain replicated.

```text
GPU0

Parameters

Gradient Part A

Optimizer Part A

GPU1

Parameters

Gradient Part B

Optimizer Part B
```

Memory savings are significantly better than Stage 1.

---

# ❓ Q8. Explain ZeRO Stage 3.

## ✅ Answer

Stage 3 shards

- Parameters
- Gradients
- Optimizer States

Everything is distributed.

```text
GPU0

Parameter Part A

Gradient Part A

Optimizer Part A

GPU1

Parameter Part B

Gradient Part B

Optimizer Part B
```

Stage 3 provides the maximum memory savings.

---

# ❓ Q9. Difference between FSDP and ZeRO?

## ✅ Answer

| FSDP | ZeRO |
|-------|------|
| Native PyTorch | DeepSpeed library |
| Shards parameters, gradients, optimizer states | Progressive sharding (Stages 1–3) |
| Uses AllGather before computation | Uses communication based on ZeRO stage |
| Easier integration with PyTorch | Rich ecosystem for massive models |

Both aim to reduce redundant memory usage.

---

# ❓ Q10. What is Mixed Precision Training?

## ✅ Answer

Mixed Precision Training uses lower precision data types to reduce memory usage and increase training speed.

Instead of

```text
FP32
```

training uses

```text
FP16

or

BF16
```

Benefits:

- Faster matrix multiplication
- Lower memory
- Larger batch sizes
- Higher throughput

---

# ❓ Q11. Difference between FP32, FP16 and BF16?

## ✅ Answer

| Precision | Bits | Memory | Numerical Stability |
|-----------|------|--------|---------------------|
| FP32 | 32 | High | Excellent |
| FP16 | 16 | Low | Can underflow/overflow |
| BF16 | 16 | Low | Better than FP16 |

---

FP16

- Smaller exponent
- More prone to overflow

BF16

- Same exponent range as FP32
- Better numerical stability

Modern GPUs often prefer BF16 for training.

---

# ❓ Q12. Why is Loss Scaling required in FP16?

## ✅ Answer

FP16 has limited numerical range.

Very small gradients may become

```text
0
```

This is called **underflow**.

To prevent this,

multiply the loss by a scaling factor.

```text
Original Loss

↓

Multiply by 1024

↓

Backward Pass

↓

Gradients become larger

↓

Divide gradients by 1024

↓

Optimizer Step
```

This preserves gradient information.

---

# ❓ Q13. Why is BF16 preferred over FP16?

## ✅ Answer

BF16 has the same exponent size as FP32.

Therefore,

- Better dynamic range
- Less overflow
- Less underflow
- Usually no manual loss scaling

This makes BF16 the preferred format on modern hardware like NVIDIA A100 and H100 GPUs.

---

# ❓ Q14. What is Gradient Accumulation?

## ✅ Answer

Sometimes GPU memory cannot fit the desired batch size.

Suppose we want

```text
Batch Size = 256
```

GPU memory only supports

```text
32
```

Instead of updating after every batch,

accumulate gradients.

Workflow

```text
Forward

↓

Backward

↓

Store Gradients

↓

Forward

↓

Backward

↓

Store Gradients

↓

Repeat

↓

Optimizer Step
```

---

# ❓ Q15. What is Effective Batch Size?

## ✅ Answer

Effective Batch Size

=

```text
Micro Batch Size

×

Gradient Accumulation Steps

×

Number of GPUs
```

Example

```text
Micro Batch = 8

GPUs = 4

Accumulation = 8
```

Effective Batch

```text
8 × 4 × 8

=

256
```

---

# ❓ Q16. Why use Gradient Accumulation?

## ✅ Answer

Advantages

- Train with larger effective batch sizes
- Reduce GPU memory usage
- Improve gradient stability
- Works well with limited GPU memory

Trade-off

- More forward and backward passes
- Longer training time

---

# ❓ Q17. Can Gradient Accumulation be combined with DDP?

## ✅ Answer

Yes.

Typical workflow

```text
Forward

↓

Backward

↓

No Optimizer Step

↓

Forward

↓

Backward

↓

No Optimizer Step

↓

Repeat N times

↓

AllReduce

↓

Optimizer Step
```

This allows extremely large effective batch sizes.

---

# ❓ Q18. How would you train a 70B VLM on eight GPUs?

## ✅ Answer

A practical strategy would be:

- FSDP or ZeRO Stage 3 for parameter sharding.
- Tensor Parallelism for large Transformer layers.
- Pipeline Parallelism for Transformer blocks.
- BF16 Mixed Precision.
- Gradient Accumulation.
- Activation Checkpointing.
- DistributedSampler.
- Efficient dataloading.
- Flash Attention (if supported).

This combination balances memory efficiency and training throughput.

---

# ⭐ Common Interview Follow-Up Questions

- Why can't DDP train a 70B model?
- What does FSDP shard?
- Difference between AllGather and AllReduce?
- Explain ZeRO Stage 3.
- Why is BF16 preferred?
- Why does FP16 require Loss Scaling?
- Explain Effective Batch Size.
- Why use Gradient Accumulation?
- Can FSDP be combined with Tensor Parallelism?
- Why does Adam require more GPU memory?

---

# 📝 Key Takeaways

- **FSDP** shards parameters, gradients, and optimizer states across GPUs, significantly reducing memory usage.
- **DeepSpeed ZeRO** progressively removes memory redundancy through Stages 1, 2, and 3.
- **Mixed Precision Training** (FP16/BF16) reduces memory consumption and increases training speed.
- **BF16** is generally preferred over FP16 because it offers better numerical stability.
- **Loss Scaling** is used with FP16 to prevent gradient underflow.
- **Gradient Accumulation** enables large effective batch sizes without requiring more GPU memory.
- Modern VLM training typically combines **FSDP/ZeRO**, **Tensor Parallelism**, **Pipeline Parallelism**, **Mixed Precision**, **Gradient Accumulation**, and **Activation Checkpointing** for efficient large-scale training.

