# 🧠 IndicVisionBench — Key Points Summary

| Category | Details |
|---|---|
| Benchmark Name | IndicVisionBench |
| Domain | Vision-Language Models (VLMs) |
| Main Goal | Evaluate VLMs in culturally diverse and multilingual Indian settings |
| Problem Addressed | Existing VLM benchmarks are mostly Western-centric |
| Region Focus | Indian subcontinent |
| Total Languages | 11 languages |
| Language Breakdown | English + 10 Indian languages |
| Total Tasks | 3 multimodal tasks |
| Tasks Included | OCR, Multimodal Machine Translation (MMT), Visual Question Answering (VQA) |
| Question Types | 6 types |
| Total Images | 5K images |
| Total QA Pairs | 37K+ QA pairs |
| Cultural Topics Covered | 13 culturally grounded topics |
| Additional Resource | Parallel annotation corpus across 10 Indic languages |
| Purpose of Parallel Corpus | Analyze linguistic and cultural biases in VLMs |
| Number of Models Evaluated | 8 models |
| Model Types | Proprietary closed-source + open-weight medium/large-scale models |
| Key Experimental Finding | Significant performance gaps across culturally diverse settings |
| Main Conclusion | Current VLMs struggle in multilingual and culturally diverse scenarios |
| Contribution | Establishes reproducible evaluation framework for inclusive multimodal AI |

---

# 📊 Important Numbers

| Metric | Value |
|---|---|
| Indian Languages | 10 |
| Total Languages Including English | 11 |
| Multimodal Tasks | 3 |
| Question Types | 6 |
| Images | 5,000 |
| QA Pairs | 37,000+ |
| Cultural Topics | 13 |
| Evaluated Models | 8 |

---

# 📌 Core Contributions

| Contribution | Explanation |
|---|---|
| First India-Centric VLM Benchmark | Focuses on Indian multilingual and cultural settings |
| Multilingual Evaluation | Supports 10 Indic languages |
| Cultural Diversity Analysis | Tests VLM understanding beyond Western datasets |
| OCR + MMT + VQA Coverage | Evaluates multiple multimodal capabilities |
| Parallel Corpus Release | Enables bias and fairness research |
| Reproducible Framework | Standardized benchmark for future VLM research |

---

# 🧠 Main Insight from the Paper

```text
Current VLMs perform significantly worse in culturally diverse and multilingual Indian contexts compared to standard Western-centric benchmarks.
```

---

# 🎤 Interview-Friendly Summary

> “IndicVisionBench is the first large-scale benchmark designed to evaluate Vision-Language Models in culturally grounded and multilingual Indian settings. It includes 5K images, 37K+ QA pairs, 11 languages, and 3 multimodal tasks such as OCR, VQA, and multimodal translation. The benchmark highlights major performance gaps in current VLMs and provides a reproducible framework for building more inclusive multimodal AI systems.”

# 🧠 IndicVisionBench — Models, Tasks, Benchmarks, and Metrics

---

# 📌 Task Categories and Evaluation Setup

| Task | Purpose | Benchmark Type | Metrics Used | Main Metric |
|---|---|---|---|---|
| OCR (Optical Character Recognition) | Extract text from images in Indic languages | OCR Benchmark | ANLS, WER, CER | ANLS |
| MMT (Multimodal Machine Translation) | Translate multimodal content across Indic languages | Translation Benchmark | BLEU, RIBES | BLEU + RIBES |
| VQA (Visual Question Answering) | Answer image-based questions | Multimodal QA Benchmark | Exact Match, GPT-4o Judge Score | Exact Match / Judge Score |

---

# 🥇 OCR Benchmark

## 📌 Purpose

Evaluate:
- text extraction capability
- Indic language OCR robustness

---

# 📊 OCR Metrics

| Metric | Full Form | Purpose |
|---|---|---|
| ANLS | Average Normalized Levenshtein Similarity | Measures OCR text similarity |
| WER | Word Error Rate | Measures incorrect words |
| CER | Character Error Rate | Measures incorrect characters |

---

# 📌 Main OCR Metric

| Metric | Reason |
|---|---|
| ANLS | More robust to OCR outliers and partial mismatches |

---

# 🧠 OCR Models Evaluated

| Model | Type |
|---|---|
| Chitrapathak | Closed-source OCR model for Indic languages |
| Surya | Open-source OCR model |

---

# 🥈 MMT Benchmark

## 📌 Purpose

Evaluate:
- multimodal translation capability
- image + language understanding

across:
- 10 Indic languages

---

# 📊 MMT Metrics

| Metric | Full Form | Purpose |
|---|---|---|
| BLEU | Bilingual Evaluation Understudy | Measures translation overlap |
| RIBES | Rank-based Intuitive Bilingual Evaluation Score | Measures word order and fluency |

---

# 📌 MMT Special Model

| Model | Description |
|---|---|
| Chitranuvad | Winning model of WMT’24 English-to-LowRes MMT shared task |

---

# 🥉 VQA Benchmark

## 📌 Purpose

Evaluate:
- image understanding
- reasoning
- cultural grounding
- multilingual QA

---

# 📊 VQA Question Types

| Question Type | Evaluation Method |
|---|---|
| Multiple Choice | Exact Match |
| True/False | Exact Match |
| Short Answer | GPT-4o Judge |
| Long Answer | GPT-4o Judge |
| Adversarial Questions | GPT-4o Judge |
| Contextual/Cultural Questions | GPT-4o Judge |

---

# 📊 VQA Metrics

| Metric | Purpose |
|---|---|
| Exact Match | Strict answer correctness |
| GPT-4o Judge Score (0–10) | Contextual and cultural evaluation |

---

# 📌 Why LLM-as-a-Judge Was Used

Exact matching fails when:
- answers are semantically correct
- wording differs

Thus:
- GPT-4o judge scoring captures:
  - contextual correctness
  - cultural appropriateness
  - reasoning quality

---

# 🧠 Families of Models Evaluated

---

# 🥇 Proprietary Closed-Source Models

| Model | Organization |
|---|---|
| Gemini-2.5 Flash | Google |
| GPT-4o | OpenAI |

---

# 🥈 Large Open-Weight VLMs

| Model | Scale |
|---|---|
| Gemma-3-27B | 27B |
| LLaMA-4-Maverick-17B | 17B |

---

# 🥉 Medium-Scale Open-Weight VLMs

| Model | Approx Size |
|---|---|
| Maya | 7B |
| PALO | 7B |
| Pangea | 7B |
| Chitrarth-1 | 7B |

---

# 📊 Complete Benchmark Structure

| Component | Details |
|---|---|
| Total Languages | 11 |
| Indic Languages | 10 |
| Total Images | 5K |
| QA Pairs | 37K+ |
| Tasks | OCR, MMT, VQA |
| Question Types | 6 |
| Topics Covered | 13 culturally grounded topics |
| Models Evaluated | 8+ |

---

# 📌 Evaluation Philosophy

The benchmark combines:

| Evaluation Style | Purpose |
|---|---|
| Deterministic Metrics | Objective scoring |
| LLM-as-a-Judge | Contextual semantic evaluation |
| OCR Similarity Metrics | Robust text extraction analysis |
| Translation Metrics | Language fluency + alignment |

---

# 🧠 Key Benchmarking Insight

```text
Current VLMs struggle significantly in multilingual and culturally grounded Indian contexts.
```

Performance gaps observed across:
- OCR
- translation
- cultural reasoning
- multilingual QA

---

# 🎤 Interview-Friendly Summary

> “IndicVisionBench evaluates Vision-Language Models across OCR, Multimodal Machine Translation, and Visual Question Answering tasks using task-specific benchmarking metrics. OCR uses ANLS, WER, and CER; MMT uses BLEU and RIBES; and VQA combines Exact Match with GPT-4o-based judge scoring for contextual and cultural evaluation. The benchmark compares proprietary and open-weight VLMs across multilingual Indic settings.”
