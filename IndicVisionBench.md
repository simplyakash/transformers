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

# 🧠 ANLS (Average Normalized Levenshtein Similarity)

ANLS is an OCR evaluation metric used to measure:

```text
how similar predicted text is to ground truth text
```

It is widely used because:
- small OCR mistakes are tolerated
- partial matches are rewarded

---

# 📌 Core Idea

ANLS is based on:

```text
Levenshtein Distance
```

which counts:
- insertions
- deletions
- substitutions

needed to convert one string into another.

---

# 📐 Formula

```text
ANLS = 1 - (Levenshtein Distance / Max String Length)


```
Max String Length=max(len(word),len(GT) )

---

# 📦 Example

## Ground Truth

```text
"amazon"
```

---

## OCR Prediction

```text
"amaz0n"
```

Difference:
- `o → 0`

Only:
- 1 substitution

Thus:

```text
Levenshtein Distance = 1
```

Maximum string length:

```text
7
```

---

# 📊 ANLS Calculation

```text
ANLS = 1 - (1 / 7)
     = 0.857
```

---

# 📌 Interpretation

| ANLS Score | Meaning |
|---|---|
| 1.0 | Perfect OCR |
| 0.8+ | Good OCR |
| 0.5 | Partial match |
| 0 | Completely wrong |

---

# 🧠 Why ANLS is Better Than Exact Match

Exact Match:

```text
amazon ≠ amaz0n
```

would give:

```text
0
```

even though prediction is almost correct.

ANLS instead gives:
- partial credit

making it more robust for OCR evaluation.

---

# 📌 Common OCR Errors Handled

| Error Type | Example |
|---|---|
| Character substitution | O ↔ 0 |
| Missing character | amazn |
| Extra character | amazoon |
| Minor spelling issue | amzon |

---



# 🎤 Interview-Friendly Explanation

> “ANLS, or Average Normalized Levenshtein Similarity, measures OCR quality by comparing predicted text with ground truth using edit distance. Unlike exact match, it gives partial credit for near-correct predictions, making it more robust for OCR evaluation.”


# 🧠 Word Error Rate (WER)

WER measures:

```text
how many words are incorrect in predicted text
```

It is widely used in:
- OCR
- speech recognition
- transcription systems

---

# 📐 Formula

```text
WER = (S + D + I) / N
```

Where:

| Symbol | Meaning |
|---|---|
| S | Substitutions |
| D | Deletions |
| I | Insertions |
| N | Total words in ground truth |

---

# 📦 Example

## Ground Truth

```text
"the cat is black"
```

---

## Prediction

```text
"the cat black"
```

Missing word:

```text
"is"
```

Thus:

| Error Type | Count |
|---|---|
| Deletions (D) | 1 |
| Substitutions (S) | 0 |
| Insertions (I) | 0 |

Total ground truth words:

```text
4
```

---

# 📊 WER Calculation

```text
WER = (0 + 1 + 0) / 4
    = 0.25
```

---

# 📌 Interpretation

```text
25% word error
```

---

# 📌 WER Range

| WER | Quality |
|---|---|
| 0 | Perfect prediction |
| Low WER | Better system |
| High WER | Poor prediction |

---

# 🧠 Error Types

| Error | Example |
|---|---|
| Substitution | cat → bat |
| Deletion | missing word |
| Insertion | extra word added |

---

# 📌 Example with All Errors

## Ground Truth

```text
"amazon is growing fast"
```

---

## Prediction

```text
"amazon growing very fast"
```

Errors:

| Type | Example |
|---|---|
| Deletion | "is" removed |
| Insertion | "very" added |

Thus:

```text
WER = (0 + 1 + 1) / 4
    = 0.5
```

---

# 📌 Difference Between WER and CER

| Metric | Unit |
|---|---|
| WER | Word-level errors |
| CER | Character-level errors |

---

# 🎤 Interview-Friendly Explanation

> “WER, or Word Error Rate, measures the percentage of incorrect words in predicted text compared to ground truth. It is calculated using substitutions, deletions, and insertions divided by the total number of ground truth words.”



# 🧠 BLEU Score Example

BLEU measures:

```text
how similar machine translation is to human reference translation
```

using:
- word overlap
- phrase overlap

---

# 📦 Example

## Reference Translation

```text
"The cat is sitting on the mat"
```

---

## Model Prediction

```text
"The cat sits on the mat"
```

---

# 🥇 Step 1 — Unigram Matching

Compare individual words.

| Predicted Word | Present in Reference? |
|---|---|
| The | ✅ |
| cat | ✅ |
| sits | ❌ |
| on | ✅ |
| the | ✅ |
| mat | ✅ |

---

# 📊 Unigram Precision

Matching words:

```text
5
```

Total predicted words:

```text
6
```

Thus:

```text
Unigram Precision = 5 / 6 = 0.833
```

---

# 🥈 Step 2 — Bigram Matching

Predicted bigrams:

```text
"The cat"
"cat sits"
"sits on"
"on the"
"the mat"
```

---

# 📌 Matching Bigrams

Reference contains:

```text
"The cat"
"on the"
"the mat"
```

Matching bigrams:

```text
3
```

Total predicted bigrams:

```text
5
```

---

# 📊 Bigram Precision

```text
3 / 5 = 0.6
```

---

# 🥉 BLEU Combines Multiple n-grams

BLEU combines:
- unigram
- bigram
- trigram
- 4-gram scores

using:
- geometric mean
- brevity penalty

---

# 📌 Final Interpretation

Prediction is:

```text
mostly correct
```

So:
- BLEU score becomes reasonably high.

---

# 📊 BLEU Score Intuition

| BLEU | Quality |
|---|---|
| 1.0 | Perfect translation |
| > 0.7 | Excellent |
| 0.4 – 0.7 | Good |
| < 0.2 | Poor |

---

# ⚠️ Limitation

BLEU checks:
- overlap

not:
- true semantic understanding

Thus:
- semantically correct sentences may still get lower BLEU.

---

# 🎤 Interview-Friendly Explanation

> “BLEU score evaluates translation quality by measuring n-gram overlap between machine-generated translation and human reference translation. It computes precision over matching phrases and combines unigram to 4-gram similarities to produce the final score.”

# 🧠 RIBES Score

RIBES stands for:

```text
Rank-based Intuitive Bilingual Evaluation Score
```

It is a machine translation metric that focuses on:

```text
word order correctness
```

and:
- sentence fluency

---

# 📌 Why RIBES is Needed

BLEU mainly checks:
- word overlap

But languages like:
- Hindi
- Japanese
- Indic languages

depend heavily on:
- correct word ordering

Thus RIBES was introduced.

---

# 📦 Example

## Reference Translation

```text
"I eat mango daily"
```

---

## Prediction 1

```text
"I eat mango daily"
```

Perfect order:
- high RIBES

---

## Prediction 2

```text
"daily mango eat I"
```

Same words:
- but wrong order

BLEU may still give:
- moderate score

But RIBES gives:
- low score

because:
- word ranking/order is incorrect.

---

# 🧠 Core Idea

RIBES measures:

```text
how well word order in prediction matches reference
```

using:
- rank correlation
- word alignment ordering

---

# 📊 What RIBES Evaluates

| Aspect | Importance |
|---|---|
| Word Order | Very High |
| Fluency | High |
| Correct Phrase Structure | High |
| Exact Word Matching | Moderate |

---

# 📌 RIBES Range

| Score | Meaning |
|---|---|
| 1.0 | Perfect translation/order |
| High | Good fluency |
| Low | Poor sentence structure |

---

# 📌 Why Useful for Indic Languages

Many Indic languages have:
- flexible grammar
- rich morphology
- different sentence structures

Thus:
- preserving proper word order becomes important.

---

# 📊 BLEU vs RIBES

| Metric | Focus |
|---|---|
| BLEU | Word overlap |
| RIBES | Word order and fluency |

---

# 🎤 Interview-Friendly Explanation

> “RIBES is a machine translation evaluation metric that focuses on word order and sentence fluency using rank correlation. Unlike BLEU, which mainly measures n-gram overlap, RIBES evaluates whether translated words appear in the correct relative order, making it especially useful for multilingual and Indic language translation tasks.”
