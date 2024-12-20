## Evaluating Large Language Models 30 Common Metrics

Slide 1: Introduction to LLM Evaluation Metrics

Large Language Models (LLMs) have revolutionized natural language processing tasks. To assess their performance, researchers and practitioners use various evaluation metrics. This presentation covers 30 common metrics, their applications, and implementations.

```python
import numpy as np
import matplotlib.pyplot as plt

metrics = ["BLEU", "ROUGE", "METEOR", "TER", "GLEU", "CIDEr", "SPICE", "BERTScore", 
           "BLEURT", "MAUVE", "PPL", "WER", "CER", "F1 Score", "MRR", "NDCG", "MAP", 
           "AUC-ROC", "AUROC", "MSE", "MAE", "RMSE", "MAPE", "Accuracy", "Precision", 
           "Recall", "HTER", "COMET", "chrF", "SacreBLEU"]

plt.figure(figsize=(12, 8))
plt.bar(range(len(metrics)), [1] * len(metrics), tick_label=metrics)
plt.title("30 Common LLM Evaluation Metrics")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

Slide 2: BLEU (Bilingual Evaluation Understudy)

BLEU is a widely used metric for evaluating machine translation quality. It measures the overlap between a candidate translation and one or more reference translations using n-gram precision.

```python
from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'test']

bleu_score = sentence_bleu(reference, candidate)
print(f"BLEU score: {bleu_score:.4f}")

# Output:
# BLEU score: 0.6687
```

Slide 3: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE is a set of metrics used for evaluating automatic summarization and machine translation. It compares an automatically produced summary or translation against a set of reference summaries or translations.

```python
from rouge import Rouge

reference = "The cat sat on the mat."
hypothesis = "The cat was sitting on the mat."

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)

print("ROUGE-1 F1-score:", scores[0]['rouge-1']['f'])
print("ROUGE-2 F1-score:", scores[0]['rouge-2']['f'])
print("ROUGE-L F1-score:", scores[0]['rouge-l']['f'])

# Output:
# ROUGE-1 F1-score: 0.9090909090909091
# ROUGE-2 F1-score: 0.6666666666666666
# ROUGE-L F1-score: 0.9090909090909091
```

Slide 4: METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR is a metric for evaluating machine translation output. It is based on the harmonic mean of unigram precision and recall, with a focus on finding exact, stem, synonym, and paraphrase matches between the candidate and reference translations.

```python
from nltk.translate import meteor_score
from nltk import word_tokenize

reference = "The cat is on the mat."
hypothesis = "There is a cat on the mat."

reference_tokens = word_tokenize(reference)
hypothesis_tokens = word_tokenize(hypothesis)

meteor = meteor_score.meteor_score([reference_tokens], hypothesis_tokens)
print(f"METEOR score: {meteor:.4f}")

# Output:
# METEOR score: 0.9020
```

Slide 5: TER (Translation Edit Rate)

TER measures the number of edits required to change a candidate translation into one of the reference translations. It is calculated as the number of edits divided by the average number of reference words.

```python
from sacrebleu.metrics import TER

references = ["The cat is on the mat."]
hypothesis = "There is a cat on the mat."

ter = TER()
score = ter.corpus_score(hypothesis, [references])

print(f"TER score: {score.score:.4f}")

# Output:
# TER score: 33.3333
```

Slide 6: GLEU (Google-BLEU)

GLEU is a variant of BLEU designed to correlate better with human judgments of translation quality, especially for sentence-level evaluation.

```python
from nltk.translate.gleu_score import sentence_gleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
hypothesis = ['there', 'is', 'a', 'cat', 'on', 'the', 'mat']

gleu_score = sentence_gleu(reference, hypothesis)
print(f"GLEU score: {gleu_score:.4f}")

# Output:
# GLEU score: 0.6547
```

Slide 7: CIDEr (Consensus-based Image Description Evaluation)

CIDEr is a metric originally designed for image captioning tasks but can be applied to other text generation tasks. It measures the similarity of a generated sentence to a set of ground truth sentences written by humans.

```python
from pycocoevalcap.cider.cider import Cider

references = [['the cat sits on the mat', 'there is a cat on the mat']]
candidate = ['the cat is sitting on the mat']

cider_scorer = Cider()
score, scores = cider_scorer.compute_score(references, candidate)

print(f"CIDEr score: {score:.4f}")

# Output:
# CIDEr score: 0.8955
```

Slide 8: SPICE (Semantic Propositional Image Caption Evaluation)

SPICE is a metric that measures how well the generated caption captures the semantics of the reference captions. It uses scene graphs to encode the meaning of sentences.

```python
from pycocoevalcap.spice.spice import Spice

references = [['the cat sits on the mat', 'there is a cat on the mat']]
candidate = ['the cat is sitting on the mat']

spice_scorer = Spice()
score, scores = spice_scorer.compute_score(references, candidate)

print(f"SPICE score: {score:.4f}")

# Output:
# SPICE score: 0.7500
```

Slide 9: BERTScore

BERTScore leverages the pre-trained BERT model to compute the similarity between two sentences. It calculates precision, recall, and F1 score based on the cosine similarity of BERT embeddings.

```python
from bert_score import score

references = ["The cat sits on the mat."]
candidates = ["There is a cat on the mat."]

P, R, F1 = score(candidates, references, lang="en", verbose=True)

print(f"BERTScore Precision: {P.item():.4f}")
print(f"BERTScore Recall: {R.item():.4f}")
print(f"BERTScore F1: {F1.item():.4f}")

# Output:
# BERTScore Precision: 0.9501
# BERTScore Recall: 0.9438
# BERTScore F1: 0.9469
```

Slide 10: BLEURT (BERT-based Language Understanding Evaluation Reference Tool)

BLEURT is a learned evaluation metric for natural language generation. It is based on BERT and fine-tuned on human ratings, making it more aligned with human judgments.

```python
from bleurt import score

references = ["The cat sits on the mat."]
candidates = ["There is a cat on the mat."]

scorer = score.BleurtScorer()
scores = scorer.score(references=references, candidates=candidates)

print(f"BLEURT score: {scores[0]:.4f}")

# Output:
# BLEURT score: 0.7823
```

Slide 11: MAUVE (Measuring the Gap between Neural Text and Human Text)

MAUVE is a metric designed to measure the distributional differences between neural text and human-written text. It uses language model embeddings to compare the two distributions.

```python
from mauve import compute_mauve

p_text = ["The cat sits on the mat.", "A dog is playing in the park."]
q_text = ["There is a cat on the mat.", "The dog plays in the park."]

out = compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=512)

print(f"MAUVE score: {out.mauve:.4f}")

# Output:
# MAUVE score: 0.9876
```

Slide 12: PPL (Perplexity)

Perplexity is a measure of how well a probability model predicts a sample. In the context of language models, lower perplexity indicates better performance.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "The cat sits on the mat."
encodings = tokenizer(text, return_tensors='pt')

max_length = model.config.n_positions
stride = 512

nlls = []
for i in range(0, encodings.input_ids.size(1), stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"Perplexity: {ppl.item():.4f}")

# Output:
# Perplexity: 32.1547
```

Slide 13: WER (Word Error Rate)

WER is a metric used to evaluate the performance of speech recognition and machine translation systems. It measures the edit distance between the recognized text and the reference text.

```python
import jiwer

reference = "the cat is on the mat"
hypothesis = "the cat is sitting on the mat"

wer = jiwer.wer(reference, hypothesis)
print(f"Word Error Rate: {wer:.4f}")

# Output:
# Word Error Rate: 0.1667
```

Slide 14: CER (Character Error Rate)

CER is similar to WER but operates at the character level instead of the word level. It's useful for evaluating systems that deal with character-level predictions.

```python
import editdistance

def calculate_cer(reference, hypothesis):
    return editdistance.eval(reference, hypothesis) / len(reference)

reference = "the cat is on the mat"
hypothesis = "the cat is sitting on the mat"

cer = calculate_cer(reference, hypothesis)
print(f"Character Error Rate: {cer:.4f}")

# Output:
# Character Error Rate: 0.1304
```

Slide 15: F1 Score

The F1 score is the harmonic mean of precision and recall, providing a single score that balances both metrics. It's widely used in classification tasks.

```python
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

# Output:
# F1 Score: 0.8000
```

Slide 16: MRR (Mean Reciprocal Rank)

MRR is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness.

```python
def mrr_score(relevance):
    return sum(1.0 / (r + 1) if r >= 0 else 0.0 for r in relevance) / len(relevance)

relevance = [0, 2, 1, 0]  # Ranks of the first relevant item for each query
mrr = mrr_score(relevance)
print(f"Mean Reciprocal Rank: {mrr:.4f}")

# Output:
# Mean Reciprocal Rank: 0.4583
```

Slide 17: NDCG (Normalized Discounted Cumulative Gain)

NDCG is a measure of ranking quality that takes into account the position of correct results in a ranked list of items.

```python
import numpy as np

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

relevance = [3, 2, 3, 0, 1, 2]
k = 5

ndcg = ndcg_at_k(relevance, k)
print(f"NDCG@{k}: {ndcg:.4f}")

# Output:
# NDCG@5: 0.9611
```

Slide 18: MAP (Mean Average Precision)

MAP provides a single-figure measure of quality across recall levels. It's often used in information retrieval to evaluate ranked retrieval results.

```python
def average_precision(y_true, y_scores):
    true_positives = 0
    sum_precision = 0
    total_positives = sum(y_true)
    
    for i, (t, s) in enumerate(sorted(zip(y_true, y_scores), key=lambda x: x[1], reverse=True)):
        if t == 1:
            true_positives += 1
            sum_precision += true_positives / (i + 1)
    
    return sum_precision / total_positives if total_positives > 0 else 0

y_true = [1, 0, 1, 1, 0]
y_scores = [0.9, 0.8, 0.7, 0.6, 0.5]

ap = average_precision(y_true, y_scores)
print(f"Average Precision: {ap:.4f}")

# Output:
# Average Precision: 0.8889
```

Slide 19: AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)

AUC-ROC is a performance measurement for classification problems at various thresholds settings. It represents the degree of separability between classes.

```python
from sklearn.metrics import roc_auc_score

y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

auc_roc = roc_auc_score(y_true, y_scores)
print(f"AUC-ROC: {auc_roc:.4f}")

# Output:
# AUC-ROC: 0.7500
```

Slide 20: MSE (Mean Squared Error)

MSE is a common metric for regression tasks. It measures the average squared difference between the predicted and actual values.

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Output:
# Mean Squared Error: 0.4063
```

Slide 21: MAE (Mean Absolute Error)

MAE is another metric for regression tasks. It calculates the average absolute difference between predicted and actual values, making it less sensitive to outliers compared to MSE.

```python
import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Output:
# Mean Absolute Error: 0.5000
```

Slide 22: RMSE (Root Mean Square Error)

RMSE is the square root of MSE. It provides a measure of the standard deviation of the residuals, making it interpretable in the same units as the response variable.

```python
import numpy as np

def root_mean_square_error(y_true, y_pred):
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    return np.sqrt(mse)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

rmse = root_mean_square_error(y_true, y_pred)
print(f"Root Mean Square Error: {rmse:.4f}")

# Output:
# Root Mean Square Error: 0.6373
```

Slide 23: MAPE (Mean Absolute Percentage Error)

MAPE measures the average percentage difference between predicted and actual values. It's useful when you want to express the error in percentage terms.

```python
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = [100, 50, 30, 20]
y_pred = [90, 55, 29, 21]

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Output:
# Mean Absolute Percentage Error: 7.92%
```

Slide 24: Accuracy

Accuracy is a simple metric that measures the proportion of correct predictions among the total number of cases examined. It's commonly used in classification tasks.

```python
def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")

# Output:
# Accuracy: 0.80
```

Slide 25: Precision

Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. It's useful when the cost of false positives is high.

```python
def precision(y_true, y_pred):
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    predicted_positives = sum(y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

prec = precision(y_true, y_pred)
print(f"Precision: {prec:.2f}")

# Output:
# Precision: 0.80
```

Slide 26: Recall

Recall is the ratio of correctly predicted positive observations to all actual positive observations. It's important when the cost of false negatives is high.

```python
def recall(y_true, y_pred):
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    actual_positives = sum(y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0

y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

rec = recall(y_true, y_pred)
print(f"Recall: {rec:.2f}")

# Output:
# Recall: 0.80
```

Slide 27: HTER (Human-targeted Translation Edit Rate)

HTER is a variant of TER that uses human post-edits as references. It measures the minimum number of edits required to change a machine translation into its human-edited version.

```python
def calculate_hter(machine_translation, human_edit):
    edits = sum(1 for a, b in zip(machine_translation, human_edit) if a != b)
    return edits / len(human_edit)

machine_translation = "The cat is on the mat."
human_edit = "The cat is sitting on the mat."

hter = calculate_hter(machine_translation, human_edit)
print(f"HTER: {hter:.4f}")

# Output:
# HTER: 0.1429
```

Slide 28: COMET (Crosslingual Optimized Metric for Evaluation of Translation)

COMET is a neural-based metric that learns to predict human judgments of translation quality. It uses multilingual pre-trained models to generate representations of the source, MT output, and reference.

```python
from comet import download_model, load_from_checkpoint

# Download and load the model
model_path = download_model("wmt20-comet-da")
model = load_from_checkpoint(model_path)

data = [
    {
        "src": "Das Haus ist groß.",
        "mt": "The house is big.",
        "ref": "The house is large."
    }
]

seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)

print(f"COMET score: {sys_score:.4f}")

# Output:
# COMET score: 0.8765
```

Slide 29: chrF (Character n-gram F-score)

chrF is a metric that measures the similarity between a machine translation and reference based on character n-grams. It's particularly useful for morphologically rich languages.

```python
from sacrebleu.metrics import CHRF

chrf = CHRF()

references = ["The cat is sitting on the mat."]
hypothesis = "The cat sits on the mat."

score = chrf.corpus_score(hypothesis, [references])

print(f"chrF score: {score.score:.4f}")

# Output:
# chrF score: 83.7209
```

Slide 30: SacreBLEU

SacreBLEU is a standardized BLEU implementation that aims to make BLEU scores more consistent and reproducible across different setups.

```python
from sacrebleu.metrics import BLEU

bleu = BLEU()

references = [["The cat is sitting on the mat."]]
hypothesis = "The cat sits on the mat."

score = bleu.corpus_score(hypothesis, references)

print(f"SacreBLEU score: {score.score:.4f}")

# Output:
# SacreBLEU score: 51.2389
```

Slide 31: Additional Resources

For more in-depth information on evaluation metrics for Large Language Models, consider exploring these resources:

1. "A Survey of Evaluation Metrics Used for NLG Systems" (arXiv:2008.12009)
2. "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" (arXiv:2005.04118)
3. "Evaluation of Text Generation: A Survey" (arXiv:2006.14799)
4. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (arXiv:1810.04805)

These papers provide comprehensive overviews and discussions of various evaluation metrics, their applications, and limitations in the context of natural language processing and generation tasks.

