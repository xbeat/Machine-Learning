## Understanding METEOR A Metric for Evaluating Machine Translation
Slide 1: Understanding METEOR Score Components

METEOR combines multiple linguistic components to evaluate text quality through a comprehensive alignment process that matches words between candidate and reference texts using exact matches, stems, synonyms and paraphrases in a weighted scoring system.

```python
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet

class METEORScorer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Weights for different match types
        self.weights = {
            'exact': 1.0,
            'stem': 0.6,
            'synonym': 0.8
        }
    
    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def stem_word(self, word):
        return self.stemmer.stem(word)

    # Example usage:
    # scorer = METEORScorer()
    # synonyms = scorer.get_synonyms("happy")
    # print(f"Synonyms of 'happy': {synonyms}")
    # stemmed = scorer.stem_word("running")
    # print(f"Stem of 'running': {stemmed}")
```

Slide 2: Token Alignment Algorithm

The alignment phase matches tokens between hypothesis and reference texts using a staged approach that first attempts exact matches, then stemmed matches, and finally synonym matches, creating a bipartite graph of matched word pairs.

```python
def align_tokens(self, hypothesis, reference):
    matches = []
    used_hyp = set()
    used_ref = set()
    
    # Stage 1: Exact matches
    for i, h_word in enumerate(hypothesis):
        for j, r_word in enumerate(reference):
            if j not in used_ref and i not in used_hyp:
                if h_word.lower() == r_word.lower():
                    matches.append(('exact', i, j))
                    used_hyp.add(i)
                    used_ref.add(j)
    
    # Stage 2: Stem matches
    for i, h_word in enumerate(hypothesis):
        if i not in used_hyp:
            h_stem = self.stem_word(h_word)
            for j, r_word in enumerate(reference):
                if j not in used_ref:
                    if h_stem == self.stem_word(r_word):
                        matches.append(('stem', i, j))
                        used_hyp.add(i)
                        used_ref.add(j)
    
    return matches
```

Slide 3: METEOR Score Calculation

The METEOR score combines precision, recall, and a penalty term for fragmentation, where fragmentation measures how well the matched words in the hypothesis are ordered compared to their corresponding matches in the reference.

```python
def calculate_meteor_score(self, matches, hypothesis_len, reference_len):
    # Calculate precision and recall
    match_count = len(matches)
    precision = match_count / hypothesis_len if hypothesis_len > 0 else 0
    recall = match_count / reference_len if reference_len > 0 else 0
    
    # Harmonic mean with alpha weight
    alpha = 0.9
    if precision > 0 and recall > 0:
        fmean = precision * recall / (alpha * precision + (1 - alpha) * recall)
    else:
        fmean = 0
    
    # Calculate fragmentation penalty
    chunks = self.count_chunks(matches)
    fragmentation = chunks / match_count if match_count > 0 else 0
    penalty = 0.5 * (fragmentation ** 3)
    
    # Final METEOR score
    score = fmean * (1 - penalty)
    return score

def count_chunks(self, matches):
    if not matches:
        return 0
    chunks = 1
    prev_ref_idx = matches[0][2]
    
    for _, _, ref_idx in matches[1:]:
        if ref_idx != prev_ref_idx + 1:
            chunks += 1
        prev_ref_idx = ref_idx
    
    return chunks
```

Slide 4: Word Order Penalty Implementation

The word order penalty component of METEOR captures how well the sequence of matched words in the hypothesis preserves the order of their counterparts in the reference, implementing a chunk-based calculation method.

```python
def calculate_order_penalty(self, matches, hypothesis_len):
    if not matches or hypothesis_len <= 1:
        return 0.0
        
    # Sort matches by hypothesis index
    sorted_matches = sorted(matches, key=lambda x: x[1])
    
    # Calculate number of crossing alignments
    crossings = 0
    for i in range(len(sorted_matches)-1):
        for j in range(i+1, len(sorted_matches)):
            if sorted_matches[i][2] > sorted_matches[j][2]:
                crossings += 1
    
    # Normalize crossings by possible pairs
    possible_pairs = (len(matches) * (len(matches) - 1)) / 2
    normalized_crossings = crossings / possible_pairs if possible_pairs > 0 else 0
    
    # Calculate penalty using gamma parameter
    gamma = 0.4
    penalty = gamma * (normalized_crossings ** 3)
    
    return penalty
```

Slide 5: Synonym Handling with WordNet

Advanced synonym matching leverages WordNet's semantic network to identify valid word substitutions, implementing a sophisticated matching system that considers word sense disambiguation and semantic similarity.

```python
from nltk.corpus import wordnet
import numpy as np

class SynonymHandler:
    def __init__(self):
        self.cache = {}
        
    def get_similarity_score(self, word1, word2):
        cache_key = tuple(sorted([word1, word2]))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        synsets1 = wordnet.synsets(word1)
        synsets2 = wordnet.synsets(word2)
        
        max_similarity = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                similarity = syn1.path_similarity(syn2)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
                    
        self.cache[cache_key] = max_similarity
        return max_similarity
        
    def are_synonyms(self, word1, word2, threshold=0.85):
        return self.get_similarity_score(word1, word2) >= threshold

# Example usage:
# handler = SynonymHandler()
# print(handler.are_synonyms("happy", "joyful"))  # True
# print(handler.are_synonyms("happy", "sad"))     # False
```

Slide 6: Advanced Stemming Techniques

Implementation of enhanced stemming algorithms that go beyond Porter Stemmer to handle morphological variants, incorporating Snowball stemmer and lemmatization for improved matching accuracy in the METEOR scoring process.

```python
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

class AdvancedStemmer:
    def __init__(self):
        self.snowball = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        self.custom_rules = {
            r'ies$': 'y',
            r'([^aeiou])ies$': r'\1y',
            r'([aeiou]y)s$': r'\1',
        }
        
    def process_word(self, word):
        # Combined approach using multiple techniques
        stemmed = self.snowball.stem(word)
        lemmatized = self.lemmatizer.lemmatize(word)
        
        # Apply custom rules
        processed = word.lower()
        for pattern, replacement in self.custom_rules.items():
            processed = re.sub(pattern, replacement, processed)
            
        # Return all variants for matching
        return {
            'stemmed': stemmed,
            'lemmatized': lemmatized,
            'custom': processed
        }

    def get_best_match(self, word1, word2):
        w1_forms = self.process_word(word1)
        w2_forms = self.process_word(word2)
        
        # Check all combinations for matches
        for form1 in w1_forms.values():
            for form2 in w2_forms.values():
                if form1 == form2:
                    return True, form1
        return False, None

# Example:
# stemmer = AdvancedStemmer()
# print(stemmer.process_word("running"))
# print(stemmer.get_best_match("running", "ran"))
```

Slide 7: Chunk Alignment Optimization

Advanced chunk alignment algorithm that implements dynamic programming to find optimal chunk boundaries while considering both local and global alignment scores, maximizing the overall METEOR score through efficient chunk identification.

```python
import numpy as np

class ChunkAligner:
    def __init__(self, min_chunk_size=2, max_gap=3):
        self.min_chunk_size = min_chunk_size
        self.max_gap = max_gap
        
    def find_optimal_chunks(self, matches):
        if not matches:
            return []
            
        # Sort matches by reference indices
        sorted_matches = sorted(matches, key=lambda x: x[2])
        n = len(sorted_matches)
        
        # Dynamic programming matrix
        dp = np.zeros((n + 1, n + 1))
        backtrack = np.zeros((n + 1, n + 1), dtype=int)
        
        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(i + self.min_chunk_size - 1, n + 1):
                chunk_score = self.calculate_chunk_score(
                    sorted_matches[i-1:j]
                )
                
                # Consider all possible previous chunks
                for k in range(i):
                    if self.is_valid_gap(sorted_matches, k, i-1):
                        score = dp[k][i-1] + chunk_score
                        if score > dp[i][j]:
                            dp[i][j] = score
                            backtrack[i][j] = k
        
        # Reconstruct optimal chunks
        chunks = self.reconstruct_chunks(backtrack, sorted_matches)
        return chunks
    
    def calculate_chunk_score(self, chunk_matches):
        if not chunk_matches:
            return 0
            
        # Consider contiguity and coverage
        gaps = sum(1 for i in range(len(chunk_matches)-1)
                  if chunk_matches[i+1][2] - chunk_matches[i][2] > 1)
        
        coverage = len(chunk_matches)
        contiguity = 1.0 / (1.0 + gaps)
        
        return coverage * contiguity
    
    def is_valid_gap(self, matches, start_idx, end_idx):
        if start_idx >= end_idx:
            return True
        gap_size = matches[end_idx][2] - matches[start_idx][2]
        return gap_size <= self.max_gap

# Example usage:
# aligner = ChunkAligner()
# chunks = aligner.find_optimal_chunks(matches)
```

Slide 8: METEOR Implementation for Multiple References

Extended METEOR implementation that handles multiple reference translations, computing scores against each reference and selecting the highest score while maintaining proper normalization and weighting schemes.

```python
class MultiReferenceMETEOR:
    def __init__(self):
        self.base_scorer = METEORScorer()
        
    def compute_multi_reference_score(self, hypothesis, references):
        scores = []
        alignments = []
        
        for reference in references:
            # Calculate score for each reference
            matches = self.base_scorer.align_tokens(hypothesis, reference)
            score = self.base_scorer.calculate_meteor_score(
                matches,
                len(hypothesis),
                len(reference)
            )
            
            scores.append(score)
            alignments.append(matches)
            
        # Select best scoring reference
        best_score_idx = np.argmax(scores)
        best_score = scores[best_score_idx]
        best_alignment = alignments[best_score_idx]
        
        return {
            'score': best_score,
            'reference_used': best_score_idx,
            'alignment': best_alignment,
            'all_scores': scores
        }

    def compute_system_score(self, hypotheses, reference_sets):
        if len(hypotheses) != len(reference_sets):
            raise ValueError("Number of hypotheses must match reference sets")
            
        segment_scores = []
        for hyp, refs in zip(hypotheses, reference_sets):
            result = self.compute_multi_reference_score(hyp, refs)
            segment_scores.append(result['score'])
            
        return {
            'system_score': np.mean(segment_scores),
            'segment_scores': segment_scores,
            'std_dev': np.std(segment_scores)
        }

# Example:
# scorer = MultiReferenceMETEOR()
# hyp = ["The", "cat", "sits", "on", "mat"]
# refs = [
#     ["A", "cat", "is", "on", "the", "mat"],
#     ["The", "cat", "sits", "on", "the", "rug"]
# ]
# result = scorer.compute_multi_reference_score(hyp, refs)
```

Slide 9: Parametric Weighting System

Implementation of a sophisticated weighting mechanism that allows dynamic adjustment of component weights based on language pairs and domain-specific requirements, using a parametric approach for optimization.

```python
class ParametricMETEORWeights:
    def __init__(self, language_pair=None):
        self.default_weights = {
            'exact': 1.0,
            'stem': 0.6,
            'synonym': 0.8,
            'paraphrase': 0.6
        }
        
        self.language_specific_weights = {
            'en-fr': {'exact': 1.0, 'stem': 0.5, 'synonym': 0.85, 'paraphrase': 0.55},
            'en-de': {'exact': 1.0, 'stem': 0.65, 'synonym': 0.75, 'paraphrase': 0.55},
            'en-es': {'exact': 1.0, 'stem': 0.55, 'synonym': 0.8, 'paraphrase': 0.6}
        }
        
        self.weights = (self.language_specific_weights.get(language_pair) 
                       or self.default_weights)
        
    def optimize_weights(self, training_data, iterations=100):
        best_correlation = -1
        best_weights = self.weights.copy()
        
        for _ in range(iterations):
            # Randomly perturb weights
            candidate_weights = {
                k: max(0, min(1, v + np.random.normal(0, 0.1)))
                for k, v in self.weights.items()
            }
            
            # Normalize weights
            total = sum(candidate_weights.values())
            candidate_weights = {k: v/total for k, v in candidate_weights.items()}
            
            # Calculate correlation with human scores
            correlation = self.calculate_correlation(candidate_weights, training_data)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_weights = candidate_weights.copy()
        
        return best_weights, best_correlation
    
    def calculate_correlation(self, weights, training_data):
        meteor_scores = []
        human_scores = []
        
        for item in training_data:
            score = self.calculate_weighted_score(
                item['matches'],
                weights,
                item['hyp_len'],
                item['ref_len']
            )
            meteor_scores.append(score)
            human_scores.append(item['human_score'])
            
        return np.corrcoef(meteor_scores, human_scores)[0,1]

# Example usage:
# weights = ParametricMETEORWeights('en-fr')
# optimized_weights, correlation = weights.optimize_weights(training_data)
```

Slide 10: Real-world Application - Translation Evaluation

Practical implementation of METEOR for evaluating machine translation quality in a production environment, including preprocessing, scoring, and statistical analysis of results.

```python
class TranslationEvaluator:
    def __init__(self):
        self.meteor = MultiReferenceMETEOR()
        self.preprocessor = TextPreprocessor()
        
    def evaluate_translation_batch(self, source_texts, mt_outputs, reference_translations):
        results = []
        
        for src, mt, refs in zip(source_texts, mt_outputs, reference_translations):
            # Preprocess all texts
            mt_tokens = self.preprocessor.normalize(mt)
            ref_tokens_list = [self.preprocessor.normalize(ref) for ref in refs]
            
            # Calculate METEOR score
            meteor_result = self.meteor.compute_multi_reference_score(
                mt_tokens, ref_tokens_list
            )
            
            # Store detailed results
            results.append({
                'source': src,
                'mt_output': mt,
                'references': refs,
                'meteor_score': meteor_result['score'],
                'best_reference_idx': meteor_result['reference_used'],
                'alignment_details': meteor_result['alignment']
            })
        
        # Calculate aggregate statistics
        scores = [r['meteor_score'] for r in results]
        statistics = {
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_dev': np.std(scores),
            'min_score': min(scores),
            'max_score': max(scores)
        }
        
        return results, statistics

class TextPreprocessor:
    def normalize(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        return tokens

# Example usage:
# evaluator = TranslationEvaluator()
# results, stats = evaluator.evaluate_translation_batch(
#     source_texts=['Source 1', 'Source 2'],
#     mt_outputs=['MT 1', 'MT 2'],
#     reference_translations=[['Ref 1A', 'Ref 1B'], ['Ref 2A', 'Ref 2B']]
# )
```

Slide 11: Statistical Significance Testing

Implementation of statistical significance testing for METEOR scores to determine if differences between machine translation systems are meaningful, using bootstrap resampling and confidence intervals.

```python
class METEORSignificanceTester:
    def __init__(self, n_bootstrap=1000, confidence_level=0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
    def compare_systems(self, scores_system1, scores_system2):
        # Calculate original difference
        original_diff = np.mean(scores_system1) - np.mean(scores_system2)
        
        # Bootstrap resampling
        n_samples = len(scores_system1)
        bootstrap_diffs = []
        
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples)
            sample1 = [scores_system1[i] for i in indices]
            sample2 = [scores_system2[i] for i in indices]
            
            # Calculate difference for this bootstrap sample
            diff = np.mean(sample1) - np.mean(sample2)
            bootstrap_diffs.append(diff)
            
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha * 100 / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 - alpha * 100 / 2)
        
        # Calculate p-value
        p_value = sum(1 for d in bootstrap_diffs if d <= 0) / self.n_bootstrap
        if p_value > 0.5:
            p_value = 1 - p_value
            
        return {
            'difference': original_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'p_value': p_value,
            'significant': 0 not in (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_diffs
        }

# Example usage:
# tester = METEORSignificanceTester()
# result = tester.compare_systems(
#     scores_system1=[0.75, 0.82, 0.78, 0.80],
#     scores_system2=[0.70, 0.75, 0.72, 0.73]
# )
```

Slide 12: Language-Specific Customization

Implementation of language-specific adaptations for METEOR, including custom tokenization rules, morphological analysis, and specialized synonym handling for different language pairs.

```python
class LanguageSpecificMETEOR:
    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.morphology_analyzers = self._init_morphology()
        self.synonym_handlers = self._init_synonyms()
        
    def _init_morphology(self):
        return {
            'ar': ArabicMorphologyAnalyzer(),
            'zh': ChineseMorphologyAnalyzer(),
            'ja': JapaneseMorphologyAnalyzer()
        }
        
    def _init_synonyms(self):
        handlers = {}
        for lang in [self.source_lang, self.target_lang]:
            if lang == 'en':
                handlers[lang] = WordNetSynonymHandler()
            elif lang in ['fr', 'es', 'de']:
                handlers[lang] = EuroWordNetHandler(lang)
            else:
                handlers[lang] = BasicSynonymHandler(lang)
        return handlers
        
    def analyze_morphology(self, word, language):
        if language in self.morphology_analyzers:
            return self.morphology_analyzers[language].analyze(word)
        return [word]
        
    def get_synonyms(self, word, language):
        if language in self.synonym_handlers:
            return self.synonym_handlers[language].get_synonyms(word)
        return set([word])
        
    def calculate_score(self, hypothesis, reference):
        # Language-specific preprocessing
        hyp_processed = self.preprocess_text(hypothesis, self.target_lang)
        ref_processed = self.preprocess_text(reference, self.target_lang)
        
        # Calculate matches considering language-specific features
        matches = self.find_matches(hyp_processed, ref_processed)
        
        return self.compute_final_score(matches, len(hyp_processed), len(ref_processed))

# Example usage:
# meteor = LanguageSpecificMETEOR('en', 'fr')
# score = meteor.calculate_score(
#     "The cat sits on the mat",
#     "Le chat est assis sur le tapis"
# )
```

Slide 13: Integration with Machine Learning Models

Implementation of a neural network-based approach to optimize METEOR parameters using supervised learning from human judgments, incorporating deep learning techniques for improved correlation with human assessments.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralMETEOR(nn.Module):
    def __init__(self, feature_dim=8):
        super(NeuralMETEOR, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return self.feature_extractor(features)

class MLEnhancedMETEOR:
    def __init__(self):
        self.base_meteor = METEORScorer()
        self.model = NeuralMETEOR()
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def extract_features(self, hypothesis, reference):
        # Extract basic METEOR features
        matches = self.base_meteor.align_tokens(hypothesis, reference)
        basic_score = self.base_meteor.calculate_meteor_score(
            matches, len(hypothesis), len(reference)
        )
        
        # Additional features
        features = torch.tensor([
            basic_score,
            len(matches) / len(hypothesis),  # Precision
            len(matches) / len(reference),   # Recall
            self.base_meteor.calculate_order_penalty(matches, len(hypothesis)),
            len(hypothesis) / len(reference),  # Length ratio
            len(set(hypothesis) & set(reference)) / len(set(hypothesis) | set(reference)),  # Jaccard
            len(matches) / max(len(hypothesis), len(reference)),  # Coverage
            float(bool(matches))  # Binary match indicator
        ])
        
        return features.float()
        
    def train(self, train_data, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for item in train_data:
                features = self.extract_features(
                    item['hypothesis'],
                    item['reference']
                )
                
                self.optimizer.zero_grad()
                prediction = self.model(features)
                loss = self.criterion(prediction, torch.tensor([item['human_score']]))
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data)}')

# Example usage:
# ml_meteor = MLEnhancedMETEOR()
# train_data = [
#     {'hypothesis': ['The', 'cat'], 'reference': ['A', 'cat'], 'human_score': 0.8},
#     # ... more training examples
# ]
# ml_meteor.train(train_data)
```

Slide 14: Advanced Report Generation

Implementation of comprehensive METEOR evaluation reports that include detailed analysis, visualizations, and statistical comparisons across multiple translation systems and language pairs.

```python
class METEORReportGenerator:
    def __init__(self):
        self.significance_tester = METEORSignificanceTester()
        
    def generate_system_comparison_report(self, systems_scores, system_names):
        report = {
            'systems': {},
            'pairwise_comparisons': [],
            'aggregate_statistics': {},
            'score_distributions': {}
        }
        
        # Calculate system-level statistics
        for name, scores in zip(system_names, systems_scores):
            report['systems'][name] = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'ci_95': self._calculate_confidence_interval(scores),
                'sample_size': len(scores)
            }
            
        # Pairwise significance testing
        for i in range(len(system_names)):
            for j in range(i + 1, len(system_names)):
                comparison = self.significance_tester.compare_systems(
                    systems_scores[i],
                    systems_scores[j]
                )
                
                report['pairwise_comparisons'].append({
                    'system1': system_names[i],
                    'system2': system_names[j],
                    'difference': comparison['difference'],
                    'p_value': comparison['p_value'],
                    'significant': comparison['significant'],
                    'ci': comparison['confidence_interval']
                })
        
        return report
    
    def _calculate_confidence_interval(self, scores, confidence=0.95):
        mean = np.mean(scores)
        std_err = np.std(scores) / np.sqrt(len(scores))
        t_value = self._get_t_value(len(scores), confidence)
        margin = t_value * std_err
        return (mean - margin, mean + margin)
    
    def _get_t_value(self, df, confidence):
        from scipy import stats
        alpha = 1 - confidence
        return stats.t.ppf(1 - alpha/2, df-1)

# Example usage:
# generator = METEORReportGenerator()
# report = generator.generate_system_comparison_report(
#     systems_scores=[
#         [0.75, 0.82, 0.78],  # System 1 scores
#         [0.70, 0.75, 0.72],  # System 2 scores
#     ],
#     system_names=['System A', 'System B']
# )
```

Slide 15: Additional Resources

*   "METEOR: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments"
    *   [https://arxiv.org/abs/1609.08756](https://arxiv.org/abs/1609.08756)
*   "Improved Statistical Machine Translation Using METEOR and Enhanced Word Alignment"
    *   [https://dl.acm.org/doi/10.5555/1626355.1626389](https://dl.acm.org/doi/10.5555/1626355.1626389)
*   "A Systematic Comparison of METEOR and Human Judgments"
    *   [https://aclanthology.org/P15-2073](https://aclanthology.org/P15-2073)
*   "METEOR Universal: Language Specific Translation Evaluation for Any Target Language"
    *   [https://aclanthology.org/W14-3348](https://aclanthology.org/W14-3348)

Suggested searches for additional information:

*   Google Scholar: "METEOR metric machine translation evaluation"
*   ACL Anthology: "METEOR extensions and implementations"
*   Research Gate: "Recent improvements in METEOR scoring"
*   ArXiv: "Neural METEOR variants"

