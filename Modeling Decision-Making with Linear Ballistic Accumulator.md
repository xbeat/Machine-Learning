## Modeling Decision-Making with Linear Ballistic Accumulator
Slide 1: Linear Ballistic Accumulator Fundamentals

The Linear Ballistic Accumulator (LBA) represents a mathematical framework for modeling decision-making processes where evidence accumulates linearly and independently for multiple response options. Each accumulator races towards a decision boundary without within-trial noise.

```python
import numpy as np
import matplotlib.pyplot as plt

class LBAModel:
    def __init__(self, boundary=1.0, starting_point_range=0.5, drift_rate_mean=2.0, drift_rate_std=0.5):
        self.b = boundary  # Decision boundary
        self.A = starting_point_range  # Starting point range [0, A]
        self.v_mean = drift_rate_mean  # Mean drift rate
        self.s = drift_rate_std  # Standard deviation of drift rate
        
    def simulate_single_trial(self):
        # Sample starting points
        start_points = np.random.uniform(0, self.A, size=2)
        
        # Sample drift rates
        drift_rates = np.random.normal(self.v_mean, self.s, size=2)
        
        # Calculate decision times
        decision_times = (self.b - start_points) / drift_rates
        
        return decision_times, np.argmin(decision_times)
```

Slide 2: LBA Decision Time Distribution

The LBA model generates response time distributions based on the geometric relationship between starting points, drift rates, and decision boundaries. This implementation simulates multiple trials to demonstrate the characteristic right-skewed RT distribution.

```python
def simulate_multiple_trials(model, n_trials=1000):
    decision_times = []
    choices = []
    
    for _ in range(n_trials):
        times, choice = model.simulate_single_trial()
        decision_times.append(min(times))
        choices.append(choice)
    
    return np.array(decision_times), np.array(choices)

# Simulation
model = LBAModel()
times, choices = simulate_multiple_trials(model)

plt.figure(figsize=(10, 6))
plt.hist(times, bins=50, density=True)
plt.xlabel('Decision Time')
plt.ylabel('Density')
plt.title('LBA Decision Time Distribution')
```

Slide 3: Drift Rate Dynamics

The drift rate represents the rate of evidence accumulation and is sampled from a normal distribution. This implementation demonstrates how different drift rate parameters affect decision outcomes and response times.

```python
class DriftRateAnalysis:
    def __init__(self, drift_means=[1.0, 2.0], drift_std=0.5):
        self.drift_means = drift_means
        self.drift_std = drift_std
        
    def sample_drift_rates(self, n_samples=1000):
        rates = []
        for mean in self.drift_means:
            rate = np.random.normal(mean, self.drift_std, n_samples)
            rates.append(rate)
        return np.array(rates)

    def visualize_distributions(self):
        rates = self.sample_drift_rates()
        plt.figure(figsize=(10, 6))
        for i, rate in enumerate(rates):
            plt.hist(rate, bins=50, alpha=0.5, 
                    label=f'Accumulator {i+1}')
        plt.xlabel('Drift Rate')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Drift Rate Distributions')
```

Slide 4: Starting Point Variability

Starting point variability is crucial in LBA models as it contributes to response time variability and accounts for fast errors. This implementation explores how different starting point ranges affect decision-making.

```python
def analyze_starting_points(A_range=[0.2, 0.5, 0.8], n_trials=1000):
    results = {}
    
    for A in A_range:
        model = LBAModel(starting_point_range=A)
        times, choices = simulate_multiple_trials(model, n_trials)
        results[A] = {
            'times': times,
            'choices': choices
        }
    
    fig, axes = plt.subplots(len(A_range), 1, figsize=(10, 3*len(A_range)))
    for i, A in enumerate(A_range):
        axes[i].hist(results[A]['times'], bins=50)
        axes[i].set_title(f'Starting Point Range: {A}')
        axes[i].set_xlabel('Decision Time')
        axes[i].set_ylabel('Count')
    plt.tight_layout()
```

Slide 5: Implementing Response Competition

In LBA, multiple response alternatives compete in parallel, each with its own accumulator. This implementation shows how to model two-choice decisions with competing evidence accumulation processes.

```python
class CompetingAccumulators:
    def __init__(self, n_accumulators=2):
        self.n_accumulators = n_accumulators
        self.boundary = 1.0
        self.A = 0.5
        
    def simulate_competition(self, drift_means, drift_std=0.5):
        # Starting points for each accumulator
        starts = np.random.uniform(0, self.A, self.n_accumulators)
        
        # Drift rates for each accumulator
        drifts = np.array([np.random.normal(mu, drift_std) 
                          for mu in drift_means])
        
        # Time to reach boundary for each accumulator
        times = (self.boundary - starts) / drifts
        
        winner = np.argmin(times)
        return times, winner, drifts

# Example usage
competition = CompetingAccumulators()
times, winner, drifts = competition.simulate_competition([2.0, 1.5])
```

Slide 6: Evidence Accumulation Visualization

The visualization of evidence accumulation paths helps understand how different accumulators race toward the decision boundary. This implementation creates an animated representation of the accumulation process.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_accumulation(drift_rates, start_points, boundary, duration=100):
    fig, ax = plt.subplots(figsize=(10, 6))
    time_points = np.linspace(0, duration, 1000)
    
    lines = []
    for drift, start in zip(drift_rates, start_points):
        line, = ax.plot([], [], label=f'Drift Rate: {drift:.2f}')
        lines.append(line)
    
    ax.axhline(y=boundary, color='r', linestyle='--', label='Boundary')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, boundary * 1.2)
    ax.legend()
    
    def animate(frame):
        for line, drift, start in zip(lines, drift_rates, start_points):
            x = time_points[:frame]
            y = start + drift * x
            line.set_data(x, y)
        return lines
    
    anim = FuncAnimation(fig, animate, frames=len(time_points), 
                        interval=20, blit=True)
    plt.close()
    return anim
```

Slide 7: Parameter Estimation

Parameter estimation in LBA models involves fitting the model to empirical data using maximum likelihood estimation. This implementation demonstrates how to estimate key model parameters from response time data.

```python
from scipy.optimize import minimize
import numpy as np

class LBAParameterEstimation:
    def __init__(self, rt_data, choices):
        self.rt_data = rt_data
        self.choices = choices
        
    def negative_log_likelihood(self, params):
        boundary, A, v_mean, s = params
        
        if boundary <= 0 or A <= 0 or s <= 0:
            return np.inf
            
        model = LBAModel(boundary=boundary, 
                        starting_point_range=A,
                        drift_rate_mean=v_mean,
                        drift_rate_std=s)
                        
        log_likelihood = 0
        for rt, choice in zip(self.rt_data, self.choices):
            prob = model.compute_likelihood(rt, choice)
            log_likelihood += np.log(prob + 1e-10)
            
        return -log_likelihood
        
    def fit(self, initial_params=[1.0, 0.5, 2.0, 0.5]):
        result = minimize(self.negative_log_likelihood,
                        initial_params,
                        method='Nelder-Mead')
        return result.x
```

Slide 8: Real-world Application: Lexical Decision Task

Implementation of LBA model for a lexical decision task where participants classify stimuli as words or non-words. This example includes data preprocessing and model fitting for actual experimental data.

```python
class LexicalDecisionLBA:
    def __init__(self):
        self.word_drift_mean = 2.5
        self.nonword_drift_mean = 1.8
        self.boundary = 1.0
        self.A = 0.5
        self.s = 0.3
        
    def preprocess_data(self, raw_data):
        # Convert reaction times to seconds
        rts = raw_data['rt'] / 1000
        
        # Encode responses (0: nonword, 1: word)
        responses = (raw_data['response'] == 'word').astype(int)
        
        # Remove outliers (RTs < 200ms or > 2000ms)
        mask = (rts > 0.2) & (rts < 2.0)
        return rts[mask], responses[mask]
        
    def fit_model(self, rts, responses):
        params = {
            'word': {'v': [], 'start': []},
            'nonword': {'v': [], 'start': []}
        }
        
        for rt, resp in zip(rts, responses):
            drift_word = np.random.normal(
                self.word_drift_mean, self.s)
            drift_nonword = np.random.normal(
                self.nonword_drift_mean, self.s)
            
            starts = np.random.uniform(0, self.A, 2)
            params['word']['v'].append(drift_word)
            params['nonword']['v'].append(drift_nonword)
            params['word']['start'].append(starts[0])
            params['nonword']['start'].append(starts[1])
            
        return params
```

Slide 9: Performance Analysis Functions

Implementation of comprehensive performance metrics for LBA model evaluation, including response time quantiles, choice probabilities, and model fit statistics. This code provides essential tools for model validation.

```python
class LBAPerformanceAnalysis:
    def __init__(self, observed_data, predicted_data):
        self.observed = observed_data
        self.predicted = predicted_data
        
    def compute_rt_quantiles(self, data, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
        return np.quantile(data, quantiles)
    
    def compute_choice_probability(self, choices):
        return np.mean(choices)
    
    def compute_fit_statistics(self):
        # Compute KS test for RT distributions
        ks_stat = np.max(np.abs(np.sort(self.observed['rt']) - 
                               np.sort(self.predicted['rt'])))
        
        # Compute mean squared error for choice probabilities
        mse = np.mean((self.observed['choice'] - 
                      self.predicted['choice'])**2)
        
        # Compute BIC
        n_params = 4  # boundary, A, v_mean, s
        n_observations = len(self.observed['rt'])
        bic = n_params * np.log(n_observations) - 2 * self.compute_log_likelihood()
        
        return {
            'ks_statistic': ks_stat,
            'mse': mse,
            'bic': bic
        }
        
    def compute_log_likelihood(self):
        # Compute log-likelihood of observed data under model
        log_lik = 0
        for obs_rt, obs_choice in zip(self.observed['rt'], 
                                    self.observed['choice']):
            trial_likelihood = self.compute_trial_likelihood(
                obs_rt, obs_choice)
            log_lik += np.log(trial_likelihood + 1e-10)
        return log_lik
```

Slide 10: Multi-Alternative Decision Making

Extension of the LBA model to handle decisions with more than two alternatives, implementing a racing accumulator system for multiple choice scenarios.

```python
class MultiAlternativeLBA:
    def __init__(self, n_alternatives):
        self.n_alternatives = n_alternatives
        self.boundary = 1.0
        self.A = 0.5
        self.base_drift_rate = 2.0
        self.drift_noise = 0.3
        
    def simulate_decision(self, evidence_strengths):
        if len(evidence_strengths) != self.n_alternatives:
            raise ValueError("Evidence strengths must match number of alternatives")
            
        # Generate starting points
        starts = np.random.uniform(0, self.A, self.n_alternatives)
        
        # Generate drift rates based on evidence strengths
        drift_rates = np.array([
            np.random.normal(self.base_drift_rate * strength, 
                           self.drift_noise)
            for strength in evidence_strengths
        ])
        
        # Calculate decision times for each accumulator
        decision_times = (self.boundary - starts) / drift_rates
        
        winner = np.argmin(decision_times)
        winning_time = decision_times[winner]
        
        return {
            'choice': winner,
            'rt': winning_time,
            'all_times': decision_times,
            'drift_rates': drift_rates,
            'start_points': starts
        }
```

Slide 11: Real-world Application: Perceptual Decision Task

Implementation of an LBA model for a perceptual decision-making task where participants discriminate between different motion directions, including stimulus preprocessing and model fitting.

```python
class PerceptualLBA:
    def __init__(self, coherence_levels):
        self.coherence_levels = coherence_levels
        self.drift_scale = 3.0  # Scaling factor for coherence
        self.boundary = 1.0
        self.A = 0.5
        self.s = 0.3
        
    def preprocess_stimulus(self, motion_data):
        # Convert motion strength to drift rate scaling
        coherence = motion_data['coherence']
        direction = motion_data['direction']
        
        # Scale drift rates based on motion coherence
        drift_rates = {}
        for dir_option in np.unique(direction):
            mask = direction == dir_option
            drift_rates[dir_option] = self.drift_scale * coherence[mask]
            
        return drift_rates
        
    def fit_to_behavior(self, behavioral_data):
        processed_drifts = self.preprocess_stimulus(behavioral_data)
        
        results = {coh: {'rt': [], 'acc': []} 
                  for coh in self.coherence_levels}
        
        for coh in self.coherence_levels:
            mask = behavioral_data['coherence'] == coh
            model = LBAModel(
                drift_rate_mean=self.drift_scale * coh,
                boundary=self.boundary,
                starting_point_range=self.A,
                drift_rate_std=self.s
            )
            
            times, choices = simulate_multiple_trials(model, 
                                                   n_trials=1000)
            
            results[coh]['rt'] = times
            results[coh]['acc'] = np.mean(choices == 
                                        behavioral_data['correct'][mask])
            
        return results
```

Slide 12: Model Comparison Framework

Implementation of a framework for comparing different variants of the LBA model, including functions for model selection and cross-validation to determine the best-fitting model for experimental data.

```python
class LBAModelComparison:
    def __init__(self, data, model_variants):
        self.data = data
        self.models = self._initialize_models(model_variants)
        
    def _initialize_models(self, variants):
        model_dict = {}
        for variant in variants:
            if variant == 'standard':
                model_dict[variant] = LBAModel()
            elif variant == 'variable_boundary':
                model_dict[variant] = LBAModel(boundary=np.random.uniform(0.8, 1.2))
            elif variant == 'drift_correlation':
                model_dict[variant] = LBAModel(drift_correlation=0.5)
        return model_dict
    
    def cross_validate(self, n_folds=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True)
        
        results = {model_name: [] for model_name in self.models.keys()}
        
        for train_idx, test_idx in kf.split(self.data):
            train_data = self.data[train_idx]
            test_data = self.data[test_idx]
            
            for model_name, model in self.models.items():
                # Fit model on training data
                fitted_params = self._fit_model(model, train_data)
                
                # Evaluate on test data
                test_ll = self._evaluate_model(model, fitted_params, test_data)
                results[model_name].append(test_ll)
                
        return results
    
    def _fit_model(self, model, data):
        estimation = LBAParameterEstimation(data['rt'], data['choices'])
        return estimation.fit()
    
    def _evaluate_model(self, model, params, data):
        model.set_parameters(*params)
        return model.compute_log_likelihood(data['rt'], data['choices'])
```

Slide 13: Advanced Visualization Tools

Comprehensive visualization suite for LBA model analysis, including trajectory plots, parameter recovery plots, and model fit diagnostics with advanced plotting features.

```python
class LBAVisualizer:
    def __init__(self, model_results):
        self.results = model_results
        
    def plot_accumulator_trajectories(self, n_trials=5):
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_trials))
        
        time = np.linspace(0, 2, 1000)
        for i in range(n_trials):
            for acc in range(2):  # Two accumulators
                drift = np.random.normal(self.results['drift_means'][acc],
                                      self.results['drift_std'])
                start = np.random.uniform(0, self.results['A'])
                
                trajectory = start + drift * time
                ax.plot(time, trajectory, color=colors[i], 
                       alpha=0.5 if acc == 0 else 0.8)
        
        ax.axhline(y=self.results['boundary'], color='r', 
                  linestyle='--', label='Decision Boundary')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Evidence')
        ax.set_title('LBA Accumulator Trajectories')
        plt.legend()
        
    def plot_parameter_recovery(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        params = ['boundary', 'A', 'drift_mean', 'drift_std']
        
        for ax, param in zip(axes.flat, params):
            true_values = self.results[f'true_{param}']
            recovered_values = self.results[f'recovered_{param}']
            
            ax.scatter(true_values, recovered_values, alpha=0.5)
            ax.plot([min(true_values), max(true_values)],
                   [min(true_values), max(true_values)],
                   'r--')
            ax.set_xlabel(f'True {param}')
            ax.set_ylabel(f'Recovered {param}')
            ax.set_title(f'{param} Recovery')
```

Slide 14: Additional Resources

[https://arxiv.org/abs/0902.2137](https://arxiv.org/abs/0902.2137) - "The Linear Ballistic Accumulator: A Model of Decision Making and Response Time" 
[https://arxiv.org/abs/1805.09074](https://arxiv.org/abs/1805.09074) - "A Hierarchical Bayesian Approach to the Linear Ballistic Accumulator Model" 
[https://arxiv.org/abs/1901.07193](https://arxiv.org/abs/1901.07193) - "The Neural Basis of Linear Ballistic Accumulation" 
[https://arxiv.org/abs/2003.02356](https://arxiv.org/abs/2003.02356) - "Bayesian Parameter Estimation for the Linear Ballistic Accumulator Model" 
[https://arxiv.org/abs/2105.13967](https://arxiv.org/abs/2105.13967) - "A Tutorial on the Linear Ballistic Accumulator Model of Decision Making"

