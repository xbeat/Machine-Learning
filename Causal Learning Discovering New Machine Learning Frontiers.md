## Causal Learning Discovering New Machine Learning Frontiers
Slide 1: Understanding Causal Learning Fundamentals

Causal learning extends beyond traditional correlation-based machine learning by explicitly modeling cause-and-effect relationships between variables. This approach enables models to understand interventions and counterfactuals, leading to more robust predictions and interpretable decision-making capabilities in complex systems.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

class CausalModel:
    def __init__(self):
        self.model = LinearRegression()
        self.causal_graph = {}
    
    def add_causal_relation(self, cause, effect):
        if cause not in self.causal_graph:
            self.causal_graph[cause] = []
        self.causal_graph[cause].append(effect)
    
    def fit(self, X, y):
        # Fit considering causal structure
        self.model.fit(X, y)
        
    def predict_intervention(self, X, intervention_var, value):
        # Predict effect of intervention
        X_modified = X.copy()
        X_modified[:, intervention_var] = value
        return self.model.predict(X_modified)

# Example usage
X = np.random.randn(1000, 3)  # Features
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(1000)*0.1  # Target

model = CausalModel()
model.add_causal_relation(0, 2)  # Feature 0 causes Feature 2
model.fit(X, y)
```

Slide 2: Structural Causal Models (SCM)

Structural Causal Models provide a mathematical framework for representing causal relationships through directed acyclic graphs and structural equations. SCMs enable us to model both observational and interventional distributions, making them powerful tools for causal inference.

```python
import networkx as nx
import matplotlib.pyplot as plt

class StructuralCausalModel:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.equations = {}
        
    def add_variable(self, name):
        self.graph.add_node(name)
        
    def add_causation(self, cause, effect, equation):
        self.graph.add_edge(cause, effect)
        self.equations[(cause, effect)] = equation
        
    def visualize(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, 
                node_color='lightblue', 
                node_size=1500, 
                arrowsize=20)
        plt.show()

# Example usage
scm = StructuralCausalModel()
scm.add_variable('Rain')
scm.add_variable('Sprinkler')
scm.add_variable('WetGrass')

scm.add_causation('Rain', 'WetGrass', 
                  lambda x: 0.9 if x else 0.1)
scm.add_causation('Sprinkler', 'WetGrass', 
                  lambda x: 0.8 if x else 0.2)
```

Slide 3: Causal Discovery Algorithms

Causal discovery algorithms automatically identify potential causal relationships from observational data using statistical independence tests and structural constraints. This implementation demonstrates the PC algorithm, a fundamental approach to learning causal structures.

```python
import numpy as np
from scipy.stats import pearsonr

class PCAlgorithm:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def independence_test(self, x, y, conditioning_set=None):
        if conditioning_set is None:
            corr, p_value = pearsonr(x, y)
            return p_value > self.alpha
        
        # Partial correlation for conditional independence
        residuals_x = x
        residuals_y = y
        for z in conditioning_set:
            residuals_x = self._residualize(residuals_x, z)
            residuals_y = self._residualize(residuals_y, z)
        corr, p_value = pearsonr(residuals_x, residuals_y)
        return p_value > self.alpha
    
    def _residualize(self, target, predictor):
        reg = LinearRegression().fit(predictor.reshape(-1, 1), target)
        return target - reg.predict(predictor.reshape(-1, 1))
    
    def find_causal_graph(self, data):
        n_vars = data.shape[1]
        adjacency_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Phase I: Learn skeleton
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if self.independence_test(data[:, i], data[:, j]):
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 0
                    
        return adjacency_matrix

# Example usage
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)
X[:, 2] = 0.7 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.1

pc = PCAlgorithm()
causal_graph = pc.find_causal_graph(X)
```

Slide 4: Dealing with Confounders in Causal Learning

Confounding variables can create spurious correlations and mislead traditional machine learning models. This implementation demonstrates how to identify and adjust for confounders using backdoor adjustment, a fundamental technique in causal inference.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ConfounderAdjustment:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def backdoor_adjustment(self, treatment, outcome, confounder):
        # Standardize variables
        t = self.scaler.fit_transform(treatment.reshape(-1, 1))
        y = self.scaler.fit_transform(outcome.reshape(-1, 1))
        c = self.scaler.fit_transform(confounder.reshape(-1, 1))
        
        # Stratify by confounder
        strata = np.quantile(c, q=[0.25, 0.5, 0.75])
        ate = 0
        
        for i in range(len(strata) + 1):
            if i == 0:
                mask = c <= strata[i]
            elif i == len(strata):
                mask = c > strata[i-1]
            else:
                mask = (c > strata[i-1]) & (c <= strata[i])
                
            # Calculate stratum-specific effect
            effect = np.mean(y[mask & (t > 0)]) - np.mean(y[mask & (t <= 0)])
            weight = np.mean(mask)
            ate += effect * weight
            
        return ate

# Example usage
np.random.seed(42)
n_samples = 1000

# Generate synthetic data with confounding
confounder = np.random.normal(0, 1, n_samples)
treatment = 0.5 * confounder + np.random.normal(0, 0.5, n_samples)
outcome = 0.7 * treatment + 0.3 * confounder + np.random.normal(0, 0.1, n_samples)

model = ConfounderAdjustment()
causal_effect = model.backdoor_adjustment(treatment, outcome, confounder)
print(f"Estimated causal effect: {causal_effect:.3f}")
```

Slide 5: Interventional Learning with Do-Calculus

Do-calculus provides a formal mathematical framework for reasoning about interventions in causal systems. This implementation shows how to estimate causal effects using the do-operator, which simulates experimental interventions in observational data.

```python
class DoCalculus:
    def __init__(self):
        self.causal_graph = {}
        self.conditional_probs = {}
        
    def add_causal_relation(self, cause, effect, probability_func):
        if cause not in self.causal_graph:
            self.causal_graph[cause] = []
        self.causal_graph[cause].append(effect)
        self.conditional_probs[(cause, effect)] = probability_func
        
    def do_intervention(self, intervention_var, value, target_var, observed_data):
        # Implementation of do(X = x)
        modified_data = observed_data.copy()
        
        # Set intervention variable to specified value
        modified_data[intervention_var] = value
        
        # Remove incoming edges to intervention variable
        parents = self._get_parents(intervention_var)
        for parent in parents:
            self.causal_graph[parent].remove(intervention_var)
            
        # Compute effect on target variable
        result = self._compute_intervention_effect(
            modified_data, intervention_var, target_var)
            
        # Restore graph structure
        for parent in parents:
            self.causal_graph[parent].append(intervention_var)
            
        return result
    
    def _get_parents(self, variable):
        return [v for v, children in self.causal_graph.items() 
                if variable in children]
        
    def _compute_intervention_effect(self, data, intervention_var, target_var):
        # Simplified computation of intervention effect
        if target_var in self.causal_graph.get(intervention_var, []):
            prob_func = self.conditional_probs[(intervention_var, target_var)]
            return prob_func(data[intervention_var])
        return None

# Example usage
import pandas as pd

# Create synthetic data
data = pd.DataFrame({
    'treatment': np.random.binomial(1, 0.5, 1000),
    'age': np.random.normal(50, 10, 1000),
    'outcome': np.random.normal(0, 1, 1000)
})

do_calc = DoCalculus()
do_calc.add_causal_relation('treatment', 'outcome', 
                           lambda x: 2*x + np.random.normal(0, 0.1))
do_calc.add_causal_relation('age', 'treatment', 
                           lambda x: x > 50)

# Estimate effect of setting treatment to 1
effect = do_calc.do_intervention('treatment', 1, 'outcome', data)
```

Slide 6: Counterfactual Reasoning

Counterfactual reasoning extends beyond interventions by asking "what if" questions about past events. This implementation demonstrates how to perform counterfactual inference using structural equations and abduction-action-prediction steps.

```python
import numpy as np
from scipy.optimize import minimize

class CounterfactualModel:
    def __init__(self):
        self.structural_equations = {}
        self.noise_distributions = {}
        
    def add_equation(self, variable, equation, noise_dist):
        self.structural_equations[variable] = equation
        self.noise_distributions[variable] = noise_dist
        
    def abduction(self, observed_data):
        # Step 1: Infer exogenous variables
        exogenous_vars = {}
        for var, equation in self.structural_equations.items():
            if var in observed_data:
                noise = self._infer_noise(equation, observed_data, var)
                exogenous_vars[var] = noise
        return exogenous_vars
    
    def action(self, counterfactual_intervention):
        # Step 2: Modify model with intervention
        modified_equations = self.structural_equations.copy()
        for var, value in counterfactual_intervention.items():
            modified_equations[var] = lambda x, u: value
        return modified_equations
    
    def prediction(self, exogenous_vars, modified_equations):
        # Step 3: Compute counterfactual predictions
        result = {}
        for var, equation in modified_equations.items():
            result[var] = equation(result, exogenous_vars[var])
        return result
    
    def _infer_noise(self, equation, observed, variable):
        def objective(noise):
            return (equation(observed, noise) - observed[variable])**2
        
        result = minimize(objective, x0=0)
        return result.x[0]

# Example usage
model = CounterfactualModel()

# Define structural equations
def treatment_eq(data, noise):
    return 0.3 * data.get('age', 0) + noise

def outcome_eq(data, noise):
    return 2 * data.get('treatment', 0) + 0.5 * noise

model.add_equation('treatment', treatment_eq, 
                  lambda: np.random.normal(0, 1))
model.add_equation('outcome', outcome_eq, 
                  lambda: np.random.normal(0, 0.5))

# Observed data
observed = {'age': 45, 'treatment': 1, 'outcome': 2.5}

# Perform counterfactual inference
exogenous = model.abduction(observed)
modified_eqs = model.action({'treatment': 0})
counterfactual = model.prediction(exogenous, modified_eqs)
```

Slide 7: Source Code for Identification of Causal Effects

Causal effect identification determines whether a causal effect can be estimated from observational data. This implementation provides tools for testing identifiability using do-calculus rules and graphical criteria.

```python
import networkx as nx
from itertools import combinations

class CausalIdentification:
    def __init__(self, graph):
        self.graph = nx.DiGraph(graph)
        
    def check_backdoor_criterion(self, treatment, outcome, adjustment_set):
        # Verify backdoor criterion for causal identification
        modified_graph = self.graph.copy()
        
        # Remove outgoing edges from treatment
        treatment_outgoing = list(self.graph.out_edges(treatment))
        modified_graph.remove_edges_from(treatment_outgoing)
        
        # Check if adjustment set blocks all backdoor paths
        paths = list(nx.all_simple_paths(modified_graph, 
                                       treatment, outcome))
        
        for path in paths:
            if not self._is_blocked(path, adjustment_set):
                return False
        return True
    
    def find_minimal_adjustment_set(self, treatment, outcome):
        nodes = set(self.graph.nodes()) - {treatment, outcome}
        
        # Try adjustment sets of increasing size
        for size in range(len(nodes) + 1):
            for adjustment_set in combinations(nodes, size):
                if self.check_backdoor_criterion(
                    treatment, outcome, set(adjustment_set)):
                    return set(adjustment_set)
        return None
    
    def _is_blocked(self, path, conditioning_set):
        # Check if path is blocked by conditioning set
        for i in range(1, len(path) - 1):
            if path[i] in conditioning_set:
                if not self._is_collider(path[i-1], path[i], path[i+1]):
                    return True
            elif self._is_collider(path[i-1], path[i], path[i+1]):
                descendants = nx.descendants(self.graph, path[i])
                if not (descendants & conditioning_set):
                    return True
        return False
    
    def _is_collider(self, a, b, c):
        return (self.graph.has_edge(a, b) and 
                self.graph.has_edge(c, b))

# Example usage
graph = {
    'X': ['Y', 'Z'],
    'Z': ['Y'],
    'U': ['X', 'Z']
}

identifier = CausalIdentification(graph)
adjustment_set = identifier.find_minimal_adjustment_set('X', 'Y')
print(f"Minimal adjustment set: {adjustment_set}")

is_identifiable = identifier.check_backdoor_criterion(
    'X', 'Y', adjustment_set)
print(f"Causal effect is identifiable: {is_identifiable}")
```

Slide 8: Implementation of Front-door Criterion

The front-door criterion provides an alternative method for causal identification when backdoor adjustment is not possible due to unobserved confounders. This implementation demonstrates how to estimate causal effects using the front-door adjustment formula.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class FrontdoorAdjustment:
    def __init__(self):
        self.scalers = {
            'X': StandardScaler(),
            'M': StandardScaler(),
            'Y': StandardScaler()
        }
        self.stage1_model = LinearRegression()
        self.stage2_model = LinearRegression()
        
    def estimate_effect(self, X, M, Y):
        # Stage 1: Estimate effect of X on M
        X_scaled = self.scalers['X'].fit_transform(X.reshape(-1, 1))
        M_scaled = self.scalers['M'].fit_transform(M.reshape(-1, 1))
        self.stage1_model.fit(X_scaled, M_scaled)
        
        # Stage 2: Estimate effect of M on Y
        Y_scaled = self.scalers['Y'].fit_transform(Y.reshape(-1, 1))
        self.stage2_model.fit(M_scaled, Y_scaled)
        
        # Compute total effect using front-door formula
        effect = (self.stage1_model.coef_[0] * 
                 self.stage2_model.coef_[0])
        
        return effect
    
    def predict_intervention(self, X_new):
        X_scaled = self.scalers['X'].transform(X_new.reshape(-1, 1))
        M_pred = self.stage1_model.predict(X_scaled)
        M_scaled = self.scalers['M'].transform(M_pred.reshape(-1, 1))
        Y_pred = self.stage2_model.predict(M_scaled)
        return self.scalers['Y'].inverse_transform(Y_pred)

# Example usage
np.random.seed(42)
n_samples = 1000

# Generate synthetic data with front-door structure
X = np.random.normal(0, 1, n_samples)
U = np.random.normal(0, 1, n_samples)  # Unobserved confounder
M = 0.6 * X + np.random.normal(0, 0.1, n_samples)
Y = 0.7 * M + 0.3 * U + np.random.normal(0, 0.1, n_samples)

model = FrontdoorAdjustment()
causal_effect = model.estimate_effect(X, M, Y)
print(f"Estimated causal effect: {causal_effect:.3f}")
```

Slide 9: Instrumental Variables in Causal Learning

Instrumental variables provide a powerful method for estimating causal effects in the presence of unmeasured confounding. This implementation shows how to use instrumental variables for causal inference.

```python
class InstrumentalVariableEstimator:
    def __init__(self):
        self.first_stage = LinearRegression()
        self.second_stage = LinearRegression()
        
    def fit(self, Z, X, Y):
        """
        Z: Instrumental variable
        X: Treatment variable
        Y: Outcome variable
        """
        # First stage: regress X on Z
        self.first_stage.fit(Z.reshape(-1, 1), X)
        
        # Get predicted X values
        X_pred = self.first_stage.predict(Z.reshape(-1, 1))
        
        # Second stage: regress Y on predicted X
        self.second_stage.fit(X_pred.reshape(-1, 1), Y)
        
        return self
    
    def estimate_causal_effect(self):
        # IV estimate is the coefficient from second stage
        return self.second_stage.coef_[0]
    
    def _check_instrument_strength(self, Z, X):
        # Compute F-statistic for instrument strength
        n = len(Z)
        X_pred = self.first_stage.predict(Z.reshape(-1, 1))
        residuals = X - X_pred
        r_squared = 1 - np.sum(residuals**2) / np.sum((X - np.mean(X))**2)
        f_stat = (r_squared / (1 - r_squared)) * (n - 2)
        return f_stat

# Example usage
np.random.seed(42)
n_samples = 1000

# Generate data with valid instrument
Z = np.random.normal(0, 1, n_samples)  # Instrument
U = np.random.normal(0, 1, n_samples)  # Unobserved confounder
X = 0.5 * Z + 0.3 * U + np.random.normal(0, 0.1, n_samples)
Y = 0.7 * X + 0.4 * U + np.random.normal(0, 0.1, n_samples)

iv_estimator = InstrumentalVariableEstimator()
iv_estimator.fit(Z, X, Y)
effect = iv_estimator.estimate_causal_effect()
print(f"Estimated causal effect using IV: {effect:.3f}")
```

Slide 10: Difference-in-Differences Causal Estimation

Difference-in-Differences (DiD) is a quasi-experimental method that estimates causal effects by comparing changes over time between treatment and control groups. This implementation demonstrates how to perform DiD analysis with multiple time periods and groups.

```python
import pandas as pd
import numpy as np
from scipy import stats

class DifferenceInDifferences:
    def __init__(self):
        self.results = {}
        
    def estimate(self, data, time_col, treatment_col, outcome_col, 
                pre_period, post_period):
        # Prepare data
        pre_data = data[data[time_col] == pre_period]
        post_data = data[data[time_col] == post_period]
        
        # Calculate group means
        treated_pre = pre_data[pre_data[treatment_col] == 1][outcome_col].mean()
        treated_post = post_data[post_data[treatment_col] == 1][outcome_col].mean()
        control_pre = pre_data[pre_data[treatment_col] == 0][outcome_col].mean()
        control_post = post_data[post_data[treatment_col] == 0][outcome_col].mean()
        
        # Calculate DiD estimate
        did_estimate = (treated_post - treated_pre) - (control_post - control_pre)
        
        # Calculate standard error and confidence interval
        treated_pre_var = pre_data[pre_data[treatment_col] == 1][outcome_col].var()
        treated_post_var = post_data[post_data[treatment_col] == 1][outcome_col].var()
        control_pre_var = pre_data[pre_data[treatment_col] == 0][outcome_col].var()
        control_post_var = post_data[post_data[treatment_col] == 0][outcome_col].var()
        
        se = np.sqrt(treated_pre_var + treated_post_var + 
                    control_pre_var + control_post_var)
        
        # Store results
        self.results = {
            'estimate': did_estimate,
            'std_error': se,
            'conf_int': stats.norm.interval(0.95, did_estimate, se),
            'groups': {
                'treated_pre': treated_pre,
                'treated_post': treated_post,
                'control_pre': control_pre,
                'control_post': control_post
            }
        }
        
        return self.results
    
    def parallel_trends_test(self, data, time_col, treatment_col, 
                           outcome_col, pre_periods):
        """Test parallel trends assumption in pre-treatment periods"""
        trends = []
        for t1, t2 in zip(pre_periods[:-1], pre_periods[1:]):
            temp_result = self.estimate(
                data, time_col, treatment_col, outcome_col, t1, t2)
            trends.append(temp_result['estimate'])
        
        # Test if pre-trends are significantly different from zero
        _, p_value = stats.ttest_1samp(trends, 0)
        return {'p_value': p_value, 'pre_trends': trends}

# Example usage
np.random.seed(42)
n_units = 1000
n_periods = 4

# Generate synthetic panel data
data = pd.DataFrame({
    'unit_id': np.repeat(range(n_units), n_periods),
    'time': np.tile(range(n_periods), n_units),
    'treatment': np.repeat(np.random.binomial(1, 0.5, n_units), n_periods),
})

# Generate outcome with treatment effect
baseline = np.random.normal(0, 1, n_units)
time_trend = 0.2 * data['time']
treatment_effect = 0.5 * (data['time'] >= 2) * data['treatment']
data['outcome'] = (np.repeat(baseline, n_periods) + time_trend + 
                  treatment_effect + np.random.normal(0, 0.1, len(data)))

# Estimate DiD
did_model = DifferenceInDifferences()
results = did_model.estimate(data, 'time', 'treatment', 'outcome', 1, 2)
print(f"DiD Estimate: {results['estimate']:.3f}")
print(f"95% CI: [{results['conf_int'][0]:.3f}, {results['conf_int'][1]:.3f}]")
```

Slide 11: Regression Discontinuity Design (RDD)

Regression Discontinuity Design is a causal inference method that exploits known thresholds in treatment assignment. This implementation demonstrates how to estimate local treatment effects around a cutoff point using both parametric and non-parametric approaches.

```python
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class RegressionDiscontinuity:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.left_model = LinearRegression()
        self.right_model = LinearRegression()
        
    def estimate(self, X, Y, cutoff, polynomial_order=1):
        """Estimate treatment effect at discontinuity"""
        # Create treatment indicator
        T = (X >= cutoff).astype(int)
        
        # Select bandwidth if not specified
        if self.bandwidth is None:
            self.bandwidth = self._select_optimal_bandwidth(X, Y, cutoff)
        
        # Filter data within bandwidth
        mask = np.abs(X - cutoff) <= self.bandwidth
        X_local = X[mask]
        Y_local = Y[mask]
        T_local = T[mask]
        
        # Generate polynomial features
        X_poly = np.column_stack([
            (X_local - cutoff)**p for p in range(1, polynomial_order + 1)
        ])
        
        # Fit separate models on each side
        left_mask = X_local < cutoff
        right_mask = X_local >= cutoff
        
        self.left_model.fit(X_poly[left_mask], Y_local[left_mask])
        self.right_model.fit(X_poly[right_mask], Y_local[right_mask])
        
        # Estimate treatment effect at cutoff
        X_cutoff = np.zeros(polynomial_order)
        treatment_effect = (self.right_model.predict([X_cutoff]) - 
                          self.left_model.predict([X_cutoff]))[0]
        
        # Calculate standard error
        se = self._calculate_standard_error(
            X_local, Y_local, T_local, cutoff)
        
        return {
            'treatment_effect': treatment_effect,
            'std_error': se,
            'conf_int': norm.interval(0.95, treatment_effect, se),
            'bandwidth': self.bandwidth
        }
    
    def _select_optimal_bandwidth(self, X, Y, cutoff):
        """Implementation of Imbens-Kalyanaraman optimal bandwidth"""
        n = len(X)
        h = 1.84 * np.std(X) * n**(-0.2)  # Rule of thumb bandwidth
        return h
    
    def _calculate_standard_error(self, X, Y, T, cutoff):
        # Simplified variance estimation
        n = len(X)
        residuals = Y - self.predict(X)
        return np.sqrt(np.var(residuals) / n)
    
    def predict(self, X):
        """Predict outcomes using fitted models"""
        X_centered = X - self.cutoff
        return np.where(X >= self.cutoff,
                       self.right_model.predict(X_centered),
                       self.left_model.predict(X_centered))

# Example usage
np.random.seed(42)
n_samples = 1000
cutoff = 0

# Generate running variable
X = np.random.normal(0, 2, n_samples)

# Generate outcome with discontinuity
true_effect = 2
Y = (0.5 * X + true_effect * (X >= cutoff) + 
     np.random.normal(0, 0.1, n_samples))

# Fit RDD
rdd = RegressionDiscontinuity()
results = rdd.estimate(X, Y, cutoff)
print(f"RDD Estimate: {results['treatment_effect']:.3f}")
print(f"95% CI: [{results['conf_int'][0]:.3f}, {results['conf_int'][1]:.3f}]")
```

Slide 12: Synthetic Control Method

The synthetic control method creates a counterfactual by weighting control units to match the pre-treatment characteristics of the treated unit. This implementation demonstrates how to construct and evaluate synthetic controls.

```python
import numpy as np
from scipy.optimize import minimize
import pandas as pd

class SyntheticControl:
    def __init__(self):
        self.weights = None
        self.donors = None
        self.treated_unit = None
        
    def fit(self, X, treated_unit, donor_units, 
            pre_treatment_period, optimization_method='SLSQP'):
        """
        X: Panel data with units as columns and time as index
        treated_unit: Name/index of treated unit
        donor_units: List of potential control units
        pre_treatment_period: Time periods before treatment
        """
        self.treated_unit = treated_unit
        self.donors = donor_units
        
        # Extract pre-treatment data
        Y1 = X.loc[pre_treatment_period, treated_unit].values
        Y0 = X.loc[pre_treatment_period, donor_units].values
        
        # Define optimization problem
        def objective(w):
            return np.sum((Y1 - np.dot(Y0, w))**2)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, 1) for _ in range(len(donor_units))]
        
        # Initial weights
        w0 = np.ones(len(donor_units)) / len(donor_units)
        
        # Optimize
        result = minimize(objective, w0, method=optimization_method,
                        constraints=constraints, bounds=bounds)
        
        self.weights = result.x
        return self
    
    def predict(self, X):
        """Generate synthetic control predictions"""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")
            
        synthetic = np.dot(X[self.donors], self.weights)
        return pd.Series(synthetic, index=X.index)
    
    def plot_results(self, X, treatment_period):
        """Plot treated unit vs synthetic control"""
        import matplotlib.pyplot as plt
        
        actual = X[self.treated_unit]
        synthetic = self.predict(X)
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual.values, 'b-', label='Treated Unit')
        plt.plot(synthetic.index, synthetic.values, 'r--', 
                label='Synthetic Control')
        plt.axvline(x=treatment_period, color='g', linestyle=':',
                   label='Treatment')
        plt.legend()
        plt.show()

# Example usage
np.random.seed(42)
n_units = 20
n_periods = 50
treatment_period = 30

# Generate synthetic panel data
time_index = pd.date_range('2020-01-01', periods=n_periods, freq='M')
units = [f'Unit_{i}' for i in range(n_units)]

# Generate data with treatment effect
data = pd.DataFrame(index=time_index, columns=units)
for unit in units:
    trend = np.linspace(0, 1, n_periods)
    noise = np.random.normal(0, 0.1, n_periods)
    data[unit] = trend + noise

# Add treatment effect to Unit_0
treatment_effect = 0.5
data.loc[time_index[treatment_period:], 'Unit_0'] += treatment_effect

# Fit synthetic control
sc = SyntheticControl()
sc.fit(data, 'Unit_0', units[1:], 
       pre_treatment_period=time_index[:treatment_period])

# Get predictions and plot results
synthetic_control = sc.predict(data)
sc.plot_results(data, time_index[treatment_period])
```

Slide 13: Neural Networks for Causal Discovery

This advanced implementation combines deep learning with causal discovery by using neural networks to learn causal structures from observational data. The approach uses gradient-based optimization to identify potential causal relationships while maintaining acyclicity constraints.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CausalDiscoveryNetwork(nn.Module):
    def __init__(self, n_variables):
        super().__init__()
        self.n_variables = n_variables
        
        # Adjacency matrix parameters
        self.W = nn.Parameter(torch.randn(n_variables, n_variables) * 0.1)
        
        # Neural networks for each variable
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_variables, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(n_variables)
        ])
        
    def forward(self, X):
        # Get adjacency matrix (enforce acyclicity)
        A = self._get_acyclic_adj_matrix()
        
        # Compute predictions for each variable
        predictions = []
        for i in range(self.n_variables):
            # Get parents of current variable
            parents = A[:, i]
            mask = parents.unsqueeze(0)
            
            # Apply mask to input
            masked_input = X * mask
            pred = self.networks[i](masked_input)
            predictions.append(pred)
            
        return torch.cat(predictions, dim=1)
    
    def _get_acyclic_adj_matrix(self):
        """Convert parameters to DAG using matrix exponential"""
        return torch.sigmoid(self.W) * (1 - torch.eye(self.n_variables))
    
    def get_dag_penalty(self):
        """Compute penalty for non-acyclicity"""
        A = self._get_acyclic_adj_matrix()
        M = torch.matrix_exp(A * A) - torch.eye(self.n_variables)
        return torch.sum(M * M)

class CausalDiscovery:
    def __init__(self, n_variables, learning_rate=0.001):
        self.model = CausalDiscoveryNetwork(n_variables)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=learning_rate)
        
    def train(self, data, n_epochs=1000, lambda_dag=0.1):
        X = torch.FloatTensor(data)
        
        for epoch in range(n_epochs):
            # Forward pass
            pred = self.model(X)
            
            # Compute loss
            reconstruction_loss = nn.MSELoss()(pred, X)
            dag_penalty = self.model.get_dag_penalty()
            loss = reconstruction_loss + lambda_dag * dag_penalty
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, "
                      f"Loss: {loss.item():.4f}, "
                      f"DAG Penalty: {dag_penalty.item():.4f}")
    
    def get_causal_matrix(self):
        """Return learned causal adjacency matrix"""
        with torch.no_grad():
            return self.model._get_acyclic_adj_matrix().numpy()

# Example usage
np.random.seed(42)
n_samples = 1000
n_variables = 4

# Generate synthetic causal data
true_dag = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

# Generate data according to causal structure
X = np.zeros((n_samples, n_variables))
X[:, 0] = np.random.normal(0, 1, n_samples)
X[:, 1] = 0.7 * X[:, 0] + np.random.normal(0, 0.1, n_samples)
X[:, 2] = 0.8 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
X[:, 3] = 0.6 * X[:, 2] + np.random.normal(0, 0.1, n_samples)

# Train model
causal_discovery = CausalDiscovery(n_variables)
causal_discovery.train(X)

# Get learned causal structure
learned_dag = causal_discovery.get_causal_matrix()
print("Learned DAG structure:")
print(learned_dag)
```

Slide 14: Additional Resources

*   "Elements of Causal Inference: Foundations and Learning Algorithms" - [https://arxiv.org/abs/1908.09907](https://arxiv.org/abs/1908.09907)
*   "Causality: Models, Reasoning, and Inference in Neural Networks" - [https://arxiv.org/abs/2002.07006](https://arxiv.org/abs/2002.07006)
*   "A Survey of Learning Causality with Data: Problems and Methods" - [https://arxiv.org/abs/1809.09337](https://arxiv.org/abs/1809.09337)
*   "Causal Discovery from Observational Data: A Survey of Recent Progress" - Search on Google Scholar
*   "Deep Learning for Causal Discovery and Inference: A Review" - Search on Google Scholar
*   "The Seven Tools of Causal Inference: A Review" - Search on Google Scholar
*   "Machine Learning Methods for Estimating Heterogeneous Causal Effects" - Check Arxiv for the latest papers on this topic

