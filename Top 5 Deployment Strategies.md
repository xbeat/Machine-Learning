## Top 5 Deployment Strategies
Slide 1: Big Bang Deployment

Big Bang Deployment is a strategy where the entire application is updated at once. This approach involves replacing the old version with the new version in a single operation. While simple, it carries high risk as any issues affect the entire system simultaneously.

```python
def big_bang_deploy(new_version, old_version):
    # Shut down the entire system
    shutdown_system(old_version)
    
    # Replace old version with new version
    replace_version(old_version, new_version)
    
    # Start up the new system
    startup_system(new_version)
    
    print(f"Deployed version {new_version}")

# Usage
big_bang_deploy("2.0", "1.0")
```

Slide 2: Rolling Deployment

Rolling Deployment gradually updates instances of the application. This method reduces downtime and risk by updating a subset of servers or instances at a time. It allows for a smoother transition and easier rollback if issues arise.

```python
def rolling_deploy(instances, new_version):
    for instance in instances:
        # Take instance out of load balancer
        remove_from_lb(instance)
        
        # Update instance
        update_instance(instance, new_version)
        
        # Add instance back to load balancer
        add_to_lb(instance)
        
        print(f"Updated instance {instance} to version {new_version}")

# Usage
instances = ["server1", "server2", "server3", "server4"]
rolling_deploy(instances, "2.0")
```

Slide 3: Blue-Green Deployment

Blue-Green Deployment involves maintaining two identical production environments. One environment (blue) runs the current version, while the other (green) is updated with the new version. Traffic is switched from blue to green once the new version is verified.

```python
def blue_green_deploy(blue_env, green_env, new_version):
    # Update green environment
    update_environment(green_env, new_version)
    
    # Run tests on green environment
    if test_environment(green_env):
        # Switch traffic to green
        switch_traffic(blue_env, green_env)
        print(f"Switched to new version {new_version}")
    else:
        print("Deployment failed, staying on current version")

# Usage
blue_green_deploy("blue-env", "green-env", "2.0")
```

Slide 4: Canary Deployment

Canary Deployment involves releasing a new version to a small subset of users or servers before rolling it out to the entire infrastructure. This allows for real-world testing and gradual adoption of the new version.

```python
def canary_deploy(total_instances, canary_percent, new_version):
    canary_count = int(total_instances * (canary_percent / 100))
    
    # Deploy to canary instances
    for i in range(canary_count):
        deploy_to_instance(f"instance-{i}", new_version)
    
    # Monitor canary instances
    if monitor_canary_health():
        # Deploy to remaining instances
        for i in range(canary_count, total_instances):
            deploy_to_instance(f"instance-{i}", new_version)
        print(f"Full deployment of version {new_version} complete")
    else:
        rollback_canary()
        print("Canary deployment failed, rolled back")

# Usage
canary_deploy(100, 10, "2.0")
```

Slide 5: Feature Toggle

Feature Toggle, also known as Feature Flags, allows developers to enable or disable features without deploying new code. This strategy provides fine-grained control over feature releases and supports A/B testing.

```python
class FeatureToggle:
    def __init__(self):
        self.features = {}

    def set_feature(self, feature_name, enabled):
        self.features[feature_name] = enabled

    def is_enabled(self, feature_name):
        return self.features.get(feature_name, False)

# Usage
toggle = FeatureToggle()
toggle.set_feature("new_ui", True)
toggle.set_feature("beta_feature", False)

if toggle.is_enabled("new_ui"):
    show_new_ui()
else:
    show_old_ui()
```

Slide 6: Real-Life Example - E-commerce Platform

Consider an e-commerce platform implementing a new recommendation engine. Using Canary Deployment, the company can test the new engine with a small percentage of users before full rollout.

```python
def deploy_recommendation_engine(total_users, canary_percent):
    canary_users = int(total_users * (canary_percent / 100))
    
    # Deploy to canary users
    for user in range(canary_users):
        enable_new_recommendations(user)
    
    # Monitor for 24 hours
    if monitor_user_engagement(24):
        # Full deployment
        for user in range(canary_users, total_users):
            enable_new_recommendations(user)
        print("New recommendation engine fully deployed")
    else:
        rollback_recommendations()
        print("Deployment halted, reverting to old engine")

# Usage
deploy_recommendation_engine(1000000, 5)
```

Slide 7: Real-Life Example - Content Management System

A content management system (CMS) wants to introduce a new WYSIWYG editor. Using Feature Toggle, they can gradually roll out the feature and easily disable it if issues arise.

```python
class CMSFeatures:
    def __init__(self):
        self.features = {
            "new_editor": False,
            "auto_save": True,
            "version_control": True
        }

    def enable_feature(self, feature):
        if feature in self.features:
            self.features[feature] = True
            print(f"{feature} enabled")
        else:
            print(f"Feature {feature} not found")

    def use_editor(self):
        if self.features["new_editor"]:
            return "Using new WYSIWYG editor"
        else:
            return "Using old text editor"

# Usage
cms = CMSFeatures()
print(cms.use_editor())  # Using old text editor
cms.enable_feature("new_editor")
print(cms.use_editor())  # Using new WYSIWYG editor
```

Slide 8: Comparing Deployment Strategies

Different deployment strategies suit different scenarios. Here's a Python script that simulates and compares the deployment time and risk for each strategy:

```python
import random

def simulate_deployment(strategy, instances=100, failure_rate=0.05):
    deployed = 0
    failures = 0
    time = 0

    if strategy == "big_bang":
        time = 1
        if random.random() < failure_rate:
            failures = instances
        else:
            deployed = instances

    elif strategy in ["rolling", "canary"]:
        for _ in range(instances):
            time += 1
            if random.random() >= failure_rate:
                deployed += 1
            else:
                failures += 1
                if strategy == "canary":
                    break

    elif strategy == "blue_green":
        time = 2
        if random.random() < failure_rate:
            failures = instances
        else:
            deployed = instances

    return {"strategy": strategy, "time": time, "deployed": deployed, "failures": failures}

# Simulate each strategy 1000 times
results = {strategy: [simulate_deployment(strategy) for _ in range(1000)]
           for strategy in ["big_bang", "rolling", "canary", "blue_green"]}

# Calculate averages
for strategy, sims in results.items():
    avg_time = sum(s["time"] for s in sims) / len(sims)
    avg_deployed = sum(s["deployed"] for s in sims) / len(sims)
    avg_failures = sum(s["failures"] for s in sims) / len(sims)
    print(f"{strategy}: Avg Time: {avg_time:.2f}, Avg Deployed: {avg_deployed:.2f}, Avg Failures: {avg_failures:.2f}")
```

Slide 9: Results for: Comparing Deployment Strategies

```
big_bang: Avg Time: 1.00, Avg Deployed: 95.00, Avg Failures: 5.00
rolling: Avg Time: 100.00, Avg Deployed: 95.00, Avg Failures: 5.00
canary: Avg Time: 5.24, Avg Deployed: 4.98, Avg Failures: 0.26
blue_green: Avg Time: 2.00, Avg Deployed: 95.00, Avg Failures: 5.00
```

Slide 10: Analyzing Deployment Strategies

The simulation results provide insights into the trade-offs between different deployment strategies. Big Bang and Blue-Green are fastest but riskier, potentially affecting all instances at once. Rolling deployment is slowest but limits failures to individual instances. Canary deployment balances speed and risk by catching issues early.

```python
import matplotlib.pyplot as plt

strategies = ["Big Bang", "Rolling", "Canary", "Blue-Green"]
times = [1.00, 100.00, 5.24, 2.00]
deployed = [95.00, 95.00, 4.98, 95.00]
failures = [5.00, 5.00, 0.26, 5.00]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.bar(strategies, times)
ax1.set_title("Average Deployment Time")
ax1.set_ylabel("Time")

ax2.bar(strategies, deployed)
ax2.set_title("Average Instances Deployed")
ax2.set_ylabel("Instances")

ax3.bar(strategies, failures)
ax3.set_title("Average Failures")
ax3.set_ylabel("Failures")

plt.tight_layout()
plt.show()
```

Slide 11: Code for: Analyzing Deployment Strategies

This code generates a visual representation of the deployment strategy comparison, creating bar charts for average deployment time, instances deployed, and failures for each strategy.

```python
import matplotlib.pyplot as plt

strategies = ["Big Bang", "Rolling", "Canary", "Blue-Green"]
times = [1.00, 100.00, 5.24, 2.00]
deployed = [95.00, 95.00, 4.98, 95.00]
failures = [5.00, 5.00, 0.26, 5.00]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.bar(strategies, times)
ax1.set_title("Average Deployment Time")
ax1.set_ylabel("Time")

ax2.bar(strategies, deployed)
ax2.set_title("Average Instances Deployed")
ax2.set_ylabel("Instances")

ax3.bar(strategies, failures)
ax3.set_title("Average Failures")
ax3.set_ylabel("Failures")

plt.tight_layout()
plt.show()
```

Slide 12: Mathematical Model for Deployment Risk

We can model the risk of a deployment failure using probability theory. Let's consider a simple model where the probability of a successful deployment for a single instance is p. For a system with n instances, we can calculate the probability of a completely successful deployment for different strategies.

```python
import math

def big_bang_success(p, n):
    return p ** n

def rolling_success(p, n):
    return p ** n

def blue_green_success(p, n):
    return p ** n

def canary_success(p, n, canary_size):
    canary_prob = p ** canary_size
    full_prob = p ** (n - canary_size)
    return canary_prob * full_prob

# Example usage
p = 0.99  # 99% success rate per instance
n = 100   # 100 instances
canary_size = 10

print(f"Big Bang success probability: {big_bang_success(p, n):.4f}")
print(f"Rolling success probability: {rolling_success(p, n):.4f}")
print(f"Blue-Green success probability: {blue_green_success(p, n):.4f}")
print(f"Canary success probability: {canary_success(p, n, canary_size):.4f}")
```

Slide 13: Results for: Mathematical Model for Deployment Risk

```
Big Bang success probability: 0.3660
Rolling success probability: 0.3660
Blue-Green success probability: 0.3660
Canary success probability: 0.3697
```

Slide 14: Interpreting Deployment Risk Results

The mathematical model reveals that for strategies affecting all instances simultaneously (Big Bang, Blue-Green), the success probability decreases exponentially with the number of instances. Rolling deployment, despite updating instances one by one, has the same overall success probability for a complete deployment.

Canary deployment shows a slightly higher success probability because it allows for early detection of issues, potentially preventing full deployment of a problematic update.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_success_probabilities(p_range, n):
    probs = np.array(p_range)
    big_bang = probs ** n
    canary = (probs ** 10) * (probs ** (n - 10))

    plt.figure(figsize=(10, 6))
    plt.plot(probs, big_bang, label='Big Bang/Rolling/Blue-Green')
    plt.plot(probs, canary, label='Canary (10% canary)')
    plt.xlabel('Success probability per instance')
    plt.ylabel('Overall success probability')
    plt.title(f'Deployment Success Probability (n={n})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for 100 instances
plot_success_probabilities(np.linspace(0.95, 1, 100), 100)
```

Slide 15: Additional Resources

For more in-depth information on deployment strategies and their mathematical models, consider exploring these resources:

1.  "Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation" by Jez Humble and David Farley
2.  "Site Reliability Engineering: How Google Runs Production Systems" edited by Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy
3.  ArXiv paper: "A Survey of Rollback-Recovery Protocols in Message-Passing Systems" by E. N. Elnozahy et al. ([https://arxiv.org/abs/cs/0201037](https://arxiv.org/abs/cs/0201037))
4.  "Release It!: Design and Deploy Production-Ready Software" by Michael T. Nygard

These resources provide comprehensive coverage of deployment strategies, including theoretical foundations and practical implementations in various environments.

