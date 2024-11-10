## Generative Modeling on Data Manifolds
Slide 1: Differential Geometry Foundations

The mathematical foundation of Riemannian Flow Matching requires understanding differential geometry concepts. We'll implement basic geometric operations and manifold calculations using Python's numerical computing capabilities.

```python
import numpy as np
from scipy.linalg import sqrtm

def compute_metric_tensor(points, jacobian):
    """
    Compute the Riemannian metric tensor G = J^T J
    
    Args:
        points: Input points on manifold (n_points, dim)
        jacobian: Jacobian matrix at each point (n_points, out_dim, in_dim)
    """
    # Compute metric tensor G = J^T J
    metric_tensor = np.einsum('...ji,...jk->...ik', jacobian, jacobian)
    
    # Ensure symmetry and positive definiteness
    metric_tensor = (metric_tensor + np.transpose(metric_tensor, (0, 2, 1))) / 2
    
    return metric_tensor

# Example usage
points = np.random.randn(10, 3)  # 10 points in 3D
jacobian = np.random.randn(10, 3, 3)  # Jacobian matrices
G = compute_metric_tensor(points, jacobian)
print("Metric tensor shape:", G.shape)
print("Sample metric tensor:\n", G[0])
```

Slide 2: Neural ODE Implementation

Neural ODEs form the backbone of the flow matching process, allowing continuous transformation between manifolds. This implementation provides the fundamental building blocks for solving differential equations with neural networks.

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, func, method='dopri5'):
        super().__init__()
        self.func = func
        self.method = method
        
    def forward(self, x, t_span):
        return odeint(self.func, x, t_span, method=self.method)

# Example usage
hidden_dim = 64
func = ODEFunc(hidden_dim)
model = NeuralODE(func)

# Generate sample data
x0 = torch.randn(16, hidden_dim)  # batch_size=16
t_span = torch.linspace(0., 1., 100)

# Solve ODE
solution = model(x0, t_span)
print("Solution shape:", solution.shape)
```

Slide 3: Pullback Metric Computation

The pullback metric is essential for maintaining geometric properties during manifold transformations. This implementation computes the pullback metric tensor and its associated geometric quantities.

```python
import torch
import torch.autograd as autograd

def compute_pullback_metric(f, x, create_graph=True):
    """
    Compute pullback metric tensor for mapping f: M → N
    
    Args:
        f: Neural network mapping function
        x: Points on source manifold (batch_size, dim)
        create_graph: Whether to create backward graph
    """
    batch_size = x.size(0)
    dim = x.size(1)
    
    # Compute Jacobian
    jacobian = torch.zeros(batch_size, dim, dim)
    for i in range(dim):
        grad = autograd.grad(f[:,i].sum(), x, 
                           create_graph=create_graph)[0]
        jacobian[:,:,i] = grad
    
    # Compute metric tensor G = J^T J
    g = torch.bmm(jacobian.transpose(1,2), jacobian)
    
    return g

# Example usage with simple network
class Mapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
    
    def forward(self, x):
        return self.net(x)

dim = 3
f = Mapping(dim)
x = torch.randn(10, dim, requires_grad=True)
g = compute_pullback_metric(f, x)
print("Pullback metric shape:", g.shape)
print("Sample metric:\n", g[0])
```

Slide 4: Riemannian Flow Matching Algorithm

The core algorithm of RFM involves matching probability distributions on manifolds through continuous flows. This implementation demonstrates the key components of the matching process using neural networks.

```python
import torch
import torch.nn as nn

class RiemannianFlowMatcher(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.velocity_field = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, t, x):
        """
        Compute the velocity field at point x and time t
        Args:
            t: Time parameter [0,1]
            x: Points on manifold (batch_size, dim)
        """
        # Concatenate time and position
        tx = torch.cat([t.expand(x.shape[0], 1), x], dim=1)
        velocity = self.velocity_field(tx)
        return velocity

    def compute_flow_loss(self, x0, x1, n_steps=50):
        t = torch.linspace(0, 1, n_steps)
        batch_size = x0.shape[0]
        
        # Compute trajectory
        current_x = x0
        loss = 0
        
        for i in range(n_steps-1):
            dt = t[i+1] - t[i]
            velocity = self.forward(t[i] * torch.ones(1), current_x)
            next_x = current_x + velocity * dt
            
            # Compute matching loss
            target_x = x0 + (x1 - x0) * t[i+1]
            loss += torch.mean((next_x - target_x)**2)
            current_x = next_x
            
        return loss / n_steps

# Example usage
dim = 3
flow_matcher = RiemannianFlowMatcher(dim)

# Generate sample data
batch_size = 32
x0 = torch.randn(batch_size, dim)
x1 = torch.randn(batch_size, dim)

# Compute loss
loss = flow_matcher.compute_flow_loss(x0, x1)
print(f"Flow matching loss: {loss.item():.4f}")
```

Slide 5: Isometric Mapping Loss

The isometric mapping loss ensures that distances between points are preserved during the transformation. This implementation provides both local and global isometry constraints.

```python
import torch
import torch.nn as nn
from torch.nn.functional import pdist, pairwise_distance

class IsometricMappingLoss(nn.Module):
    def __init__(self, use_global=True, neighborhood_size=10):
        super().__init__()
        self.use_global = use_global
        self.neighborhood_size = neighborhood_size
    
    def compute_local_loss(self, x, fx):
        """Compute local isometry loss using k-nearest neighbors"""
        # Compute pairwise distances in input space
        d_input = pairwise_distance(x, x)
        
        # Compute pairwise distances in output space
        d_output = pairwise_distance(fx, fx)
        
        # Get k-nearest neighbors
        _, indices = torch.topk(d_input, self.neighborhood_size, 
                              dim=1, largest=False)
        
        local_loss = 0
        for i in range(x.shape[0]):
            neighbors = indices[i]
            d_in = d_input[i][neighbors]
            d_out = d_output[i][neighbors]
            local_loss += torch.mean((d_in - d_out)**2)
            
        return local_loss / x.shape[0]
    
    def compute_global_loss(self, x, fx):
        """Compute global isometry loss using all pairs"""
        d_input = pdist(x)
        d_output = pdist(fx)
        return torch.mean((d_input - d_output)**2)
    
    def forward(self, x, fx):
        if self.use_global:
            return self.compute_global_loss(x, fx)
        return self.compute_local_loss(x, fx)

# Example usage
batch_size, dim = 50, 3
x = torch.randn(batch_size, dim)
fx = torch.randn(batch_size, dim, requires_grad=True)

# Compute both local and global losses
local_criterion = IsometricMappingLoss(use_global=False)
global_criterion = IsometricMappingLoss(use_global=True)

local_loss = local_criterion(x, fx)
global_loss = global_criterion(x, fx)

print(f"Local isometry loss: {local_loss.item():.4f}")
print(f"Global isometry loss: {global_loss.item():.4f}")
```

Slide 6: Graph Matching Implementation

Graph matching provides an alternative approach to ensuring geometric consistency during manifold transformations. This implementation focuses on preserving local structure through graph-based metrics.

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class GraphMatcher(nn.Module):
    def __init__(self, k_neighbors=5):
        super().__init__()
        self.k_neighbors = k_neighbors
        
    def build_graph(self, x):
        """Build k-NN graph from points"""
        dist_matrix = torch.cdist(x, x)
        # Get k nearest neighbors
        _, indices = torch.topk(dist_matrix, self.k_neighbors + 1, 
                              largest=False)
        return indices[:, 1:]  # Exclude self-connections
    
    def compute_graph_matching_loss(self, x, fx):
        """
        Compute loss based on preservation of graph structure
        """
        # Build graphs in both spaces
        source_graph = self.build_graph(x)
        target_graph = self.build_graph(fx)
        
        # Compute edge weights
        source_weights = torch.zeros_like(source_graph, dtype=torch.float)
        target_weights = torch.zeros_like(target_graph, dtype=torch.float)
        
        for i in range(x.shape[0]):
            source_weights[i] = torch.norm(
                x[source_graph[i]] - x[i].unsqueeze(0), dim=1)
            target_weights[i] = torch.norm(
                fx[target_graph[i]] - fx[i].unsqueeze(0), dim=1)
        
        # Compare graph structures
        loss = torch.mean((source_weights - target_weights)**2)
        
        return loss

# Example usage
batch_size, dim = 100, 3
x = torch.randn(batch_size, dim)
fx = torch.randn(batch_size, dim, requires_grad=True)

matcher = GraphMatcher(k_neighbors=5)
loss = matcher.compute_graph_matching_loss(x, fx)

print(f"Graph matching loss: {loss.item():.4f}")
```

Slide 7: Neural ODE Solver for Manifold Flows

The implementation of a specialized Neural ODE solver for handling flows on manifolds requires careful consideration of the geometric structure. This solver incorporates Riemannian metrics into the integration process.

```python
import torch
import torch.nn as nn
from torch.autograd import grad

class ManifoldNeuralODE(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, t, states):
        """
        Compute the vector field incorporating manifold structure
        Args:
            t: Current time
            states: Current state on manifold
        """
        # Compute tangent vectors
        batch_size = states.shape[0]
        t_batch = t.expand(batch_size, 1)
        
        # Compute metric-aware coordinates
        metric_coords = self.compute_metric_coords(states)
        
        # Concatenate time, state, and metric information
        inputs = torch.cat([t_batch, states, metric_coords], dim=1)
        
        # Compute vector field
        vector_field = self.net(inputs)
        
        return vector_field
    
    def compute_metric_coords(self, x):
        """
        Compute metric-aware coordinates for better flow
        """
        # Compute local coordinate basis
        basis = torch.eye(x.shape[1], device=x.device)
        basis = basis.expand(x.shape[0], -1, -1)
        
        # Project onto tangent space
        metric_coords = torch.bmm(basis, basis.transpose(1, 2))
        return metric_coords.reshape(x.shape[0], -1)
    
    def integrate(self, x0, steps=100):
        """
        Integrate the flow from initial conditions
        """
        t = torch.linspace(0, 1, steps)
        trajectory = [x0]
        current_x = x0
        
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            # Runge-Kutta 4 integration step
            k1 = self.forward(t[i], current_x)
            k2 = self.forward(t[i] + dt/2, current_x + dt*k1/2)
            k3 = self.forward(t[i] + dt/2, current_x + dt*k2/2)
            k4 = self.forward(t[i] + dt, current_x + dt*k3)
            
            current_x = current_x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(current_x)
            
        return torch.stack(trajectory)

# Example usage
dim = 3
model = ManifoldNeuralODE(dim)

# Generate sample data
batch_size = 16
x0 = torch.randn(batch_size, dim)

# Integrate flow
trajectory = model.integrate(x0)
print("Trajectory shape:", trajectory.shape)
print("Final states:\n", trajectory[-1])
```

Slide 8: Protein Binding Dataset Implementation

Working with real protein binding data requires specialized preprocessing and model adaptation. This implementation demonstrates how to handle molecular structure data for manifold learning.

```python
import torch
import torch.nn as nn
import numpy as np
from Bio.PDB import *
from scipy.spatial.distance import pdist, squareform

class ProteinManifoldMapper:
    def __init__(self, pdb_file):
        self.parser = PDBParser()
        self.structure = self.parser.get_structure('protein', pdb_file)
        
    def extract_features(self):
        """Extract relevant geometric features from protein structure"""
        coords = []
        features = []
        
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    # Get CA atoms for backbone structure
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        coords.append(ca_atom.get_coord())
                        
                        # Extract additional features (e.g., hydrophobicity)
                        features.append(self.compute_residue_features(residue))
        
        return np.array(coords), np.array(features)
    
    def compute_residue_features(self, residue):
        """Compute chemical and physical features for each residue"""
        # Simplified feature computation
        hydrophobicity = {
            'ALA': 0.31, 'ARG': -1.01, 'ASN': -0.60,
            'ASP': -0.77, 'CYS': 1.54, 'GLN': -0.22,
            'GLU': -0.64, 'GLY': 0.00, 'HIS': 0.13,
            'ILE': 1.80, 'LEU': 1.70, 'LYS': -0.99,
            'MET': 1.23, 'PHE': 1.79, 'PRO': 0.72,
            'SER': -0.04, 'THR': 0.26, 'TRP': 2.25,
            'TYR': 0.96, 'VAL': 1.22
        }
        
        return np.array([
            hydrophobicity.get(residue.get_resname(), 0.0),
            residue.get_full_id()[3][1]  # Residue number
        ])
    
    def compute_distance_matrix(self, coords):
        """Compute pairwise distances between residues"""
        return squareform(pdist(coords))

# Example usage
class ProteinBindingModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

# Simulate protein data processing
coords = np.random.randn(100, 3)  # Simulated protein coordinates
features = np.random.randn(100, 2)  # Simulated protein features

# Prepare data for model
X = torch.tensor(np.concatenate([coords, features], axis=1), dtype=torch.float32)
model = ProteinBindingModel(input_dim=5, latent_dim=2)

# Forward pass
latent_coords = model(X)
print("Latent coordinates shape:", latent_coords.shape)
```

Slide 9: Riemannian Optimization Implementation

The optimization process on Riemannian manifolds requires specialized algorithms that respect the manifold's geometry. This implementation shows how to perform gradient descent while maintaining geometric constraints.

```python
import torch
import torch.optim as optim

class RiemannianOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
        
    def compute_riemannian_gradient(self, param, metric_tensor):
        """
        Compute Riemannian gradient using metric tensor
        """
        euclidean_grad = param.grad
        if euclidean_grad is None:
            return None
            
        # Solve metric equation G * grad_R = grad_E
        riemannian_grad = torch.linalg.solve(
            metric_tensor, 
            euclidean_grad.unsqueeze(-1)
        ).squeeze(-1)
        
        return riemannian_grad
    
    def step(self, metric_tensors):
        """
        Perform one step of Riemannian gradient descent
        """
        for param, metric_tensor in zip(self.params, metric_tensors):
            if param.grad is None:
                continue
                
            riem_grad = self.compute_riemannian_gradient(param, metric_tensor)
            
            # Update parameter using retraction
            param.data = self.retraction(param.data, -self.lr * riem_grad)
            
    def retraction(self, point, vector):
        """
        Project updated point back onto manifold
        """
        # Simple exponential map approximation
        return point + vector

# Example usage
class ManifoldModel(torch.nn.Module):
    def __init__(self, dim, manifold_dim):
        super().__init__()
        self.projection = torch.nn.Linear(dim, manifold_dim)
        
    def forward(self, x):
        return self.projection(x)
    
    def compute_metric_tensor(self, x):
        """Compute metric tensor at point x"""
        batch_size = x.shape[0]
        dim = x.shape[1]
        metric = torch.eye(dim).repeat(batch_size, 1, 1)
        return metric

# Training example
dim, manifold_dim = 5, 3
model = ManifoldModel(dim, manifold_dim)
optimizer = RiemannianOptimizer(model.parameters())

# Sample data
x = torch.randn(10, dim)
y = torch.randn(10, manifold_dim)

# Training step
output = model(x)
loss = torch.nn.functional.mse_loss(output, y)
loss.backward()

# Compute metric tensors for all parameters
metric_tensors = [torch.eye(p.shape[-1]) for p in model.parameters()]
optimizer.step(metric_tensors)

print(f"Training loss: {loss.item():.4f}")
```

Slide 10: Geodesic Distance Computation

Computing geodesic distances on manifolds is crucial for understanding the geometric structure. This implementation provides methods for both exact and approximate geodesic distance calculations.

```python
import torch
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

class GeodesicDistanceCalculator:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def compute_distance_matrix(self, points):
        """
        Compute pairwise Euclidean distances
        """
        return torch.cdist(points, points)
    
    def build_adjacency_matrix(self, points):
        """
        Build sparse adjacency matrix using k-NN
        """
        dist_matrix = self.compute_distance_matrix(points)
        
        # Get k nearest neighbors
        _, indices = torch.topk(-dist_matrix, self.n_neighbors + 1, dim=1)
        
        n_points = points.shape[0]
        adj_matrix = torch.zeros_like(dist_matrix)
        
        # Fill adjacency matrix with distances to nearest neighbors
        for i in range(n_points):
            for j in indices[i][1:]:  # Skip self
                adj_matrix[i, j] = dist_matrix[i, j]
                adj_matrix[j, i] = dist_matrix[i, j]  # Symmetry
                
        return adj_matrix
    
    def compute_geodesic_distances(self, points):
        """
        Compute approximate geodesic distances using shortest paths
        """
        adj_matrix = self.build_adjacency_matrix(points)
        
        # Convert to scipy sparse matrix
        adj_matrix_np = adj_matrix.numpy()
        sparse_adj = csr_matrix(adj_matrix_np)
        
        # Compute shortest paths
        geodesic_dist_matrix = shortest_path(
            sparse_adj, 
            method='D',
            directed=False
        )
        
        return torch.from_numpy(geodesic_dist_matrix)
    
    def compute_geodesic_distance_pair(self, p1, p2, metric_tensor):
        """
        Compute exact geodesic distance between two points
        using metric tensor
        """
        # Simple numerical integration of geodesic equation
        steps = 100
        t = torch.linspace(0, 1, steps)
        dt = t[1] - t[0]
        
        # Initial velocity
        v0 = p2 - p1
        
        # Solve geodesic equation numerically
        current_p = p1
        current_v = v0
        
        for i in range(steps-1):
            # Christoffel symbols (simplified)
            G = metric_tensor(current_p)
            G_inv = torch.inverse(G)
            
            # Update position and velocity using geodesic equation
            current_p = current_p + current_v * dt
            current_v = current_v - 0.5 * torch.mv(G_inv, 
                torch.mv(G, current_v)) * dt
            
        return torch.norm(current_p - p1)

# Example usage
points = torch.randn(100, 3)
calculator = GeodesicDistanceCalculator()

# Compute all pairwise geodesic distances
geodesic_distances = calculator.compute_geodesic_distances(points)

print("Geodesic distance matrix shape:", geodesic_distances.shape)
print("Sample distances:\n", geodesic_distances[:5, :5])
```

Slide 11: Real-world Example - Protein Conformational Changes

This implementation demonstrates how to apply RFM to analyze protein conformational changes, incorporating both structural and chemical features in the manifold learning process.

```python
import torch
import torch.nn as nn
import numpy as np
from Bio import PDB
from scipy.spatial.transform import Rotation

class ProteinConformationAnalyzer:
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def extract_conformational_features(self, pdb_id, chain_id='A'):
        """Extract conformational features from PDB structure"""
        parser = PDB.PDBParser()
        structure = parser.get_structure(pdb_id, f"{pdb_id}.pdb")
        
        features = []
        coords = []
        
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if 'CA' in residue:  # Alpha carbon
                            ca_coord = residue['CA'].get_coord()
                            coords.append(ca_coord)
                            
                            # Extract torsion angles
                            phi, psi = self.compute_torsion_angles(residue)
                            features.append([
                                phi, psi,
                                self.get_residue_property(residue, 'hydrophobicity'),
                                self.get_residue_property(residue, 'volume'),
                                self.get_secondary_structure_encoding(residue)
                            ])
        
        return np.array(coords), np.array(features)
    
    def compute_torsion_angles(self, residue):
        """Compute phi and psi torsion angles"""
        phi = psi = 0.0
        try:
            phi = PDB.calc_phi(residue)
            psi = PDB.calc_psi(residue)
        except:
            pass
        return phi or 0.0, psi or 0.0
    
    def get_residue_property(self, residue, property_type):
        """Get physical/chemical properties of residues"""
        properties = {
            'hydrophobicity': {
                'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
                'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
                'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
                'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
                'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
            },
            'volume': {
                'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1,
                'CYS': 108.5, 'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1,
                'HIS': 153.2, 'ILE': 166.7, 'LEU': 166.7, 'LYS': 168.6,
                'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7, 'SER': 89.0,
                'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
            }
        }
        
        return properties[property_type].get(residue.get_resname(), 0.0)
    
    def get_secondary_structure_encoding(self, residue):
        """Encode secondary structure as numeric value"""
        dssp = PDB.DSSP(residue.get_parent().get_parent(), f"{residue.get_full_id()[0]}.pdb")
        ss = dssp[residue.get_full_id()[2:]]
        
        ss_encoding = {
            'H': 1.0,  # Alpha helix
            'B': 2.0,  # Beta bridge
            'E': 3.0,  # Extended strand
            'G': 4.0,  # 3-10 helix
            'I': 5.0,  # Pi helix
            'T': 6.0,  # Turn
            'S': 7.0   # Bend
        }
        
        return ss_encoding.get(ss, 0.0)

# Example usage
analyzer = ProteinConformationAnalyzer()

# Simulate protein data
coords = np.random.randn(100, 3)
features = np.random.randn(100, 5)

# Convert to torch tensors
coords_tensor = torch.FloatTensor(coords)
features_tensor = torch.FloatTensor(features)

# Encode features
latent_features = analyzer.encoder(features_tensor)

print("Coordinates shape:", coords_tensor.shape)
print("Latent features shape:", latent_features.shape)
```

Slide 12: Results Analysis and Visualization

Implementation of tools to analyze and visualize the results of Riemannian Flow Matching, including manifold embeddings and flow trajectories.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

class RFMVisualizer:
    def __init__(self):
        self.tsne = TSNE(n_components=2)
        self.umap = UMAP(n_components=2)
        
    def plot_manifold_embedding(self, points, labels=None, method='tsne'):
        """
        Visualize high-dimensional manifold in 2D
        """
        if method == 'tsne':
            embedding = self.tsne.fit_transform(points.detach().numpy())
        else:
            embedding = self.umap.fit_transform(points.detach().numpy())
            
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=labels if labels is not None else None,
                            cmap='viridis')
        
        if labels is not None:
            plt.colorbar(scatter)
            
        plt.title(f'Manifold Embedding ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        return embedding
    
    def plot_flow_trajectory(self, trajectory, time_points):
        """
        Visualize flow trajectory on manifold
        """
        embedding = self.tsne.fit_transform(
            trajectory.reshape(-1, trajectory.shape[-1]).detach().numpy()
        )
        
        embedding = embedding.reshape(trajectory.shape[0], -1, 2)
        
        plt.figure(figsize=(12, 8))
        for i in range(embedding.shape[1]):
            plt.plot(embedding[:, i, 0], embedding[:, i, 1], 
                    alpha=0.5, linewidth=1)
            plt.scatter(embedding[0, i, 0], embedding[0, i, 1], 
                       c='g', label='Start' if i == 0 else None)
            plt.scatter(embedding[-1, i, 0], embedding[-1, i, 1], 
                       c='r', label='End' if i == 0 else None)
            
        plt.title('Flow Trajectory Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        
    def plot_metric_tensor_heatmap(self, metric_tensor):
        """
        Visualize metric tensor as heatmap
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(metric_tensor.detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Metric Tensor Heatmap')
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')

# Example usage
visualizer = RFMVisualizer()

# Generate sample data
points = torch.randn(100, 10)
labels = torch.randn(100)
trajectory = torch.randn(50, 20, 10)  # time steps × batch size × dimension
time_points = torch.linspace(0, 1, 50)
metric_tensor = torch.randn(10, 10)

# Plot embeddings and trajectories
embedding = visualizer.plot_manifold_embedding(points, labels)
visualizer.plot_flow_trajectory(trajectory, time_points)
visualizer.plot_metric_tensor_heatmap(metric_tensor)

print("Embedding shape:", embedding.shape)
```

Slide 13: Performance Metrics Implementation

This implementation provides comprehensive metrics for evaluating the quality of manifold learning and flow matching, including distortion measures and topology preservation scores.

```python
import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

class ManifoldMetrics:
    def __init__(self):
        self.knn = NearestNeighbors(n_neighbors=10)
        
    def compute_distortion(self, original_distances, embedded_distances):
        """
        Compute distance distortion between original and embedded spaces
        """
        # Normalize distances
        orig_norm = original_distances / original_distances.max()
        emb_norm = embedded_distances / embedded_distances.max()
        
        # Compute mean relative difference
        distortion = torch.mean(torch.abs(orig_norm - emb_norm) / orig_norm)
        
        return distortion.item()
    
    def compute_trustworthiness(self, X, Y, n_neighbors=10):
        """
        Compute trustworthiness of the embedding
        """
        n_samples = X.shape[0]
        
        # Compute rankings in original space
        self.knn.fit(X)
        orig_neighbors = self.knn.kneighbors(X, return_distance=False)
        
        # Compute rankings in embedded space
        self.knn.fit(Y)
        emb_neighbors = self.knn.kneighbors(Y, return_distance=False)
        
        # Compute trustworthiness
        trust = 0.0
        for i in range(n_samples):
            orig_set = set(orig_neighbors[i])
            emb_set = set(emb_neighbors[i])
            
            # Find points that are neighbors in embedding but not in original
            violations = emb_set - orig_set
            
            # Compute ranking differences
            for j in violations:
                r_ij = np.where(orig_neighbors[i] == j)[0]
                if len(r_ij) > 0:
                    trust += (r_ij[0] + 1 - n_neighbors)
                    
        trust = 1 - (2 / (n_samples * n_neighbors * (2 * n_neighbors - 3))) * trust
        return trust
    
    def compute_continuity(self, X, Y, n_neighbors=10):
        """
        Compute continuity of the embedding
        """
        n_samples = X.shape[0]
        
        # Compute rankings in both spaces
        self.knn.fit(X)
        orig_neighbors = self.knn.kneighbors(X, return_distance=False)
        
        self.knn.fit(Y)
        emb_neighbors = self.knn.kneighbors(Y, return_distance=False)
        
        # Compute continuity
        continuity = 0.0
        for i in range(n_samples):
            emb_set = set(emb_neighbors[i])
            orig_set = set(orig_neighbors[i])
            
            # Find points that are neighbors in original but not in embedding
            violations = orig_set - emb_set
            
            # Compute ranking differences
            for j in violations:
                r_ij = np.where(emb_neighbors[i] == j)[0]
                if len(r_ij) > 0:
                    continuity += (r_ij[0] + 1 - n_neighbors)
                    
        continuity = 1 - (2 / (n_samples * n_neighbors * (2 * n_neighbors - 3))) * continuity
        return continuity
    
    def evaluate_flow(self, source_points, target_points, flow_points):
        """
        Evaluate quality of the learned flow
        """
        metrics = {}
        
        # Compute distance matrices
        source_dist = torch.cdist(source_points, source_points)
        target_dist = torch.cdist(target_points, target_points)
        flow_dist = torch.cdist(flow_points, flow_points)
        
        # Compute distortion
        metrics['source_distortion'] = self.compute_distortion(source_dist, flow_dist)
        metrics['target_distortion'] = self.compute_distortion(target_dist, flow_dist)
        
        # Compute topology preservation metrics
        metrics['trustworthiness'] = self.compute_trustworthiness(
            source_points.numpy(), flow_points.numpy())
        metrics['continuity'] = self.compute_continuity(
            source_points.numpy(), flow_points.numpy())
        
        return metrics

# Example usage
metrics_calculator = ManifoldMetrics()

# Generate sample data
source = torch.randn(100, 10)
target = torch.randn(100, 10)
flow = torch.randn(100, 10)

# Evaluate flow quality
results = metrics_calculator.evaluate_flow(source, target, flow)

print("Performance Metrics:")
for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")
```

Slide 14: Additional Resources

*   "Neural Ordinary Differential Equations" - [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
*   "Riemannian Flow Matching on General Geometries" - [https://arxiv.org/abs/2302.03660](https://arxiv.org/abs/2302.03660)
*   "Learning on Lie Groups for Invariant Geometric Processing" - [https://arxiv.org/abs/2012.12093](https://arxiv.org/abs/2012.12093)
*   "Continuous Normalizing Flows on Riemannian Manifolds" - [https://arxiv.org/abs/2006.10605](https://arxiv.org/abs/2006.10605)
*   "Geodesic Neural Networks for Learning on Manifolds" - [https://arxiv.org/abs/2002.07242](https://arxiv.org/abs/2002.07242)

