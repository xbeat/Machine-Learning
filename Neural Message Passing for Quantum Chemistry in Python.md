## Neural Message Passing for Quantum Chemistry in Python
Slide 1: Neural Message Passing for Quantum Chemistry

Neural message passing is a powerful technique that combines graph neural networks with quantum chemistry to predict molecular properties. This approach leverages the inherent graph-like structure of molecules to process and analyze chemical data efficiently. By representing atoms as nodes and bonds as edges, we can apply graph-based neural networks to learn and predict various molecular properties.

```python
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing

class QMMessagePassing(MessagePassing):
    def __init__(self):
        super(QMMessagePassing, self).__init__(aggr='add')
        self.mlp = torch.nn.Linear(2 * 64, 64)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.mlp(tmp)
```

Slide 2: Molecular Representation

In quantum chemistry, molecules are represented as graphs where atoms are nodes and chemical bonds are edges. Each atom has associated features such as atomic number, electronegativity, and valence electrons. Bonds can be characterized by their type (single, double, triple) and length. This graph representation allows us to capture the structural and chemical properties of molecules effectively.

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_molecule_graph(atoms, bonds):
    G = nx.Graph()
    for i, atom in enumerate(atoms):
        G.add_node(i, element=atom)
    for bond in bonds:
        G.add_edge(bond[0], bond[1])
    return G

# Example: Water molecule
atoms = ['O', 'H', 'H']
bonds = [(0, 1), (0, 2)]
water_molecule = create_molecule_graph(atoms, bonds)

nx.draw(water_molecule, with_labels=True)
plt.show()
```

Slide 3: Feature Engineering

Feature engineering is crucial in neural message passing for quantum chemistry. We need to encode atomic and molecular features in a way that neural networks can process. Common features include atomic number, electronegativity, covalent radius, and various molecular descriptors. These features are typically represented as vectors for each atom and molecule.

```python
import numpy as np

def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
        atom.GetMass()
    ])

def mol_features(mol):
    return np.array([
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        mol.GetNumHeavyAtoms(),
        mol.GetNumRotatableBonds(),
        Descriptors.ExactMolWt(mol)
    ])

# Example usage
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles('CCO')
atom_feats = np.array([atom_features(atom) for atom in mol.GetAtoms()])
molecule_feats = mol_features(mol)

print("Atom features shape:", atom_feats.shape)
print("Molecule features shape:", molecule_feats.shape)
```

Slide 4: Message Passing Neural Network Architecture

The core of neural message passing is the message passing neural network (MPNN) architecture. It consists of multiple layers that update node representations based on their neighbors' features. The MPNN iteratively refines atom representations by exchanging information along chemical bonds, allowing the network to capture complex molecular interactions.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class MPNN(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
        super(MPNN, self).__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output(x)
```

Slide 5: Quantum Chemistry Dataset Preparation

To train our neural message passing model, we need a dataset of molecules with known quantum chemical properties. Common datasets include QM9 and PubChemQC, which contain molecular geometries and properties calculated using density functional theory (DFT). We'll demonstrate how to load and preprocess such a dataset for training our model.

```python
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader

def prepare_qm9_dataset():
    dataset = QM9(root='path/to/data/QM9')
    
    # Split dataset into train, validation, and test sets
    train_dataset = dataset[:10000]
    val_dataset = dataset[10000:11000]
    test_dataset = dataset[11000:]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, val_loader, test_loader

# Usage
train_loader, val_loader, test_loader = prepare_qm9_dataset()

# Example of iterating through the data
for batch in train_loader:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Number of atoms: {batch.x.shape[0]}")
    print(f"Number of bonds: {batch.edge_index.shape[1]}")
    break
```

Slide 6: Training the Neural Message Passing Model

Now that we have our dataset and model architecture, we can train the neural message passing model to predict quantum chemical properties. We'll use a standard training loop with backpropagation and optimize the model using a suitable loss function, such as mean squared error for regression tasks.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch

def train_mpnn(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                val_loss += criterion(out, batch.y).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Usage
model = MPNN(num_layers=3, in_channels=11, hidden_channels=64, out_channels=1)
train_mpnn(model, train_loader, val_loader)
```

Slide 7: Predicting Molecular Properties

After training our neural message passing model, we can use it to predict quantum chemical properties of new molecules. This process involves converting the molecular structure into a graph representation, feeding it through our trained model, and interpreting the output. Let's demonstrate this with an example of predicting the HOMO-LUMO gap of a molecule.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data

def predict_property(model, smiles):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # Extract atom features and bonds
    atom_features = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    bonds = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]).t().contiguous()
    
    # Create PyTorch Geometric data object
    data = Data(x=atom_features, edge_index=bonds)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(data.x, data.edge_index)
    
    return prediction.item()

# Example usage
model = MPNN(num_layers=3, in_channels=11, hidden_channels=64, out_channels=1)
# Assume the model is trained and loaded

# Predict HOMO-LUMO gap for ethanol
smiles = "CCO"
predicted_gap = predict_property(model, smiles)
print(f"Predicted HOMO-LUMO gap for ethanol: {predicted_gap:.2f} eV")
```

Slide 8: Interpreting Model Predictions

Interpreting the predictions of neural message passing models is crucial for understanding their behavior and gaining insights into molecular properties. We can use techniques like attention mechanisms or gradient-based methods to visualize which atoms or bonds contribute most to the predicted properties. Let's implement a simple attention-based interpretation method.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool

class AttentionMPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(AttentionMPNNLayer, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.attention = nn.Linear(2 * in_channels, 1)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        alpha = torch.sigmoid(self.attention(tmp))
        return alpha * self.mlp(tmp)

class InterpretableMP(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
        super(InterpretableMP, self).__init__()
        self.layers = nn.ModuleList([
            AttentionMPNNLayer(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = global_add_pool(x, batch)
        return self.output(x)

# Usage
model = InterpretableMP(num_layers=3, in_channels=11, hidden_channels=64, out_channels=1)
# Train the model...

# Interpret predictions
def interpret_prediction(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in model.layers:
            messages = layer.message(x[edge_index[0]], x[edge_index[1]])
            attention_weights = messages.norm(dim=1)
            x = layer(x, edge_index)
    
    return attention_weights

# Visualize attention weights on molecule
```

Slide 9: Handling Large Molecules

Neural message passing for quantum chemistry often faces challenges when dealing with large molecules due to increased computational complexity. To address this, we can implement techniques like graph coarsening or hierarchical message passing. Let's explore a simple graph coarsening approach to handle larger molecular structures efficiently.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, GCNConv

class CoarsenedMPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(CoarsenedMPNN, self).__init__()
        self.initial_conv = GCNConv(in_channels, hidden_channels)
        self.coarsening_layers = nn.ModuleList([
            SAGPooling(hidden_channels, ratio=0.5)
            for _ in range(num_layers - 1)
        ])
        self.conv_layers = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1)
        ])
        self.final_conv = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.initial_conv(x, edge_index)
        for pool, conv in zip(self.coarsening_layers, self.conv_layers):
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
            x = conv(x, edge_index)
        x = self.final_conv(x, edge_index)
        return x

# Usage
model = CoarsenedMPNN(in_channels=11, hidden_channels=64, out_channels=1, num_layers=3)

# Example of using the model
def process_large_molecule(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.batch)
    return out

# Assume 'large_molecule_data' is a PyG Data object for a large molecule
result = process_large_molecule(model, large_molecule_data)
print(f"Processed large molecule. Output shape: {result.shape}")
```

Slide 10: Multi-task Learning for Quantum Properties

In quantum chemistry, we often want to predict multiple properties simultaneously. Neural message passing can be extended to multi-task learning, where a single model predicts various quantum properties. This approach can leverage the shared information between different properties and potentially improve overall prediction accuracy.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool

class MultiTaskMPNN(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, num_tasks):
        super(MultiTaskMPNN, self).__init__()
        self.layers = nn.ModuleList([
            MessagePassing(aggr='add') for _ in range(num_layers)
        ])
        self.node_mlp = nn.Linear(in_channels, hidden_channels)
        self.edge_mlp = nn.Linear(2 * hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = self.node_mlp(x)
            edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
            edge_features = self.edge_mlp(edge_features)
            x = layer(x, edge_index, edge_features)
        x = global_add_pool(x, batch)
        return self.output(x)

# Usage
model = MultiTaskMPNN(num_layers=3, in_channels=11, hidden_channels=64, num_tasks=5)
# Train the model to predict multiple quantum properties simultaneously
```

Slide 11: Transfer Learning in Quantum Chemistry

Transfer learning is a powerful technique in machine learning where knowledge gained from one task is applied to a different but related task. In the context of quantum chemistry, we can pre-train a neural message passing model on a large dataset of molecules and then fine-tune it for specific tasks or smaller datasets. This approach can significantly improve performance and reduce training time for new tasks.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class TransferableQMNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_tasks):
        super(TransferableQMNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.output(x)

# Pre-training on a large dataset
model = TransferableQMNN(num_features=11, hidden_channels=64, num_tasks=1)
# Train the model on a large dataset

# Fine-tuning for a specific task
model.output = nn.Linear(64, 1)  # Replace the output layer for the new task
# Fine-tune the model on a smaller, task-specific dataset
```

Slide 12: Incorporating 3D Structural Information

Many quantum chemical properties depend on the three-dimensional structure of molecules. We can enhance our neural message passing model by incorporating 3D structural information, such as atomic coordinates and interatomic distances. This approach allows the model to capture spatial relationships between atoms more accurately.

```python
import torch
from torch_geometric.nn import MessagePassing

class Spatial3DMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Spatial3DMPNN, self).__init__(aggr='add')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2 + 1, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        dist = torch.norm(pos_i - pos_j, dim=1).unsqueeze(1)
        return self.mlp(torch.cat([x_i, x_j, dist], dim=1))

# Usage in a full model
class Spatial3DModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_tasks):
        super(Spatial3DModel, self).__init__()
        self.conv = Spatial3DMPNN(num_features, hidden_channels)
        self.output = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        x = self.conv(x, pos, edge_index)
        return self.output(x.mean(dim=0))

# Create and use the model
model = Spatial3DModel(num_features=11, hidden_channels=64, num_tasks=1)
```

Slide 13: Uncertainty Quantification in Predictions

In quantum chemistry predictions, it's crucial to quantify the uncertainty of our model's outputs. This information can guide further experimental work or indicate when the model might be making unreliable predictions. We can implement uncertainty quantification using techniques like ensemble methods or Bayesian neural networks.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class BayesianQMNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_tasks):
        super(BayesianQMNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.output_mean = nn.Linear(hidden_channels, num_tasks)
        self.output_var = nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        mean = self.output_mean(x)
        var = torch.exp(self.output_var(x))  # Ensure positive variance
        return mean, var

# Usage
model = BayesianQMNN(num_features=11, hidden_channels=64, num_tasks=1)

def predict_with_uncertainty(model, data):
    model.eval()
    with torch.no_grad():
        mean, var = model(data.x, data.edge_index, data.batch)
    return mean.item(), var.item()

# Example prediction
mean, variance = predict_with_uncertainty(model, molecule_data)
print(f"Predicted value: {mean:.2f} ± {np.sqrt(variance):.2f}")
```

Slide 14: Additional Resources

For those interested in delving deeper into neural message passing for quantum chemistry, here are some valuable resources:

1. "Neural Message Passing for Quantum Chemistry" by Gilmer et al. (2017) ArXiv: [https://arxiv.org/abs/1704.01212](https://arxiv.org/abs/1704.01212)
2. "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions" by Schütt et al. (2017) ArXiv: [https://arxiv.org/abs/1706.08566](https://arxiv.org/abs/1706.08566)
3. "Quantum-chemical insights from deep tensor neural networks" by Schütt et al. (2017) ArXiv: [https://arxiv.org/abs/1609.08259](https://arxiv.org/abs/1609.08259)
4. "Machine learning in chemistry: Data-driven algorithms, learning systems, and predictions" by Pyzer-Knapp et al. (2018) DOI: 10.1021/acs.accounts.8b00087

These papers provide in-depth discussions on the theory and applications of neural message passing in quantum chemistry, offering valuable insights for both beginners and advanced practitioners in the field.

