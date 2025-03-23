import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

# Load the saved graph
data: Data = torch.load("output1.pt")

# Define a simple GraphSAGE-based encoder model
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize model
model = GNNEncoder(in_channels=data.num_node_features, hidden_channels=64, out_channels=32)

# Forward pass to get embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# Save the embeddings to file
torch.save(embeddings, "output1_node_embeddings.pt")
print("✅ Node embeddings saved to: output1_node_embeddings.pt")

# Optional: print shape or first few rows
print("Embedding shape:", embeddings.shape)
print("First few node embeddings:\n", embeddings[:5])
