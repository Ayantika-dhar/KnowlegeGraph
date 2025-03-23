import torch
import networkx as nx
from torch_geometric.data import Data
import json
import sys
import os

def load_custom_edgelist(path):
    G = nx.DiGraph()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                source = parts[0]
                target = parts[1]
                # Extract relation string safely
                relation = ' '.join(parts[2:]).strip("{}").split(":")[-1].strip(" '\"")
                G.add_edge(source, target, relation=relation)
    return G

def convert_nx_to_pyg(graph):
    node2idx = {node: i for i, node in enumerate(graph.nodes())}
    edge_index = []
    edge_attr = []

    for u, v, data in graph.edges(data=True):
        edge_index.append([node2idx[u], node2idx[v]])
        edge_attr.append(data.get("relation", "related_to"))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.eye(len(node2idx), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data, node2idx, edge_attr

def save_outputs(data, node2idx, edge_attr, output_prefix="graph_data"):
    torch.save(data, f"{output_prefix}.pt")
    with open(f"{output_prefix}_node2idx.json", "w") as f:
        json.dump(node2idx, f, indent=2)
    with open(f"{output_prefix}_edge_attr.txt", "w") as f:
        for rel in edge_attr:
            f.write(f"{rel}\n")
    print(f"\n✅ Saved PyG Data to: {output_prefix}.pt")
    print(f"✅ Saved Node Index Map to: {output_prefix}_node2idx.json")
    print(f"✅ Saved Edge Attributes to: {output_prefix}_edge_attr.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 edge_to_pyg.py <edge_list_file>")
        sys.exit(1)

    path = sys.argv[1]
    output_prefix = path.replace(".edgelist", "")

    # Load and convert
    graph = load_custom_edgelist(path)
    data, node2idx, edge_attr = convert_nx_to_pyg(graph)

    # Display
    print("✅ Loaded PyG graph:")
    print(data)
    print("\n🔢 Node Mapping:")
    for node, idx in node2idx.items():
        print(f"{idx}: {node}")
    print("\n🔗 Edge Relations:")
    print(edge_attr)

    # Save all
    save_outputs(data, node2idx, edge_attr, output_prefix=output_prefix)
