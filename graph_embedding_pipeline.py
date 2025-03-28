import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import stanza
import networkx as nx
import json
import os

# Initialize Stanza once globally
stanza.download('en', verbose=False)
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)

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

def extract_svo_triples(doc):
    triples = []
    for sentence in doc.sentences:
        words = {word.id: word.text for word in sentence.words}
        for word in sentence.words:
            if word.deprel in ["nsubj", "nsubj:pass"]:
                subject = words.get(word.id)
                verb = words.get(word.head)
                obj = None
                for w in sentence.words:
                    if w.head == word.head and w.deprel in ["obj", "obl", "xcomp", "ccomp"]:
                        obj = words.get(w.id)
                if subject and verb and obj:
                    triples.append((subject.lower(), verb.lower(), obj.lower()))
    return triples

def build_graph(triples):
    G = nx.DiGraph()
    for subj, rel, obj in triples:
        G.add_edge(subj, obj, relation=rel)
    return G

def graph_to_pyg(graph):
    node2idx = {node: i for i, node in enumerate(graph.nodes())}
    edge_index = []
    for u, v in graph.edges():
        edge_index.append([node2idx[u], node2idx[v]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.eye(len(node2idx), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def text_to_graph_embedding(text, out_dim=768, hidden_dim=64):
    if not text.strip():
        return torch.zeros((1, out_dim))

    doc = nlp(text)
    triples = extract_svo_triples(doc)
    if not triples:
        return torch.zeros((1, out_dim))

    graph = build_graph(triples)
    data = graph_to_pyg(graph)

    model = GNNEncoder(in_channels=data.num_node_features, hidden_channels=hidden_dim, out_channels=out_dim)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        return embeddings.mean(dim=0).unsqueeze(0)  # Shape: (1, out_dim)

# This function is to be used in ASRSentiment/OCRSentiment pipeline like:
# embedding = text_to_graph_embedding(text)
# return {'features': embedding, 'asr': text, ...} where applicable
