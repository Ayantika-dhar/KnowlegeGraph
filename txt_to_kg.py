import stanza
import networkx as nx
import sys
import os

# Load Stanza English model (download only if not already done)
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')

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
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, relation=rel)
    return G

def write_edge_list_with_relation(graph, output_path):
    print("Edge data:")
    for u, v, d in graph.edges(data=True):
        print(f"{u} -> {v}, data = {d}")
    
    with open(output_path, 'w') as f:
        for u, v, data in graph.edges(data=True):
            rel = data.get('relation', 'related_to')
            f.write(f"{u} {v} {{'relation': '{rel}'}}\n")
    print(f"\nEdge list written to: {output_path}")

def process_txt_file(input_path, output_edge_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    doc = nlp(text)
    triples = extract_svo_triples(doc)
    graph = build_graph(triples)

    if output_edge_path:
        write_edge_list_with_relation(graph, output_edge_path)
    
    return graph

# CLI Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 txt_to_kg.py <input_txt> [output_edge_file]")
        sys.exit(1)

    input_txt = sys.argv[1]
    output_edge_file = sys.argv[2] if len(sys.argv) > 2 else None
    process_txt_file(input_txt, output_edge_file)
