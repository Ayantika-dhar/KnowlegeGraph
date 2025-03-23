import stanza
import networkx as nx

# Load Stanza English model (download only if not already done)
stanza.download('en', verbose=False)
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)

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
    with open(output_path, 'w') as f:
        for u, v, data in graph.edges(data=True):
            rel = data.get('relation', 'related_to')
            f.write(f"{u} {v} {{'relation': '{rel}'}}\n")

def process_txt_string(text, output_edge_path):
    doc = nlp(text)
    triples = extract_svo_triples(doc)
    graph = build_graph(triples)
    write_edge_list_with_relation(graph, output_edge_path)
    return graph
