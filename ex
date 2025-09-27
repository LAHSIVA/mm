import random

def h1(x, n_shingles):
    return (x + 2) % n_shingles

def h2(x, n_shingles):
    return (3 * x + 1) % n_shingles

def h3(x, n_shingles):
    return (x + 4) % n_shingles

def minhash_signature_matrix(shingle_doc_matrix, hash_functions):
    n_shingles = len(shingle_doc_matrix)
    n_docs = len(shingle_doc_matrix[0])
    n_hashes = len(hash_functions)
    signature_matrix = [[float('inf')] * n_docs for _ in range(n_hashes)]

    for i in range(n_shingles):
        for j in range(n_docs):
            if shingle_doc_matrix[i][j] == 1:
                for k in range(n_hashes):
                    hash_val = hash_functions[k](i, n_shingles)
                    if hash_val < signature_matrix[k][j]:
                        signature_matrix[k][j] = hash_val
    return signature_matrix


def get_shingles(text, k=3):
    words = text.lower().split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(shingle)
    return list(shingles)

def build_shingle_vocab(docs, k=3):
    vocab = set()
    for doc in docs:
        vocab.update(get_shingles(doc, k))
    return sorted(list(vocab))

def create_shingle_doc_matrix(docs, k=3):
    vocab = build_shingle_vocab(docs, k)
    n_shingles = len(vocab)
    n_docs = len(docs)
    matrix = [[0] * n_docs for _ in range(n_shingles)]

    for i, shingle in enumerate(vocab):
        for j, doc in enumerate(docs):
            if shingle in get_shingles(doc, k):
                matrix[i][j] = 1
    return matrix, vocab

def jaccard(s1, s2):
    inter = len(set(s1).intersection(s2))
    union = len(set(s1).union(s2))
    if union == 0:
        return 0.0
    return inter / union

def estimate_jaccard(sig1, sig2):
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)


documents = [
    "This is document one",
    "This is document two",
    "Another document here",
    "This is document one and two"
]

hash_functions = [h1, h2, h3]

shingle_doc_matrix_example, shingle_vocab = create_shingle_doc_matrix(documents, k=2)

print("Shingle Vocabulary:", shingle_vocab)
print("\nShingle Document Matrix:")
for row in shingle_doc_matrix_example:
    print(row)

signature_matrix_example = minhash_signature_matrix(shingle_doc_matrix_example, hash_functions)
print("\nMinHash Signature Matrix for Example Documents:")
for row in signature_matrix_example:
    print(row)

print("\nEstimated Jaccard Similarity from MinHash Signatures:")
signatures_per_doc = [[signature_matrix_example[j][i] for j in range(len(signature_matrix_example))] for i in range(len(signature_matrix_example[0]))]

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        est_sim = estimate_jaccard(signatures_per_doc[i], signatures_per_doc[j])
        print(f"Estimated Jaccard(Doc{i+1}, Doc{j+1}): {est_sim:.4f}")

print("\nActual Jaccard Similarity:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        shingles1 = set(get_shingles(documents[i], k=2))
        shingles2 = set(get_shingles(documents[j], k=2))
        actual_sim = jaccard(shingles1, shingles2)
        print(f"Actual Jaccard(Doc{i+1}, Doc{j+1}): {actual_sim:.4f}")


===========================


import random

def generate_random_permutations(n_rows, n_perms=3, seed=42):
    random.seed(seed)
    permutations = []
    i=0
    while i < n_perms:
        perm = list(range(n_rows))
        random.shuffle(perm)
        if perm not in permutations:
          permutations.append(perm)
          i+=1
    return permutations

def minhash_with_given_permutations(matrix, permutations):
    n_docs = len(matrix[0])
    signature_matrix = [[0] * n_docs for _ in range(len(permutations))]

    for perm_idx, perm in enumerate(permutations):
        for doc in range(n_docs):
            for pos, row in enumerate(perm):
                if matrix[row][doc] == 1:
                    signature_matrix[perm_idx][doc] = pos + 1
                    break
    return signature_matrix

def estimate_jaccard_from_signatures(sig_matrix):
    n_perms = len(sig_matrix)
    n_docs = len(sig_matrix[0])
    signatures_per_doc = [[sig_matrix[j][i] for j in range(n_perms)] for i in range(n_docs)]

    est_sim = {}
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            matches = sum(1 for k in range(n_perms) if signatures_per_doc[i][k] == signatures_per_doc[j][k])
            est_sim[(i, j)] = matches / n_perms
    return est_sim

def actual_jaccard(matrix):
    n_docs = len(matrix[0])
    sims = {}
    for i in range(n_docs):
        set_i = {r for r in range(len(matrix)) if matrix[r][i] == 1}
        for j in range(i+1, n_docs):
            set_j = {r for r in range(len(matrix)) if matrix[r][j] == 1}
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            sims[(i, j)] = inter / union if union > 0 else 0.0
    return sims

shingle_doc_matrix = [
    [1,0,1,0],
    [1,0,0,1],
    [0,1,0,1],
    [0,1,0,1],
    [0,1,0,1],
    [1,0,1,0],
    [1,0,1,0]
]

permutations = [
    [0,2,6,5,1,4,3],
    [3,1,0,2,5,6,4],
    [2,3,6,5,0,1,4]
]
# permutations = generate_random_permutations(n_rows=len(shingle_doc_matrix), n_perms=3, seed=101)

print("Random Permutations:")
for p in permutations:
    print(p)

# Step 1: Signature matrix
signature_matrix = minhash_with_given_permutations(shingle_doc_matrix, permutations)
print("Signature Matrix (using given permutations):")
for row in signature_matrix:
    print(row)

# Step 2: Estimated Jaccard
est_sims = estimate_jaccard_from_signatures(signature_matrix)
print("\nEstimated Jaccard Similarities:")
for (i, j), sim in est_sims.items():
    print(f"Doc{i+1} vs Doc{j+1}: {sim:.3f}")

# Step 3: Actual Jaccard
act_sims = actual_jaccard(shingle_doc_matrix)
print("\nActual Jaccard Similarities:")
for (i, j), sim in act_sims.items():
    print(f"Doc{i+1} vs Doc{j+1}: {sim:.3f}")


======================================


#pageRank
def pagerank(graph, damping_factor=0.5, max_iterations=100, tolerance=1e-6):
    """
    Calculate PageRank for nodes in a directed graph.
    graph: Dictionary where keys are nodes and values are lists of nodes they link to
    damping_factor: Probability of following a link (typically 0.85)
    max_iterations: Maximum number of iterations
    tolerance: Convergence threshold
    """
    # Initialize variables
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    # Initialize PageRank scores
    pr = {node: 1/n for node in nodes}
    temp_pr = pr.copy()

    for _ in range(max_iterations):
        # Calculate new PageRank for each node
        for node in nodes:
            # Sum PageRank of incoming nodes
            incoming_pr = 0
            for other_node in nodes:
                if node in graph.get(other_node, []):
                    outlinks = len(graph[other_node])
                    incoming_pr += pr[other_node] * (1 / outlinks)

            # Update PageRank with damping factor
            temp_pr[node] = (1 - damping_factor) / n + damping_factor * incoming_pr

        # Check for convergence
        total_diff = sum(abs(temp_pr[node] - pr[node]) for node in nodes)
        pr = temp_pr.copy()

        if total_diff < tolerance:
            break

    return pr

# Example usage
if __name__ == "__main__":
    # Example graph: {node: [outgoing links]}
    example_graph = {
        'A': ['B'],
        'B': ['A','C'],
        'C': ['B'],
    }

    ranks = pagerank(example_graph)
    ranks = {node: rank for node, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)}
    print("PageRank Scores:")
    for node, rank in ranks.items():
        print(f"Node {node}: {rank:.4f}")


==================================



#eigen vector page rank
import numpy as np

# Example graph: A->B,C; B->C; C->A; D->C
nodes = ['A', 'B', 'C']
n = len(nodes)
transition = np.zeros((n, n))

# Fill transition matrix (row: from, col: to)
links = {'A': ['B'], 'B': ['A','C'], 'C': ['B']}
node_idx = {node: i for i, node in enumerate(nodes)}

for from_node, to_nodes in links.items():
    out_degree = len(to_nodes)
    for to_node in to_nodes:
        transition[node_idx[from_node], node_idx[to_node]] = 1 / out_degree

# Google matrix with damping=0.85
damping = 0.5
google_matrix = damping * transition + (1 - damping) / n * np.ones((n, n))

# Find eigenvectors; take the one with eigenvalue closest to 1
eigenvalues, eigenvectors = np.linalg.eig(google_matrix.T)  # Transpose for right eigenvector
print(eigenvalues)
print(eigenvectors)
idx = np.argmin(abs(eigenvalues - 1))
print(idx)
pr_vector = abs(eigenvectors[:, idx])
print(pr_vector)
pr_vector /= pr_vector.sum()  # Normalize
print(pr_vector)

# Output
ranks = {nodes[i]: pr_vector[i] for i in range(n)}
print(ranks)


======================================


import numpy as np

# ==============================
# Step 1: Example documents
# ==============================
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat played with the dog",
    "dogs and cats are friends"
]

# ==============================
# Step 2: Preprocess & Vocabulary
# ==============================
def tokenize(doc):
    return doc.lower().split()

# Build vocabulary
vocab = sorted(set(word for doc in docs for word in tokenize(doc)))
word_index = {w: i for i, w in enumerate(vocab)}

# ==============================
# Step 3: Build Term–Document Matrix
# ==============================
A = np.zeros((len(vocab), len(docs)), dtype=float)

for j, doc in enumerate(docs):
    for word in tokenize(doc):
        A[word_index[word], j] += 1

print("Term–Document Matrix (A):")
print(A)

# ==============================
# Step 4: Singular Value Decomposition (SVD)
# ==============================
# A = U Σ V^T
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Convert singular values into diagonal Σ
Sigma = np.diag(s)

print("\nU (terms -> concepts):\n", U)
print("\nΣ (singular values):\n", Sigma)
print("\nV^T (docs -> concepts):\n", Vt)

# ==============================
# Step 5: Dimensionality Reduction (LSI)
# ==============================
k = 2  # latent dimension
U_k = U[:, :k]
Sigma_k = Sigma[:k, :k]
Vt_k = Vt[:k, :]

# Reduced doc vectors in LSI space
doc_vectors = np.dot(Sigma_k, Vt_k).T  # shape: (n_docs, k)
print("\nReduced Document Representations (LSI space):\n", doc_vectors)

# ==============================
# Step 6: Query Projection
# ==============================
query = "cat and dog play together"
q_vec = np.zeros((len(vocab), 1))

for word in tokenize(query):
    if word in word_index:
        q_vec[word_index[word], 0] += 1

# Project query into LSI space: q' = (q^T U_k) Σ_k^-1
q_lsi = np.dot(np.dot(q_vec.T, U_k), np.linalg.inv(Sigma_k))
print("\nQuery Representation (LSI space):\n", q_lsi)

# ==============================
# Step 7: Cosine Similarity
# ==============================
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nSimilarity of query with each document:")
for i, doc_vec in enumerate(doc_vectors):
    sim = cosine_sim(q_lsi.flatten(), doc_vec)
    print(f"Doc{i+1}: {sim:.3f}")

========================


import networkx as nx
import matplotlib.pyplot as plt

# Define the example graph
graph = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['A'],
    'D': ['C']
}

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for node, neighbors in graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Set up the plot
plt.figure(figsize=(8, 6))

# Draw the graph
nx.draw(G, with_labels=True,arrows=True)

# Add a title
plt.title("Directed Graph Visualization")

# Show the plot
plt.show()
