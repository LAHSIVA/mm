# ------------ Boolean WS -----------

import re
from collections import defaultdict, Counter

# Preprocessing function

def preprocess(text):
    # Lowercase and extract words
    return re.findall(r'\w+', text.lower())

# Build Inverted Index

def build_inverted_index(docs, stop_words=set()):
    index = defaultdict(set)  # term -> set of docIDs
    
    for doc_id, text in enumerate(docs):
        terms = preprocess(text)

        for term in terms:
            if term not in stop_words:
                index[term].add(doc_id)
    
    return index


# Boolean Query Evaluation

def boolean_query(query, index, total_docs):
    tokens = query.lower().split()
    
    def eval_token(token):
        if token in ["and", "or", "not"]:
            return token
        
        return index.get(token, set()) 
    
    # Infix to Postfix using Shunting-Yard 
    output, stack = [], []
    precedence = {"not": 3, "and": 2, "or": 1}
    
    for token in tokens:
        if token in ["and", "or", "not"]:
            while stack and precedence.get(stack[-1], 0) >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
    
        else:
            output.append(eval_token(token))
    
    while stack:
        output.append(stack.pop())
    
    # Evaluate postfix expression
    eval_stack = []
    for token in output:
        if isinstance(token, set):
            eval_stack.append(token)
        else:
            if token == "not":
                s = eval_stack.pop()
                all_docs = set(range(total_docs))
                eval_stack.append(all_docs - s)
            else:
                b = eval_stack.pop()
                a = eval_stack.pop()
                if token == "and":
                    eval_stack.append(a & b)
                elif token == "or":
                    eval_stack.append(a | b)
    
    return eval_stack.pop()


docs = [
    "Information retrieval is the process of obtaining information.",
    "Retrieval models include boolean, vector space, and probabilistic.",
    "Boolean retrieval uses logical operators like AND, OR, NOT.",
    "Vector space model represents documents as vectors.",
    "Probabilistic model estimates relevance using probability theory.",
    "Stop words like the, is, in, and of appear frequently.",
    "This document talks about information systems and retrieval.",
    "Another example document with simple retrieval process.",
    "Indexing is important in information retrieval.",
    "Final document is here."
]

# Step 1: Find top 10 frequent words → stop words
all_terms = [term for doc in docs for term in preprocess(doc)]
stop_words = set([w for w, _ in Counter(all_terms).most_common(10)])

# Step 2: Build index without stop words (for size reporting only)
index_no_stop = build_inverted_index(docs, stop_words)

# Step 3: Build full index (with stop words) for querying
index_full = build_inverted_index(docs)

# Report
print("Stop Words:", stop_words)
print("Index Size after removing stop words:", len(index_no_stop))

# Step 4: Run Boolean Queries
queries = [
    "information AND retrieval",
    "retrieval OR probabilistic",
    "information AND NOT boolean",
    "vector AND space",
    "boolean OR indexing"
]

for q in queries:
    result = boolean_query(q, index_full, len(docs))
    print(f"Query: {q} → Docs: {result}")



# -------------------------------------- Plagiarism detector WS -----------------------------------------

import re
import math
from collections import Counter
from typing import List, Tuple, Dict, Set

# Basic text utilities

WORD_RE = re.compile(r"[a-z0-9]+")

def normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def title_from_record(record_text: str) -> Tuple[str, str]:
    # Split into (title, content)
    txt = record_text.strip()
    if ":" in txt:
        title, rest = txt.split(":", 1)
        return normalize_spaces(title), rest.strip()
    return "", txt


# Similarity functions

def binary_distance(u: str, v: str) -> int:
    # 0 if identical titles (normalized), else 1
    def clean(t: str) -> str:
        return normalize_spaces(re.sub(r"[^a-z0-9 ]", " ", t.lower()))
    
    if clean(u) == clean(v):
        return 0
    else:
        return 1

def cosine_sim(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    dot = 0.0
    
    for term, val in vec_a.items():
        dot += val * vec_b.get(term, 0.0)
    
    na = math.sqrt(sum(v*v for v in vec_a.values()))
    nb = math.sqrt(sum(v*v for v in vec_b.values()))
    
    if na == 0 or nb == 0:
        return 0.0
    
    return dot / (na * nb)

def shingles(tokens: List[str], k: int = 3) -> Set[Tuple[str, ...]]:
    sh = set()
    
    if len(tokens) >= k:
        for i in range(len(tokens) - k + 1):
            sh.add(tuple(tokens[i:i+k]))
    
    return sh

def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    
    if not a or not b:
        return 0.0
    
    inter = len(a.intersection(b))
    union = len(a.union(b))
    
    return inter / union


# TF-IDF weighting

def build_tfidf_index(docs_tokens: List[List[str]]):
    N = len(docs_tokens)
    df = Counter()
    
    for toks in docs_tokens:
        for term in set(toks):
            df[term] += 1
    
    idf = {}
    for term, dfk in df.items():
        idf[term] = math.log((N + 1) / (0.5 + dfk))

    def vectorize(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        L = len(tokens) if tokens else 1
        vec = {}
        for term, c in tf.items():
            if term in idf:
                vec[term] = (c / L) * idf[term]
            else:
                vec[term] = (c / L) * math.log((N + 1) / 0.5)
        return vec

    doc_vecs = []
    for toks in docs_tokens:
        doc_vecs.append(vectorize(toks))

    return doc_vecs, vectorize


# BM25 scoring

def build_bm25_index(docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
    N = len(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens) / N if N else 0.0
    df = Counter()
    
    for toks in docs_tokens:
        for term in set(toks):
            df[term] += 1
    
    idf = {}
    for t, dfk in df.items():
        idf[t] = math.log((N - dfk + 0.5) / (dfk + 0.5) + 1.0)

    def score(doc_tokens: List[str], query_tokens: List[str]) -> float:
        tf = Counter(doc_tokens)
        L = len(doc_tokens) if doc_tokens else 0
        
        if avgdl > 0:
            denom_norm = k1 * (1 - b + b * (L / avgdl))
        else:
            denom_norm = k1
        
        s = 0.0
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if f == 0:
                continue
            if t in idf:
                idf_val = idf[t]
            else:
                idf_val = math.log((N + 0.5) / 0.5 + 1.0)
            s += idf_val * (f * (k1 + 1)) / (f + denom_norm)
        
        return s

    return score


# Main checker

def plagiarism_checker(
    db_records: List[Tuple[str, str]],
    new_records: List[Tuple[str, str]],
    alpha_cos: float = 0.85,
    beta_jaccard: float = 0.80,
    tau_bm25: float = 6.0,
    k_shingle: int = 3
):
    # Database preparation
    db_titles = []
    db_contents = []
    db_tokens = []
    for t, c in db_records:
        db_titles.append(normalize_spaces(t))
        db_contents.append(c)
        db_tokens.append(tokenize(c))

    # Build TF-IDF and BM25
    doc_vecs, vectorize_query = build_tfidf_index(db_tokens)
    bm25_score = build_bm25_index(db_tokens)

    # Process each new doc
    for idx, (new_title, new_content) in enumerate(new_records, start=1):
        print(f"\n=== Checking new doc {idx} ===")
        print("Title:", new_title)
        print("Content:", new_content, "\n")

        new_tokens = tokenize(new_content)

        # A) Title check
        title_hits = []
        for i, t in enumerate(db_titles):
            if binary_distance(t, new_title) == 0:
                title_hits.append(i)
        if title_hits:
            print(f"A) Title exact match → duplicate with DB docs {title_hits}")
        else:
            print("A) Title exact match → no")

        # B + C) Cosine similarity
        qvec = vectorize_query(new_tokens)
        cos_sims = []
        for i in range(len(db_tokens)):
            cos_sims.append((i, cosine_sim(qvec, doc_vecs[i])))
        cos_sims.sort(key=lambda x: x[1], reverse=True)
        print("C) Cosine top-3:", cos_sims[:3])
        print("   Duplicate?", cos_sims[0][1] >= alpha_cos, "(alpha=", alpha_cos, ")")

        # D) Jaccard shingles
        new_sh = shingles(new_tokens, k_shingle)
        jac_sims = []
        for i in range(len(db_tokens)):
            sim = jaccard(new_sh, shingles(db_tokens[i], k_shingle))
            jac_sims.append((i, sim))
        jac_sims.sort(key=lambda x: x[1], reverse=True)
        print(f"D) Jaccard(k={k_shingle}) top-3:", jac_sims[:3])
        print("   Duplicate?", jac_sims[0][1] >= beta_jaccard, "(beta=", beta_jaccard, ")")

        # E) BM25
        bm25_scores = []
        for i in range(len(db_tokens)):
            bm25_scores.append((i, bm25_score(db_tokens[i], new_tokens)))
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        print("E) BM25 top-3:", bm25_scores[:3])
        print("   Duplicate?", bm25_scores[0][1] >= tau_bm25, "(tau=", tau_bm25, ")")

        # Final decision
        final = (
            bool(title_hits)
            or cos_sims[0][1] >= alpha_cos
            or jac_sims[0][1] >= beta_jaccard
            or bm25_scores[0][1] >= tau_bm25
        )
        print("FINAL decision:", "DUPLICATE" if final else "NOT duplicate")



db_lines = [
    "Information requirement: query considers the user feedback as information requirement to search.",
    "Information retrieval: query depends on the model of information retrieval used.",
    "Prediction problem: Many problems in information retrieval can be viewed as prediction problems",
    "Search: A search engine is one of applications of information retrieval models."
]

db_records = []
for l in db_lines:
    db_records.append(title_from_record(l))

new_lines = [
    "Feedback: feedback is typically used by the system to modify the query and improve prediction",
    "information retrieval: ranking in information retrieval algorithms depends on user query",
    "Predictionssss: Many problems in information retrieval can be viewed as prediction problems"
]

new_records = []
for l in new_lines:
    new_records.append(title_from_record(l))


plagiarism_checker(db_records, new_records)







# -------------------- BIM PHASE 1, 2 -------------------------

import math

# Preprocessing
def preprocess(docs):
    vocab_set = set()
    processed = []

    for doc in docs:
        tokens = set(doc.lower().split())
        processed.append(tokens)
        vocab_set.update(tokens)

    vocab = sorted(vocab_set)

    binary_matrix = []
    for doc_tokens in processed:
        row = []
        for term in vocab:
            if term in doc_tokens:
                row.append(1)
            else:
                row.append(0)
        binary_matrix.append(row)

    return vocab, binary_matrix


# Phase I estimation (no relevance info)

def phase1_estimate(query_terms, vocab, binary_matrix):
    N_d = len(binary_matrix)
    estimates = {}

    for term in query_terms:
        if term not in vocab:
            continue

        term_idx = vocab.index(term)

        # d_k = document frequency
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)

        # p_k ≈ 0.5
        p_k = 0.5

        # q_k with smoothing
        q_k = (d_k + 0.5) / (N_d + 1)

        estimates[term] = {"d_k": d_k, "p_k": p_k, "q_k": q_k}

    return estimates



# Phase II estimation (with relevance info)

def phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs):
    N_d = len(binary_matrix)
    N_r = len(relevant_docs)
    estimates = {}

    for term in query_terms:
        if term not in vocab:
            continue

        term_idx = vocab.index(term)

        # r_k = number of relevant docs containing term
        r_k = 0
        for doc_id in relevant_docs:
            if binary_matrix[doc_id][term_idx] == 1:
                r_k += 1

        # d_k = total docs containing term
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)

        # p_k (with smoothing)
        p_k = (r_k + 0.5) / (N_r + 1)

        # q_k (with smoothing)
        q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)

        estimates[term] = {"r_k": r_k, "d_k": d_k, "N_r": N_r, "p_k": p_k, "q_k": q_k}

    return estimates



# Calculate Retrieval Status Value (RSV)

def calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix):
    rsv = 0
    for term in query_terms:
        if term not in estimates:
            continue

        term_idx = vocab.index(term)
        p_k = estimates[term]["p_k"]
        q_k = estimates[term]["q_k"]

        if binary_matrix[doc_id][term_idx] == 1:
            if p_k > 0 and q_k > 0:
                rsv += math.log(p_k / q_k)
        else:
            if p_k < 1 and q_k < 1:
                rsv += math.log((1 - p_k) / (1 - q_k))
    return rsv



# Search functions

def search_phase1(query, vocab, binary_matrix, top_k=5):
    query_terms = query.lower().split()
    estimates = phase1_estimate(query_terms, vocab, binary_matrix)

    doc_scores = []
    for doc_id in range(len(binary_matrix)):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


def search_phase2(query, vocab, binary_matrix, relevant_docs, top_k=5):
    query_terms = query.lower().split()
    estimates = phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs)

    doc_scores = []
    for doc_id in range(len(binary_matrix)):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


docs = [
    "information retrieval system",
    "database search query",
    "information system database",
    "web search engine",
    "query processing system"
]

vocab, binary_matrix = preprocess(docs)

query = "information system"
query_terms = query.split()

print("\n\n")
print("PHASE I (No Relevance Info)")
est1 = phase1_estimate(query_terms, vocab, binary_matrix)

for term, e in est1.items():
    print(f"{term}: d_k={e['d_k']}, p_k={e['p_k']:.3f}, q_k={e['q_k']:.3f}")

results1 = search_phase1(query, vocab, binary_matrix)
print("Phase I Results:", results1)

print("\n\n")
print("PHASE II (With Relevance Feedback)")
relevant_docs = [0, 2]  # Assume docs 0,2 are relevant
est2 = phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs)

for term, e in est2.items():
    print(f"{term}: r_k={e['r_k']}, d_k={e['d_k']}, N_r={e['N_r']}, p_k={e['p_k']:.3f}, q_k={e['q_k']:.3f}")

results2 = search_phase2(query, vocab, binary_matrix, relevant_docs)
print("Phase II Results:", results2)





# --------------------- Vector Space Model CJDD --------------

import math
from collections import Counter

def preprocess(docs):
    processed_docs = []
    vocab_set = set()

    for doc in docs:
        tokens = doc.lower().split()
        processed_docs.append(tokens)
        vocab_set.update(tokens)

    vocab = sorted(vocab_set)

    # Build TF matrix
    tf_matrix = []
    for doc_tokens in processed_docs:
        counts = Counter(doc_tokens)
        row = []
        for term in vocab:
            row.append(counts.get(term, 0))
        tf_matrix.append(row)

    return vocab, tf_matrix


# IDF and TF-IDF

def compute_idf(tf_matrix, vocab):
    N = len(tf_matrix)
    idf = []
    for term_idx in range(len(vocab)):
        df = 0
        for doc in tf_matrix:
            if doc[term_idx] > 0:
                df += 1
        if df > 0:
            idf.append(math.log(N / df))
        else:
            idf.append(0)
    return idf


def compute_tfidf(tf_matrix, idf):
    tfidf_matrix = []
    for doc in tf_matrix:
        row = []
        for i, tf in enumerate(doc):
            row.append(tf * idf[i])
        tfidf_matrix.append(row)
    return tfidf_matrix


def query_to_vector(query, vocab, idf):
    query_tf = Counter(query.lower().split())
    qvec = []
    for i, term in enumerate(vocab):
        qvec.append(query_tf.get(term, 0) * idf[i])
    return qvec


# Similarity Measures

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)


def jaccard_coefficient(v1, v2):
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    inter = sum(a & b for a, b in zip(b1, b2))
    union = sum(a | b for a, b in zip(b1, b2))
    return inter / union if union > 0 else 0


def dice_coefficient(v1, v2):
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    inter = sum(a & b for a, b in zip(b1, b2))
    total = sum(b1) + sum(b2)
    return (2 * inter) / total if total > 0 else 0


def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def search(query, vocab, tfidf_matrix, idf, similarity="cosine", top_k=5):
    qvec = query_to_vector(query, vocab, idf)

    similarities = []
    for doc_id, dvec in enumerate(tfidf_matrix):
        if similarity == "cosine":
            sim = cosine_similarity(qvec, dvec)
        elif similarity == "jaccard":
            sim = jaccard_coefficient(qvec, dvec)
        elif similarity == "dice":
            sim = dice_coefficient(qvec, dvec)
        elif similarity == "dot":
            sim = dot_product(qvec, dvec)
        else:
            sim = cosine_similarity(qvec, dvec)
        similarities.append((doc_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


docs = [
    "information retrieval system",
    "machine learning data",
    "web search engine"
]

vocab, tf_matrix = preprocess(docs)
idf = compute_idf(tf_matrix, vocab)
tfidf_matrix = compute_tfidf(tf_matrix, idf)

query = "information search"
print("Cosine:", search(query, vocab, tfidf_matrix, idf, "cosine"))
print("Jaccard:", search(query, vocab, tfidf_matrix, idf, "jaccard"))
print("Dice:", search(query, vocab, tfidf_matrix, idf, "dice"))
print("Dot Product:", search(query, vocab, tfidf_matrix, idf, "dot"))




# ---------------------- TERM DOCUMENT MATRIX - BOOLEAN ----------------------

def build_matrix(documents):
    all_terms = set()
    processed_docs = []

    # Tokenize docs and collect vocab
    for doc in documents:
        tokens = doc.lower().split()
        processed_docs.append(tokens)
        all_terms.update(tokens)

    vocab = sorted(all_terms)

    # Build binary matrix
    term_doc_matrix = []
    for term in vocab:
        row = []
        for doc_tokens in processed_docs:
            if term in doc_tokens:
                row.append(1)
            else:
                row.append(0)
        term_doc_matrix.append(row)

    return vocab, term_doc_matrix



def get_term_vector(term, vocab, term_doc_matrix, num_docs):
    term = term.lower()
    if term not in vocab:
        return [0] * num_docs
    term_idx = vocab.index(term)
    return term_doc_matrix[term_idx]


def boolean_and(vec1, vec2):
    return [a & b for a, b in zip(vec1, vec2)]


def boolean_or(vec1, vec2):
    return [a | b for a, b in zip(vec1, vec2)]


def boolean_not(vec):
    return [1 - x for x in vec]


# Boolean Search

def search(query, vocab, term_doc_matrix, num_docs):
    query = query.lower().strip()

    # Single term
    if " " not in query:
        result_vector = get_term_vector(query, vocab, term_doc_matrix, num_docs)

    # AND
    elif " and " in query:
        terms = [t.strip() for t in query.split(" and ")]
        result_vector = get_term_vector(terms[0], vocab, term_doc_matrix, num_docs)
        for term in terms[1:]:
            term_vec = get_term_vector(term, vocab, term_doc_matrix, num_docs)
            result_vector = boolean_and(result_vector, term_vec)

    # OR
    elif " or " in query:
        terms = [t.strip() for t in query.split(" or ")]
        result_vector = get_term_vector(terms[0], vocab, term_doc_matrix, num_docs)
        for term in terms[1:]:
            term_vec = get_term_vector(term, vocab, term_doc_matrix, num_docs)
            result_vector = boolean_or(result_vector, term_vec)

    # NOT
    elif " not " in query:
        parts = query.split(" not ")
        pos_term = parts[0].strip()
        neg_term = parts[1].strip()

        pos_vec = get_term_vector(pos_term, vocab, term_doc_matrix, num_docs)
        neg_vec = get_term_vector(neg_term, vocab, term_doc_matrix, num_docs)
        neg_vec = boolean_not(neg_vec)

        result_vector = boolean_and(pos_vec, neg_vec)

    else:
        result_vector = [0] * num_docs

    # Return doc indices where result=1
    results = []
    for i, val in enumerate(result_vector):
        if val == 1:
            results.append(i)
    return results


def print_matrix(vocab, term_doc_matrix, documents):
    print("Term-Document Matrix:")
    print("Terms\\Docs", end="")
    for i in range(len(documents)):
        print(f"\tD{i}", end="")
    print()

    for i, term in enumerate(vocab):
        print(f"{term:<10}", end="")
        for val in term_doc_matrix[i]:
            print(f"\t{val}", end="")
        print()


docs = [
    "information retrieval system",
    "database search query",
    "information system database",
    "web search engine",
    "query processing system"
]

vocab, term_doc_matrix = build_matrix(docs)
print_matrix(vocab, term_doc_matrix, docs)

print("\nSearch Results:")
print("'information':", search("information", vocab, term_doc_matrix, len(docs)))
print("'information and system':", search("information and system", vocab, term_doc_matrix, len(docs)))
print("'search or query':", search("search or query", vocab, term_doc_matrix, len(docs)))
print("'system not database':", search("system not database", vocab, term_doc_matrix, len(docs)))




# ---------------- Inverted Index Boolean ----------------

# Build inverted index

def build_inverted_index(docs):
    index = {}
    for i, doc in enumerate(docs):
        tokens = set(doc.lower().split())
        for term in tokens:
            if term not in index:
                index[term] = []
            index[term].append(i)
    return index


# Basic Boolean operations

def get_postings(index, term):
    return index.get(term.lower(), [])


def AND(list1, list2):
    return [x for x in list1 if x in list2]


def OR(list1, list2):
    return sorted(set(list1 + list2))


def NOT(posting_list, total_docs):
    all_docs = list(range(total_docs))
    return [x for x in all_docs if x not in posting_list]


def optimize_terms(index, terms, operation="and"):
    term_lengths = []
    for term in terms:
        postings = get_postings(index, term)
        term_lengths.append((term, len(postings)))

    if operation == "and":
        # For AND: shortest posting lists first
        sorted_terms = sorted(term_lengths, key=lambda x: x[1])
    else:  # OR
        # For OR: longest posting lists first
        sorted_terms = sorted(term_lengths, key=lambda x: x[1], reverse=True)

    return [term for term, _ in sorted_terms]


# Search

def search(index, docs, query):
    q = query.lower().strip()
    total_docs = len(docs)

    if " and " in q:
        terms = [t.strip() for t in q.split(" and ")]
        terms = optimize_terms(index, terms, "and")  # shortest first
        result = get_postings(index, terms[0])
        for term in terms[1:]:
            result = AND(result, get_postings(index, term))
            if not result:
                break
        return result

    elif " or " in q:
        terms = [t.strip() for t in q.split(" or ")]
        terms = optimize_terms(index, terms, "or")  # longest first
        result = get_postings(index, terms[0])
        for term in terms[1:]:
            result = OR(result, get_postings(index, term))
        return result

    elif " not " in q:
        pos, neg = q.split(" not ")
        pos_list = get_postings(index, pos.strip())
        neg_list = get_postings(index, neg.strip())
        return AND(pos_list, NOT(neg_list, total_docs))

    else:
        return get_postings(index, q)


docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]
index = build_inverted_index(docs)

print("Index:", index)

print("\nQuery: 'cat and bird and dog'")
terms = ["cat", "bird", "dog"]
print("Posting list sizes:")
for term in terms:
    print(f"  {term}: {len(get_postings(index, term))} docs")

optimized = optimize_terms(index, terms, "and")
print("Optimized order (AND):", optimized)
print("Result:", search(index, docs, "cat and bird and dog"))

or_optimized = optimize_terms(index, terms, "or")
print("\nOptimized order (OR):", or_optimized)
