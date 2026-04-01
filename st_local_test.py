import time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = [
    "Karnataka government scheme eligibility criteria for farmers.",
    "How to apply for a subsidy scheme and required documents.",
    "Annual family income limit and age criteria for beneficiaries.",
]
QUERY = "What is the income limit for eligibility?"


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def main() -> None:
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)  # replace with local folder path if needed
    t1 = time.time()

    emb_texts = model.encode(TEXTS, normalize_embeddings=True)
    emb_query = model.encode([QUERY], normalize_embeddings=True)[0]
    t2 = time.time()

    sims = [cos_sim(emb_query, emb_texts[i]) for i in range(len(TEXTS))]
    best_i = int(np.argmax(sims))

    print(f"Model: {MODEL_NAME}")
    print(f"Load time: {t1 - t0:.2f}s")
    print(f"Embed time (3 docs + 1 query): {t2 - t1:.2f}s")
    print(f"Embedding dim: {len(emb_query)}")
    print("Similarities:", [round(s, 4) for s in sims])
    print("\nTop match:")
    print(TEXTS[best_i])


if __name__ == "__main__":
    main()

