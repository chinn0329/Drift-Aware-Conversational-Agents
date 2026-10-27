from sentence_transformers import SentenceTransformer
from src.signals.contradiction import ContradictionTracker

emb = SentenceTransformer("all-MiniLM-L6-v2")
ct = ContradictionTracker()

old = ["The age is stored in the first two characters of each string.",
       "Passengers aged 60 or older count as seniors."]

cases = {
    "identical":       old,
    "paraphrase":      ["Age is found in the first two characters of every string.",
                        "Seniors are passengers who are at least 60."],
    "real revision":   ["The age is stored in characters 11 and 12.",
                        "Passengers aged 60 or older count as seniors."],
    "new unrelated":   ["The list has up to 100 entries."],
    "full reversal":   ["The age is NOT in the first two characters; it is in characters 12-13.",
                        "Only passengers strictly older than 60 count."],
}
for name, new in cases.items():
    print(f"{name:14s} -> {ct.check_contradiction(old, new, embedder=emb):.3f}")