from contractguard.ingestion import ingest_pdf
from contractguard.chunking import chunk_document
from contractguard.retrieval.vector import InMemoryVectorIndex

parsed = ingest_pdf(
    "data/contracts/BELLICUMPHARMACEUTICALS,INC_05_07_2019-EX-10.1-Supply Agreement.PDF"
)
chunks = chunk_document(parsed, chunk_size=1000, overlap=150)
index = InMemoryVectorIndex.from_chunks(chunks)

for query in ["what is the liability cap", "what is the renewal term"]:
    print(f"\n=== {query} ===")
    for result in index.search(query, k=5):
        c = result.chunk
        print(f"{c.chunk_id}  score={result.score:.4f}  pages={c.page_start}-{c.page_end}")
        print(c.text[:150].replace("\n", " "))
        print()