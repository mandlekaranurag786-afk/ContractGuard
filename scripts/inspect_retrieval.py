from contractguard.ingestion import ingest_pdf
from contractguard.chunking import chunk_document
from contractguard.retrieval.vector import InMemoryVectorIndex

parsed = ingest_pdf(
    "/home/anurag-mandlekar/Desktop/ContractGuard/data/contracts/OPERALTD_04_30_2020-EX-4.14-SERVICE AGREEMENT.PDF"
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



# from contractguard.ingestion import load_parsed_document
# from contractguard.chunking.fixed import chunk_document

# doc = load_parsed_document("/home/anurag-mandlekar/Desktop/ContractGuard/data/parsed/2a5f3c4f00225f134976c2ca5a5a1923bb02a679a772a942792c45c77f978047.json")
# chunks = chunk_document(doc)

# for c in chunks:
#     if "Renewal Term" in c.text:
#         print(c.chunk_id, c.page_start, c.page_end)

# from contractguard.retrieval.vector import InMemoryVectorIndex

# index = InMemoryVectorIndex.from_chunks(chunks)
# results = index.search("what is the renewal term", k=20)

# for rank, r in enumerate(results, start=1):
#     print(rank, r.chunk.chunk_id, round(r.score, 4), r.chunk.page_start)


# for c in chunks:
#     if c.chunk_id.endswith(":20"):
#         print(c.text)