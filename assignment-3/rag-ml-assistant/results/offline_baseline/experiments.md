## Retrieval experiments

Embedding: `tfidf` / `tfidf` · store: `sparse` · chunks: 367 (size 500, overlap 100)

### Effect of k

| k | hit rate | MRR | precision@k | keyword coverage | avg context tokens |
|---|---|---|---|---|---|
| 3 | 0.875 | 0.844 | 0.667 | 0.958 | 1348 |
| 5 | 0.938 | 0.859 | 0.613 | 0.979 | 2204 |
| 7 | 0.938 | 0.859 | 0.607 | 0.979 | 3136 |

### Effect of chunk size / overlap (k=5)

| chunk size | overlap | #chunks | mean tokens | hit rate | MRR | precision@5 | keyword coverage | avg context tokens |
|---|---|---|---|---|---|---|---|---|
| 300 | 50 | 612 | 257.8 | 0.875 | 0.833 | 0.637 | 0.938 | 1296 |
| 500 | 100 | 367 | 450.8 | 0.938 | 0.859 | 0.613 | 0.979 | 2204 |
| 800 | 150 | 219 | 754.9 | 0.938 | 0.906 | 0.588 | 0.979 | 3803 |
| 1000 | 200 | 178 | 941.5 | 0.938 | 0.875 | 0.562 | 1.0 | 4674 |

### Similarity-score profile (for choosing the refusal threshold)

In-scope top-1 score: {'min': 0.078, 'median': 0.183, 'max': 0.352}

| out-of-scope question | top-1 score |
|---|---|
| Who won the 2018 FIFA World Cup? | 0.09 |
| How do transformers use self-attention in large language models? | 0.127 |
| Give me a recipe for chocolate cake. | 0.0 |

Suggested `min_score`: **None**
