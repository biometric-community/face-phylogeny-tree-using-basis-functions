# DEVIATIONS — Face Phylogeny Tree

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Modalities | Face + iris + fingerprint | LFW face only | Local public payloads |
| D2 | Bases | Legendre, Chebyshev, Gabor, Gaussian/Bump RBF | Chebyshev/Legendre separable poly | Core IPT pipeline |
| D3 | Transforms | Photometric + geometric + DL | Photometric chain | Scope |
| D4 | Tree algorithm | DFS spanning as in prior IPT | Greedy best-parent edges | Simpler reconstruction |
