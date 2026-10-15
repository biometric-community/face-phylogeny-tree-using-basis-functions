# IMPLEMENTATION_PLAN — Face Phylogeny IPT

| Component | Choice |
|-----------|--------|
| Data | LFW subjects with ≥5 images |
| Edits | brightness / contrast / gamma / gaussian / median chains |
| Basis | Chebyshev (default) / Legendre separable poly α |
| Densities | Parzen pf / pb on known forward/backward pairs |
| IPT | Likelihood-ratio score matrix → root out-degree + edges |
