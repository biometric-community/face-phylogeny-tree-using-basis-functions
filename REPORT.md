# REPORT — Face Phylogeny Tree Using Basis Functions

**Protocol:** `full`

Lab twin: photometric near-duplicate chains on LFW; Chebyshev/Legendre
α fitting; Parzen likelihood-ratio scores; IPT root + edge recovery.

## Density fitting

- Subjects: 120
- Forward / backward pairs: 1440 / 1440
- Basis: chebyshev order=6
- Elapsed (s): 105.4

## IPT evaluation

- Sets: 60 (chain length 4)
- Rank-1 root: **33.33%**
- Rank-2 root: **55.00%**
- Rank-3 root: **78.33%**
- IPT edge accuracy: **33.33%**

![Ranks](outputs/figures/full/ipt_ranks.svg)

## Notes

- Paper also evaluates iris/fingerprint and geometric edits; this twin focuses on LFW photometric chains.
- See `DEVIATIONS.md`.
