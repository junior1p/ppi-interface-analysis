# PPI Interface Analysis Pipeline

Protein-protein interface hotspot prediction using computational alanine scanning and SASA-based BSA analysis.

## Method

**Mode A — Interface Analysis (CPU, fast)**
Given a PDB file, computes:
- Interface residues via BSA (buried surface area) differential
- Contact map (Cα-Cα < 8Å, heavy atom < 5Å)
- Computational alanine scanning with hotspot scoring
- Shape complementarity proxy
- Polar/hydrophobic/charged composition

**Demo:** PD-1/PD-L1 immune checkpoint complex (PDB: 4ZQK) — a key cancer immunotherapy target.

## Results Summary

| Metric | Value |
|--------|-------|
| Total BSA | 1823.7 Å² (typical Ab-Ag: 1200–2000 Å²) |
| Shape Complementarity | 0.666 (good, >0.65) |
| Chain A (PD-L1) hotspots | 12 residues |
| Chain B (PD-1) hotspots | 10 residues |

**Top hotspots:** B134ILE, A56TYR, A125ARG, B128LEU, A113ARG

## Usage

```bash
pip install biopython numpy pandas matplotlib seaborn scipy requests

python ppi_pipeline.py
```

## Output Files

- `interface_residues.csv` — all interface residues with BSA and hotspot scores
- `hotspots.json` — hotspot list for downstream binder design
- `interface_bsa.png` — BSA bar chart
- `contact_map.png` — inter-chain contact heatmap
- `composition_radar.png` — polar/apolar composition radar

## References

- Bogan & Thorn (1998). Anatomy of hot spots in protein interfaces. *JMB*
- Mirdita et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*
