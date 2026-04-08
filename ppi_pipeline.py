#!/usr/bin/env python3
"""
PPI Interface Analysis Pipeline
Based on the Protein-Protein Interface Analysis & Hotspot Prediction Skill
Demo: PD-1/PD-L1 immune checkpoint complex (PDB: 4ZQK)
"""

import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
import copy, warnings
warnings.filterwarnings("ignore")

os.makedirs("/root/ppi_interface/results", exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_pdb(pdb_id: str, out_dir: str = "data/") -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    out_path = f"{out_dir}/{pdb_id}.pdb"
    if os.path.exists(out_path):
        print(f"PDB {pdb_id} already downloaded.")
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        out_path = f"{out_dir}/{pdb_id}.cif"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    with open(out_path, "w") as f:
        f.write(response.text)
    print(f"Downloaded PDB {pdb_id} to {out_path}")
    return out_path

def load_structure(pdb_path: str):
    return PDBParser(QUIET=True).get_structure("complex", pdb_path)

def res_key(res) -> str:
    """Stable string key for a residue: chainID:resnum:resname"""
    return f"{res.get_parent().id}:{res.id[1]}:{res.resname}"

# ── SASA of an isolated chain (built from scratch) ────────────────────────────
def sasa_isolated(chain, n_points: int = 250) -> dict:
    """Return {res_key -> sasa_value} for each residue in the chain."""
    iso = Structure("iso")
    iso.add(Model(0))
    iso[0].add(Chain(chain.id))
    iso_chain = iso[0][chain.id]
    for orig_res in chain.get_residues():
        if orig_res.id[0] != ' ':
            continue
        new_res = Residue(orig_res.id, orig_res.resname, orig_res.segid)
        for atom in orig_res.get_atoms():
            new_atom = Atom(
                atom.name, copy.copy(atom.coord),
                atom.bfactor, atom.occupancy, atom.altloc,
                atom.fullname, None,
            )
            new_res.add(new_atom)
        iso_chain.add(new_res)
    sr = ShrakeRupley(n_points=n_points)
    sr.compute(iso, level="R")
    return {res_key(res): res.sasa for res in iso.get_residues() if res.id[0] == ' ' and res.sasa}

# ── Interface Identification ────────────────────────────────────────────────────
def identify_interface(structure, chain_a_id: str, chain_b_id: str,
                      contact_cutoff: float = 8.0,
                      heavy_atom_cutoff: float = 5.0,
                      bsa_cutoff: float = 1.0) -> dict:
    model = structure[0]
    chain_a = model[chain_a_id]
    chain_b = model[chain_b_id]

    # Distance-based contacts
    contact_pairs = []
    for res_a in chain_a.get_residues():
        if res_a.id[0] != ' ': continue
        for res_b in chain_b.get_residues():
            if res_b.id[0] != ' ': continue
            try:
                if (res_a["CA"] - res_b["CA"]) < contact_cutoff:
                    min_dist = min(
                        (a1 - a2)
                        for a1 in res_a.get_atoms()
                        for a2 in res_b.get_atoms()
                        if a1.element != 'H' and a2.element != 'H'
                    )
                    if min_dist < heavy_atom_cutoff:
                        contact_pairs.append((res_a, res_b, round(min_dist, 2)))
            except KeyError:
                pass

    # SASA of complex
    sr = ShrakeRupley(n_points=250)
    sr.compute(structure, level="R")
    sasa_complex = {res_key(res): res.sasa
                   for res in structure.get_residues()
                   if res.id[0] == ' ' and res.sasa}

    # SASA of each chain alone
    sasa_a = sasa_isolated(chain_a)
    sasa_b = sasa_isolated(chain_b)

    # BSA per residue
    bsa_a, bsa_b = {}, {}
    for res in chain_a.get_residues():
        if res.id[0] != ' ': continue
        key = res_key(res)
        iso = sasa_a.get(key)
        bound = sasa_complex.get(key)
        if iso is not None and bound is not None:
            bsa = iso - bound
            if bsa > bsa_cutoff:
                bsa_a[res] = bsa

    for res in chain_b.get_residues():
        if res.id[0] != ' ': continue
        key = res_key(res)
        iso = sasa_b.get(key)
        bound = sasa_complex.get(key)
        if iso is not None and bound is not None:
            bsa = iso - bound
            if bsa > bsa_cutoff:
                bsa_b[res] = bsa

    total_bsa = sum(bsa_a.values()) + sum(bsa_b.values())
    print(f"\n--- Interface Summary ---")
    print(f"  Chain {chain_a_id} interface residues: {len(bsa_a)}")
    print(f"  Chain {chain_b_id} interface residues: {len(bsa_b)}")
    print(f"  Total BSA: {total_bsa:.1f} Å²")
    print(f"  Contact pairs (heavy atom < {heavy_atom_cutoff}Å): {len(contact_pairs)}")
    print(f"  Typical Ab-Ag interface BSA: 1200–2000 Å²")

    return {
        "bsa_a": bsa_a, "bsa_b": bsa_b,
        "total_bsa": total_bsa,
        "contact_pairs": contact_pairs,
        "sasa_a": sasa_a, "sasa_b": sasa_b,
    }

# ── Alanine Scanning ────────────────────────────────────────────────────────────
HOTSPOT_WEIGHTS = {
    "TRP": 3.0, "TYR": 2.5, "ARG": 2.0, "PHE": 2.0,
    "LEU": 1.5, "ILE": 1.5, "MET": 1.5, "LYS": 1.3,
    "VAL": 1.2, "HIS": 1.8, "ASP": 1.2, "GLU": 1.2,
    "ASN": 1.0, "GLN": 1.0, "PRO": 0.8, "CYS": 1.8,
    "THR": 0.7, "SER": 0.6, "ALA": 0.5, "GLY": 0.3,
}
HOTSPOT_THRESHOLD = 25.0

def alanine_scan(interface_data: dict, chain_id: str) -> list:
    bsa_dict = interface_data[f"bsa_{chain_id.lower()}"]
    results = []
    for res, bsa in bsa_dict.items():
        rn = res.get_resname()
        score = bsa * HOTSPOT_WEIGHTS.get(rn, 1.0)
        results.append({
            "res": res,
            "label": f"{chain_id}{res.id[1]}{rn}",
            "resname": rn, "resnum": res.id[1], "chain": chain_id,
            "BSA_A2": round(bsa, 2),
            "hotspot_score": round(score, 2),
            "is_hotspot": bsa >= HOTSPOT_THRESHOLD,
        })
    results.sort(key=lambda x: x["hotspot_score"], reverse=True)
    n_hs = sum(1 for r in results if r["is_hotspot"])
    print(f"\n--- Alanine Scanning: Chain {chain_id} ---")
    print(f"  Interface residues: {len(results)}, Hotspots (BSA≥{HOTSPOT_THRESHOLD}Å²): {n_hs}")
    print(f"  {'Residue':<12} {'BSA (Å²)':<12} {'Score':<10} Hotspot?")
    print(f"  {'-'*45}")
    for r in results[:10]:
        flag = "🔥" if r["is_hotspot"] else ""
        print(f"  {r['label']:<12} {r['BSA_A2']:<12.1f} {r['hotspot_score']:<10.2f} {flag}")
    return results

# ── Interface Composition ─────────────────────────────────────────────────────
def analyze_composition(interface_data, chain_a_id, chain_b_id, ha, hb) -> dict:
    POLAR = {"SER","THR","ASN","GLN","TYR","TRP","CYS"}
    AP    = {"ALA","VAL","LEU","ILE","MET","PHE","PRO","GLY"}
    CP    = {"ARG","LYS","HIS"}
    CN    = {"ASP","GLU"}

    def classify(lst):
        c = {"polar":0,"apolar":0,"charged+":0,"charged-":0}
        for r in lst:
            n = r["resname"] if isinstance(r,dict) else r.get_resname()
            if n in POLAR: c["polar"]+=1
            elif n in AP:  c["apolar"]+=1
            elif n in CP:  c["charged+"]+=1
            elif n in CN:  c["charged-"]+=1
        return c

    ca, cb = classify(ha), classify(hb)
    bsa = interface_data["total_bsa"]
    denom = sum(interface_data["sasa_a"].get(res_key(r["res"]), 0) for r in ha) + \
            sum(interface_data["sasa_b"].get(res_key(r["res"]), 0) for r in hb)
    sc = (2 * bsa / denom) if denom > 0 else 0
    return {
        "chain_a": ca, "chain_b": cb,
        "total_bsa_A2": round(bsa, 1),
        "shape_comp": round(sc, 3),
        "n_ha": sum(1 for r in ha if r["is_hotspot"]),
        "n_hb": sum(1 for r in hb if r["is_hotspot"]),
        "n_contacts": len(interface_data["contact_pairs"]),
    }

# ── Visualizations ────────────────────────────────────────────────────────────
def plot_bsa(hotspots_a, hotspots_b, cid_a, cid_b, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, hs, cid in [(axes[0], hotspots_a, cid_a), (axes[1], hotspots_b, cid_b)]:
        if not hs:
            ax.set_title(f"Chain {cid}: no interface"); continue
        lbs = [r["label"] for r in hs]
        bsas = [r["BSA_A2"] for r in hs]
        colors = ["#e74c3c" if r["is_hotspot"] else "#95a5a6" for r in hs]
        ax.bar(range(len(lbs)), bsas, color=colors, edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(lbs))); ax.set_xticklabels(lbs, rotation=60, ha="right", fontsize=7)
        ax.axhline(25, color="#e74c3c", ls="--", alpha=0.6, label="Hotspot threshold")
        ax.set_ylabel("BSA (Å²)"); ax.set_title(f"Chain {cid} Interface")
        ax.legend(handles=[mpatches.Patch(color="#e74c3c",label="Hotspot"), mpatches.Patch(color="#95a5a6",label="Non-hotspot")])
    plt.suptitle("Computational Alanine Scanning — BSA per Interface Residue", y=1.02)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"BSA bar chart: {out}")

def plot_contact_map(interface_data, cid_a, cid_b, out):
    pairs = interface_data["contact_pairs"]
    if not pairs: print("No contacts."); return
    ra = sorted(set(r[0].id[1] for r in pairs))
    rb = sorted(set(r[1].id[1] for r in pairs))
    mat = np.full((len(ra), len(rb)), np.nan)
    ai = {r: i for i, r in enumerate(ra)}; bi = {r: i for i, r in enumerate(rb)}
    for a, b, d in pairs:
        i, j = ai[a.id[1]], bi[b.id[1]]
        if np.isnan(mat[i,j]) or d < mat[i,j]: mat[i,j] = d
    fig, ax = plt.subplots(figsize=(max(8,len(rb)*0.3), max(6,len(ra)*0.3)))
    im = ax.imshow(np.where(np.isnan(mat), 5.0, mat), cmap="Blues_r", vmin=2.0, vmax=5.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Min heavy-atom dist (Å)")
    ax.set_xticks(range(len(rb))); ax.set_xticklabels([f"{cid_b}{r}" for r in rb], rotation=90, fontsize=6)
    ax.set_yticks(range(len(ra))); ax.set_yticklabels([f"{cid_a}{r}" for r in ra], fontsize=6)
    ax.set_xlabel(f"Chain {cid_b}"); ax.set_ylabel(f"Chain {cid_a}")
    ax.set_title("Interface Contact Map")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Contact map: {out}")

def plot_radar(comp, cid_a, cid_b, out):
    cats = ["polar","apolar","charged+","charged-"]
    def norm(c): t=sum(c.values()) or 1; return [c.get(x,0)/t for x in cats]
    va = norm(comp["chain_a"]); vb = norm(comp["chain_b"])
    ang = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    va += va[:1]; vb += vb[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(ang, va, "o-", c="#3498db", label=f"Chain {cid_a}", lw=2); ax.fill(ang, va, c="#3498db", alpha=0.25)
    ax.plot(ang, vb, "s-", c="#e74c3c", label=f"Chain {cid_b}", lw=2); ax.fill(ang, vb, c="#e74c3c", alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats); ax.legend(); ax.set_title("Interface Composition", pad=20)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Radar chart: {out}")

# ── Save Results ───────────────────────────────────────────────────────────────
def save(hs_a, hs_b, comp, cid_a, cid_b, pdb_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"chain":r["chain"],"resnum":r["resnum"],"resname":r["resname"],
             "label":r["label"],"BSA_A2":r["BSA_A2"],
             "hotspot_score":r["hotspot_score"],"is_hotspot":r["is_hotspot"]}
            for r in hs_a + hs_b]
    pd.DataFrame(rows).sort_values("hotspot_score",ascending=False).to_csv(f"{out_dir}/interface_residues.csv", index=False)
    with open(f"{out_dir}/hotspots.json","w") as f:
        json.dump({cid_a:[r["label"] for r in hs_a if r["is_hotspot"]],
                   cid_b:[r["label"] for r in hs_b if r["is_hotspot"]]}, f, indent=2)
    all_hs = sorted([r for r in hs_a+hs_b if r["is_hotspot"]], key=lambda x:x["hotspot_score"], reverse=True)
    report = ["="*65, "PROTEIN-PROTEIN INTERFACE ANALYSIS REPORT", "="*65,
              f"Input: {pdb_path}", f"Chains: {cid_a} / {cid_b}", "",
              "--- Interface Metrics ---",
              f"  Total BSA:              {comp['total_bsa_A2']} Å²",
              f"  Contact pairs:          {comp['n_contacts']}",
              f"  Shape complementarity:   {comp['shape_comp']}  [>0.65=good, <0.50=poor]",
              f"  Hotspots (chain {cid_a}):  {comp['n_ha']}",
              f"  Hotspots (chain {cid_b}):  {comp['n_hb']}",
              "", "--- Top Hotspot Residues ---"]
    for r in all_hs[:15]:
        report.append(f"  {r['label']:<14} BSA={r['BSA_A2']:.1f} Å²  score={r['hotspot_score']:.1f}")
    report += ["","--- Output Files ---",
               f"  {out_dir}/interface_residues.csv", f"  {out_dir}/hotspots.json",
               f"  {out_dir}/interface_bsa.png", f"  {out_dir}/contact_map.png",
               f"  {out_dir}/composition_radar.png", "="*65]
    print("\n" + "\n".join(report))
    with open(f"{out_dir}/interface_report.txt","w") as f: f.write("\n".join(report))
    print(f"\n✅ All outputs: {out_dir}/")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("PPI INTERFACE ANALYSIS — PD-1/PD-L1 (PDB: 4ZQK)")
    print("="*65)

    pdb_path = fetch_pdb("4ZQK", "/root/ppi_interface/data")
    structure = load_structure(pdb_path)

    ifd = identify_interface(structure, "A", "B")
    hs_a = alanine_scan(ifd, "A")
    hs_b = alanine_scan(ifd, "B")
    comp = analyze_composition(ifd, "A", "B", hs_a, hs_b)

    out = "/root/ppi_interface/results"
    plot_bsa(hs_a, hs_b, "A", "B", f"{out}/interface_bsa.png")
    plot_contact_map(ifd, "A", "B", f"{out}/contact_map.png")
    plot_radar(comp, "A", "B", f"{out}/composition_radar.png")
    save(hs_a, hs_b, comp, "A", "B", pdb_path, out)

if __name__ == "__main__":
    main()
