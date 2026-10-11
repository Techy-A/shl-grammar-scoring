"""CASMI 2026: spectral library search + a validation set that mimics the hidden test.

Everything a notebook needs lives here, so the Kaggle notebook can stay short:
    mass helpers, library building, cosine search, MRR@25, and the novelty-class split.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROTON = 1.007276
# m/z = M + shift  (all test adducts are singly charged)
ADDUCT_SHIFT = {
    '[M+H]+': PROTON,
    '[M+NH4]+': 18.033823,
    '[M-H2O+H]+': PROTON - 18.010565,
    '[M-2H2O+H]+': PROTON - 2 * 18.010565,
    '[M+Na]+': 22.989218,
    '[M+K]+': 38.963158,
    '[M-H]-': -PROTON,
    '[M-H2O-H]-': -PROTON - 18.010565,
    '[M+CH2O2-H]-': 44.998201,
    '[M+Cl]-': 34.969402,
}
MONO = {'C': 12.0, 'H': 1.0078250319, 'N': 14.0030740052, 'O': 15.9949146221, 'P': 30.97376151,
        'S': 31.97207069, 'F': 18.99840320, 'Cl': 34.96885271, 'Br': 78.9183376, 'I': 126.904468,
        'Si': 27.9769265327, 'B': 11.0093055, 'Se': 79.9165218, 'Na': 22.98976966, 'K': 38.9637069,
        'As': 74.9215964, 'Fe': 55.9349375, 'Co': 58.933195, 'Cu': 62.9295975, 'Zn': 63.9291422,
        'Mg': 23.9850417, 'Ca': 39.9625912, 'Al': 26.98153853, 'Sn': 119.9021947, 'Hg': 201.970643,
        'Pt': 194.9647911, 'Ge': 73.9211778, 'Li': 7.0160040, 'Mn': 54.9380496, 'Ni': 57.9353429,
        'Gd': 157.9241039, 'Ti': 47.9479463, 'Cr': 51.9405075, 'Ru': 101.9043493, 'Te': 129.9062244,
        'Sb': 120.9038157, 'Bi': 208.9803987, 'Ag': 106.905093, 'Au': 196.966552, 'Pd': 105.903486,
        'Rh': 102.905504, 'W': 183.950933, 'Mo': 97.9054082, 'V': 50.9439595, 'Tl': 204.974427,
        'Pb': 207.976652, 'Ba': 137.905247, 'Sr': 87.9056121, 'Cs': 132.905451, 'Rb': 84.911789,
        'D': 2.0141017778}
TEST_ADDUCTS = list(ADDUCT_SHIFT)


def formula_mass(formula):
    """Monoisotopic neutral mass of a formula like 'C9H6O3'; NaN if it has a charge or unknown element."""
    if not isinstance(formula, str) or not re.fullmatch(r'([A-Z][a-z]?\d*)+', formula):
        return np.nan
    total = 0.0
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if el not in MONO:
            return np.nan
        total += MONO[el] * (int(n) if n else 1)
    return total


def neutral_mass(precursor_mz, adduct):
    return precursor_mz - ADDUCT_SHIFT[adduct]


# ---------------------------------------------------------------- spectra

def clean_peaks(mzs, ints, precursor_mz, top_k=60):
    """Keep peaks below the precursor + 2 Da, the top_k most intense, sqrt-scaled and L2-normalised."""
    mz = np.asarray(mzs, dtype=np.float64)
    it = np.asarray(ints, dtype=np.float64)
    keep = (mz < precursor_mz + 2) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) > top_k:
        idx = np.argpartition(-it, top_k)[:top_k]
        mz, it = mz[idx], it[idx]
    order = np.argsort(mz)
    mz, it = mz[order], np.sqrt(it[order])
    norm = np.linalg.norm(it)
    return mz.astype(np.float32), (it / norm if norm > 0 else it).astype(np.float32)


def cosine(mz_a, it_a, mz_b, it_b, tol=0.02):
    """Greedy cosine: match peak pairs within tol, highest products first, each peak used once."""
    if len(mz_a) == 0 or len(mz_b) == 0:
        return 0.0
    lo = np.searchsorted(mz_b, mz_a - tol)
    hi = np.searchsorted(mz_b, mz_a + tol)
    pairs = [(it_a[i] * it_b[j], i, j) for i in range(len(mz_a)) for j in range(lo[i], hi[i])]
    if not pairs:
        return 0.0
    pairs.sort(reverse=True)
    used_a, used_b, score = set(), set(), 0.0
    for p, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j)
            score += p
    return float(score)


class Library:
    """Reference spectra stored flat (one peak array + offsets), indexed by neutral structure mass."""

    def __init__(self, meta, mz, it, offsets, struct_mass, struct_smiles):
        self.meta = meta                    # per-spectrum: inchikey14, ionization_mode, ingest_lib
        self.mz, self.it, self.offsets = mz, it, offsets
        self.struct_keys = struct_mass.index.to_numpy()
        self.struct_mass = struct_mass.to_numpy()
        order = np.argsort(self.struct_mass)
        self.struct_keys, self.struct_mass = self.struct_keys[order], self.struct_mass[order]
        self.smiles = struct_smiles
        self.spectra_of = meta.groupby('inchikey14').indices

    @classmethod
    def build(cls, path, keep=None, top_k=60, batch_size=50_000):
        """keep(df) -> boolean mask selecting the library rows (default: all rows)."""
        cols = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'ionization_mode', 'ingest_lib',
                'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
        metas, mzs, its, lens = [], [], [], []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()
            if keep is not None:
                df = df[keep(df)]
            for p_mz, a, b in zip(df.precursor_mz.to_numpy(), df.ms2_mzs, df.ms2_normalized_intensities):
                m, i = clean_peaks(a, b, p_mz, top_k)
                mzs.append(m), its.append(i), lens.append(len(m))
            metas.append(df[['inchikey14', 'normalized_smiles', 'molecular_formula', 'ionization_mode', 'ingest_lib']])
        meta = pd.concat(metas, ignore_index=True)
        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lens)
        structs = meta.drop_duplicates('inchikey14').set_index('inchikey14')
        struct_mass = structs.molecular_formula.map(formula_mass).dropna()
        return cls(meta[['inchikey14', 'ionization_mode', 'ingest_lib']].copy(),
                   np.concatenate(mzs), np.concatenate(its), offsets, struct_mass,
                   structs.normalized_smiles.to_dict())

    def peaks(self, k):
        a, b = self.offsets[k], self.offsets[k + 1]
        return self.mz[a:b], self.it[a:b]

    def structures_near(self, mass, ppm=10):
        tol = mass * ppm * 1e-6
        lo, hi = np.searchsorted(self.struct_mass, [mass - tol, mass + tol])
        return self.struct_keys[lo:hi]


# ---------------------------------------------------------------- search

def molecule_mass(spectra):
    """Median neutral mass over a molecule's spectra (robust to one odd precursor)."""
    return float(np.median([neutral_mass(p, a) for p, a in zip(spectra.precursor_mz, spectra.adduct)]))


def rank_molecule(lib, spectra, ppm=10, tol=0.02, top_k=60, n=25):
    """Rank library structures for one molecule.

    Candidates are structures whose formula mass is within ppm of the molecule's neutral mass.
    Each candidate scores the sum, over the query spectra, of its best cosine against any
    reference spectrum of that structure in the same ionisation mode.
    """
    cands = lib.structures_near(molecule_mass(spectra), ppm)
    if len(cands) == 0:
        return [], {}
    queries = [(clean_peaks(a, b, p, top_k), mode) for a, b, p, mode in
               zip(spectra.ms2_mzs, spectra.ms2_normalized_intensities, spectra.precursor_mz,
                   spectra.ionization_mode)]
    modes = lib.meta.ionization_mode.to_numpy()
    scores = {}
    for key in cands:
        total = 0.0
        refs = lib.spectra_of.get(key, [])
        for (q_mz, q_it), mode in queries:
            best = 0.0
            for k in refs:
                if modes[k] == mode:
                    r_mz, r_it = lib.peaks(k)
                    best = max(best, cosine(q_mz, q_it, r_mz, r_it, tol))
            total += best
        scores[key] = total / len(queries)
    ranked = sorted(scores, key=scores.get, reverse=True)[:n]
    return ranked, scores


# ---------------------------------------------------------------- metric + validation

def mrr_at_25(ranked_keys, true_key):
    for r, k in enumerate(ranked_keys[:25], 1):
        if k == true_key:
            return 1.0 / r
    return 0.0


def make_validation(meta, query_lib='enveda-np-examples', n_class1=125, seed=0):
    """Split the timsTOF natural-product molecules into simulated novelty classes.

    class 1 (in public libraries): the molecule's other public spectra stay in the reference library.
    class 2/3 (no public spectra): every spectrum of the structure is removed from the library,
        so only a structure database (PubChem / COCONUT) or a de novo model can still find it.
    Returns the query rows and a mask of training rows allowed in the reference library.
    """
    rng = np.random.default_rng(seed)
    q = meta[(meta.ingest_lib == query_lib) & meta.adduct.isin(TEST_ADDUCTS)]
    keys = np.array(sorted(q.inchikey14.unique()))
    rng.shuffle(keys)
    novelty = pd.Series(2, index=keys)
    novelty.iloc[:n_class1] = 1
    banned = set(novelty.index[novelty == 2])
    library_mask = (meta.ingest_lib != query_lib) & ~meta.inchikey14.isin(banned)
    return q.index.to_numpy(), novelty, library_mask.to_numpy()
