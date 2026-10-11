"""Score the cosine library-search baseline on the simulated novelty classes.

Usage: python validate.py <train.parquet>
"""
import sys
import time

import numpy as np
import pandas as pd

from casmi import TEST_ADDUCTS, Library, make_validation, mrr_at_25, rank_molecule

path = sys.argv[1]
meta = pd.read_parquet(path, columns=['inchikey14', 'ingest_lib', 'adduct'])
q_rows, novelty, _ = make_validation(meta)
banned = set(novelty.index[novelty == 2])
print(f'validation: {len(novelty)} molecules ({(novelty == 1).sum()} class 1, {(novelty == 2).sum()} class 2/3), '
      f'{len(q_rows)} spectra')

t = time.time()
lib = Library.build(path, keep=lambda df: (df.ingest_lib != 'enveda-np-examples')
                    & df.adduct.isin(TEST_ADDUCTS) & ~df.inchikey14.isin(banned))
print(f'library: {len(lib.meta):,} spectra, {len(lib.struct_keys):,} structures, built in {time.time() - t:.0f}s')

queries = pd.read_parquet(path, filters=[('ingest_lib', '=', 'enveda-np-examples')])
queries = queries[queries.adduct.isin(TEST_ADDUCTS)]
rows = []
t = time.time()
for key, spectra in queries.groupby('inchikey14'):
    ranked, _ = rank_molecule(lib, spectra)
    rows.append(dict(key=key, novelty=novelty[key], rr=mrr_at_25(ranked, key), n_cands=len(ranked),
                     in_lib=key in lib.spectra_of))
res = pd.DataFrame(rows)
print(f'searched in {time.time() - t:.0f}s')
print(res.groupby('novelty').agg(molecules=('rr', 'size'), mrr=('rr', 'mean'), top1=('rr', lambda r: (r == 1).mean()),
                                 in_library=('in_lib', 'mean')).round(3))
for share in (0.3, 0.5, 0.7):
    c1 = res[res.novelty == 1].rr.mean()
    print(f'expected test MRR if {share:.0%} of molecules are class 1: {share * c1:.3f}')
res.to_csv('validation_results.csv', index=False)
