"""Assemble notebook_validation.ipynb from casmi.py plus the analysis cells below."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# CASMI26: a validation set that behaves like the hidden test

**TL;DR**
- The visible `test.parquet` is copied from **`enveda-180`**, a library of mostly drug-like synthetic compounds. The hidden test is natural-product-like. A score on the visible file says little about the leaderboard.
- `enveda-np-examples` is the only training library that is **both natural-product-like and measured on timsTOF**, the test instrument. Its 250 molecules make a natural validation set.
- Removing a molecule's spectra from the reference library **simulates the three novelty classes**, so every idea can be scored per class instead of in one blended number.
- A plain cosine library search scores **MRR 0.94 on class 1 and 0.00 on classes 2/3**, in about 3 minutes. Its public leaderboard score then shows roughly **how much of the hidden test is class 1**.

Everything here is CPU only, needs no external data, and produces a valid submission."""),
md("## Setup\nAll helper code is in the next cell: mass arithmetic, peak cleaning, a greedy cosine, the library index, MRR@25 and the novelty-class split."),
code((HERE / 'casmi.py').read_text().split('"""', 2)[2].strip()),
code("""import glob, time
import matplotlib.pyplot as plt

found = glob.glob('/kaggle/input/**/train.parquet', recursive=True)
DATA = Path(found[0]).parent if found else Path('../../enveda_data')
TRAIN, TEST = DATA / 'train.parquet', DATA / 'test.parquet'
meta = pd.read_parquet(TRAIN, columns=['inchikey14', 'ingest_lib', 'adduct', 'instrument_type',
                                       'molecular_formula', 'precursor_mz', 'base_peak_intensity'])
test = pd.read_parquet(TEST)
print(f'train: {len(meta):,} spectra, {meta.inchikey14.nunique():,} structures | test: {len(test):,} spectra, '
      f'{test.molecule_id.nunique()} molecules')""".replace('import glob, time', 'import glob, time\nfrom pathlib import Path')),
md("""## 1. Where do the training spectra come from?
Only two libraries were measured on the test instrument (timsTOF), and they are very different:"""),
code("""lib = meta.groupby('ingest_lib').agg(spectra=('inchikey14', 'size'), structures=('inchikey14', 'nunique'))
lib['timsTOF %'] = (meta.instrument_type.fillna('').str.contains('tims', case=False)
                    .groupby(meta.ingest_lib).mean() * 100).round(0)
public = set(meta.loc[~meta.ingest_lib.str.startswith('enveda'), 'inchikey14'])
lib['also in public libraries %'] = [round(100 * len(set(g) & public) / len(set(g)), 1) if name.startswith('enveda')
                                     else '' for name, g in meta.groupby('ingest_lib').inchikey14]
lib.sort_values('spectra', ascending=False)"""),
md("""- **`enveda-180`** holds 1.15M timsTOF spectra, but almost none of its structures (about 1%) appear in any public library. The formulas are full of F and Cl, which points to screening-library compounds, not natural products.
- **`enveda-np-examples`** holds 250 natural products on timsTOF, and **every one** has public spectra elsewhere in the training set.

## 2. The visible test file is from `enveda-180`
Joining the visible test spectra to the training set on precursor m/z, raw base-peak intensity and adduct matches every row:"""),
code("""j = test.merge(meta[['precursor_mz', 'base_peak_intensity', 'adduct', 'ingest_lib']],
               on=['precursor_mz', 'base_peak_intensity', 'adduct'], how='left')
j.ingest_lib.value_counts(dropna=False)"""),
md("""So any score on the visible file measures how well a method finds **synthetic** compounds **that are in the training set**. The hidden test is natural-product-like and mostly *not* in the training set. That is why "class 1 MRR = 1.000 on test" in some public notebooks doesn't carry over to the leaderboard.

The masses tell the same story. The overview gives the hidden test a median of 348 Da; natural products run heavier:"""),
code("""def masses(df):
    return pd.Series([neutral_mass(p, a) for p, a in zip(df.precursor_mz, df.adduct) if a in ADDUCT_SHIFT])

fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.linspace(100, 1200, 56)
for name, df in [('enveda-180 (= visible test)', meta[meta.ingest_lib == 'enveda-180'].sample(50_000, random_state=0)),
                 ('enveda-np-examples', meta[meta.ingest_lib == 'enveda-np-examples'])]:
    ax.hist(masses(df), bins=bins, density=True, histtype='step', lw=2, label=name)
ax.axvline(348, color='k', ls='--', lw=1, label='hidden test median (348 Da)')
ax.set_xlabel('neutral mass (Da)'); ax.set_yticks([]); ax.legend(frameon=False)
plt.tight_layout(); plt.show()"""),
md("""## 3. Simulating the novelty classes
The hidden test mixes three classes: **(1)** the structure has public spectra, **(2)** a known structure without public spectra, **(3)** a novel structure. With the 250 `enveda-np-examples` molecules as queries:

- **class 1:** half the molecules keep their *public* spectra in the reference library. Only their own timsTOF spectra are held out.
- **class 2/3:** for the other half, *every* spectrum of the structure is removed. Only a structure database (PubChem, COCONUT) or a de novo model could still find them.

The split is by structure (InChIKey14), the same identity the metric uses, so a held-out molecule can't leak back in under a different stereo form."""),
code("""q_rows, novelty, _ = make_validation(meta)
banned = set(novelty.index[novelty == 2])
print(f'{len(novelty)} validation molecules: {(novelty == 1).sum()} class 1, {(novelty == 2).sum()} class 2/3, '
      f'{len(q_rows)} query spectra')"""),
md("""## 4. Baseline: cosine library search
For each molecule:
1. Take the neutral mass as the median over its spectra (precursor m/z minus the adduct shift).
2. Use as candidates every library structure whose formula mass is within **10 ppm**.
3. Give each candidate, for each query spectrum, its best cosine against that structure's reference spectra in the same ion mode (0.02 Da tolerance, sqrt intensities, top 60 peaks). Average over the query spectra.

The reference library uses only the ten test adducts. Building it takes about 2–3 minutes."""),
code("""t = time.time()
val_lib = Library.build(TRAIN, keep=lambda df: (df.ingest_lib != 'enveda-np-examples')
                        & df.adduct.isin(TEST_ADDUCTS) & ~df.inchikey14.isin(banned))
print(f'library: {len(val_lib.meta):,} spectra, {len(val_lib.struct_keys):,} structures, {time.time() - t:.0f}s')

queries = pd.read_parquet(TRAIN, filters=[('ingest_lib', '=', 'enveda-np-examples')])
queries = queries[queries.adduct.isin(TEST_ADDUCTS)]
rows = []
for key, spectra in queries.groupby('inchikey14'):
    ranked, scores = rank_molecule(val_lib, spectra)
    rows.append(dict(key=key, novelty=novelty[key], rr=mrr_at_25(ranked, key), candidates=len(scores),
                     top_score=max(scores.values(), default=0)))
res = pd.DataFrame(rows)
res.groupby('novelty').agg(molecules=('rr', 'size'), MRR=('rr', 'mean'), top1=('rr', lambda r: (r == 1).mean()),
                           median_candidates=('candidates', 'median')).round(3)"""),
md("""Class 1 is close to solved by plain cosine search. Classes 2/3 score exactly zero, because the right answer isn't among the candidates at all.

The best cosine score also says which class a molecule is probably in, which is useful for gating other methods:"""),
code("""fig, ax = plt.subplots(figsize=(8, 3.2))
for c, label in [(1, 'class 1 (in library)'), (2, 'class 2/3 (not in library)')]:
    ax.hist(res.loc[res.novelty == c, 'top_score'], bins=np.linspace(0, 1, 26), alpha=.6, label=label)
ax.set_xlabel('best candidate cosine'); ax.set_ylabel('molecules'); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
for th in (0.5, 0.6, 0.7):
    hit = res.top_score >= th
    print(f'cosine >= {th}: flags {hit.mean():.0%} of molecules, {(res.novelty[hit] == 1).mean():.0%} of them truly class 1')"""),
md("""## 5. What this means for the leaderboard
If cosine search reaches MRR ≈ 0.94 on class 1 and 0 elsewhere, its leaderboard score is about `0.94 × (share of class 1)`. Submitting this notebook shows that share for the public test, and so how much of the score is left in classes 2/3."""),
code("""c1 = res.loc[res.novelty == 1, 'rr'].mean()
for lb in (0.15, 0.20, 0.25, 0.30):
    print(f'public LB {lb:.2f}  ->  about {lb / c1:.0%} of public test molecules are class 1')"""),
md("## 6. Submission\nThe same search against the *full* training library (all ten test adducts, nothing held out)."),
code("""del val_lib   # free memory before building the full library
t = time.time()
full_lib = Library.build(TRAIN, keep=lambda df: df.adduct.isin(TEST_ADDUCTS))
print(f'full library: {len(full_lib.meta):,} spectra, {time.time() - t:.0f}s')

sub = []
for mol, spectra in test.groupby('molecule_id', sort=False):
    spectra = spectra[spectra.adduct.isin(TEST_ADDUCTS)]   # the overview promises only these; guard anyway
    ranked = rank_molecule(full_lib, spectra)[0] if len(spectra) else []
    smiles = [full_lib.smiles[k] for k in ranked] or ['C']
    sub.append((mol, ';'.join(smiles)))
sub = pd.DataFrame(sub, columns=['molecule_id', 'smiles'])
sub.to_csv('submission.csv', index=False)
print(sub.shape, 'molecules with no mass-matched candidate:', (sub.smiles == 'C').sum())
sub.head()"""),
md("""## Takeaways
1. **Don't tune on the visible test file.** It is synthetic `enveda-180` data that is already in the training set.
2. **Score ideas per class.** A change that helps class 2/3 can look flat in a blended number while class 1 stays near 0.94.
3. **The headroom is in classes 2/3.** That means structure databases (COCONUT, PubChem) ranked by a spectrum → fingerprint model or in-silico fragmentation. I'll look at those next, using this validation set.

*If this helped, an upvote helps others find it. Questions and corrections are welcome in the comments.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
nbf.write(nb, HERE / 'notebook_validation.ipynb')
print('wrote', HERE / 'notebook_validation.ipynb', len(cells), 'cells')
