COMS4054A/7062A/7066A  NLP  Project 2 - Language in Culture and Responsible NLP
Sibonelo Mlambo (536187)

"Politics or Print? Separating content from shortcuts when classifying
 Imvo Zabantsundu and Isigidimi sama-Xosa (1884-1888)"

------------------------------------------------------------------------
1. What the code does
------------------------------------------------------------------------
Research question: when a classifier tells Imvo Zabantsundu apart from
Isigidimi sama-Xosa, is it reading political content, or newspaper
identity, transcription noise and editorial orthography?

src/data.py            loading, 200-token windows, cleaning ladder L0-L4,
                       issue-grouped splits, near-duplicate (reprint) filter
src/features.py        non-content features (transcription noise, orthography),
                       political / religious stem lexicons, synthetic OCR noise
src/models.py          classifiers (word/char TF-IDF + logistic regression,
                       Complement NB, fastText trained on the training split),
                       validation grid search, metrics, cluster bootstrap
src/run_experiments.py runs every experiment (E1-E8 below) -> outputs/
src/make_figures.py    figures + LaTeX tables for the report -> outputs/figures
src/check_grid_edge.py validation scores beyond the largest grid value of C
report/                LaTeX source of the extended abstract (main.tex)

Experiments
 E1  cleaning ladder L0..L4 x 4 models x 3 issue-grouped splits (seeds 13,21,42)
       L0 full page text (isiXhosa + English)
       L1 isiXhosa paragraphs only (dataset's XhTxt files)
       L2 L1 minus newspaper self-references, digits, punctuation
       L3 L2 with capitalised non-sentence-initial tokens masked (names)
       L4 L3 lower-cased, orthography equalised (bh->b, xh->x, elision
          apostrophes removed), restricted to words seen >=5 times in BOTH
          papers' training data
     + the same split WITHOUT the near-duplicate filter (leakage check)
 E2  temporal hold-out: train/validate on Nov 1884-1887, test on 1888
 E3  feature families: top-100 frequent words ("style"), remaining shared
     words ("content"), transcription-noise features, orthography features
 E4  evaluation on test pairs matched on all non-content features
 E5  synthetic OCR-noise injection into the cleaner paper's test segments
 E6  most influential features (CSV files)
 E7  out-of-sample probe: Isigidimi 1876-Oct 1884 (pre-Jabavu and Jabavu
     editorial eras) scored by the content classifier
 E8  political / religious lexicon rates per issue and per era

Hyper-parameters are chosen on the validation split by macro-F1; the test
split is used once. Test CIs: cluster bootstrap over issues (1000 resamples).

------------------------------------------------------------------------
2. Setup
------------------------------------------------------------------------
Python 3.10+ (tested with 3.13).   CPU only; no GPU or internet needed.

    pip install -r requirements.txt

Data: unzip the two course archives so that the page files sit in

    data/TxtFiles/IMVO_1884-11-03_p001.txt ...   (from 1._TxtFiles1870-1894_TB.zip)
    data/XhTxt/IMVO_1884-11-03_p001.txt ...      (XhTxt folder of TxtLangFiles1870-1894_TB.zip)

(The EnTxt folder is not needed.)

------------------------------------------------------------------------
3. Running
------------------------------------------------------------------------
One command (installs packages, unzips the course data if data/ is
missing, runs everything, builds figures):

    bash reproduction.sh            # full run, ~65 min on 2 CPU cores
    bash reproduction.sh --fast     # no fastText / grid check, ~35 min
    bash reproduction.sh --smoke    # 30% sample, code check, ~10 min

Put the two course zips next to the project folder, or pass --zips DIR.

Google Colab: open reproduce_colab.ipynb, upload the two course zips to
/content, and run the cells in order. The notebook clones this repository,
moves into Project2_isiXhosa/ and calls reproduction.sh.

The individual steps the script runs are:

Full reproduction (about 1.5-2 h on 2 CPU cores):

    python src/run_experiments.py --data data --out outputs --seeds 13
    python src/run_experiments.py --data data --out outputs --seeds 21 42 --e1-only
    python src/check_grid_edge.py --data data --out outputs   (optional, ~10 min)
    python src/make_figures.py --out outputs

(The two run_experiments commands are independent and can run in
parallel; the first one also runs E2-E8 on seed 13.  Running
--seeds 13 21 42 in a single command is equivalent but slower.)

Quicker options:
    --fast    skip the fastText model (roughly halves the run time)
    --smoke   30% subsample of issues, only to check that the code runs

------------------------------------------------------------------------
4. Outputs (outputs/)
------------------------------------------------------------------------
results.csv, results_e1_21_42.csv  every model run: split, level, model,
                                   chosen hyper-parameters, val/test scores, CIs
results_seed_mean_sd.csv           mean/sd over the three splits
matched_eval.csv                   E4
noise_injection.csv                E5
top_*_features.csv                 E6
probe_by_issue.csv, probe_segments.csv   E7
lexicon_by_issue.csv               E8
surface_features.csv               non-content features per segment
stats.json                         corpus counts, duplicate counts, matching
                                   balance, probe and lexicon summaries
figures/                           figures used in the report
log_*.txt                          run logs
