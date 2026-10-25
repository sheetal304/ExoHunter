\# ExoHunter: Exoplanet Candidate Classification with Machine Learning



ExoHunter trains and evaluates machine learning models on NASA's Kepler Objects of Interest (KOI) catalog to classify observations as CONFIRMED planets or FALSE POSITIVES, and serves the trained model in an interactive Streamlit dashboard.



This project classifies existing catalog entries. It does not discover new exoplanets.



\## Research question

How effectively can machine learning classify exoplanet candidates using astronomical measurements, and which factors contribute to incorrect predictions?



\## Data

\- Source: NASA Exoplanet Archive, Kepler Objects of Interest Cumulative Table (DOI: 10.26133/NEA4)

\- Downloaded on 25 October 2026 (the table is updated periodically, so newer downloads may differ)

\- The raw CSV is not included in this repository. Download it from the NASA Exoplanet Archive and place it in data/raw/

\- Attribution: this data is made available by the NASA Exoplanet Science Institute at IPAC, operated by Caltech under contract with NASA. Please check the archive's acknowledgment page for the current citation wording.



\## Method

\- Labels: CONFIRMED (1) vs FALSE POSITIVE (0). CANDIDATE entries are excluded from training and evaluation because their true status is unknown.

\- 11 input features: koi\_period, koi\_duration, koi\_depth, koi\_prad, koi\_teq, koi\_insol, koi\_model\_snr, koi\_impact, koi\_steff, koi\_slogg, koi\_srad.

\- Leakage prevention: disposition and vetting columns (koi\_score, koi\_pdisposition, koi\_fpflag\_\*, koi\_vet\_\*, koi\_disp\_prov, koi\_comment) and identifiers such as kepler\_name are excluded.

\- 80/20 stratified train/test split (random\_state=42). Missing values are filled with the training-set median inside a scikit-learn Pipeline, so no test information leaks into training.

\- Models: Random Forest and Gradient Boosting, with class balancing.



\## Results

Held-out test set (1,518 observations):



| Model | Accuracy | Precision | Recall | F1 |

|---|---|---|---|---|

| Random Forest | 0.930 | 0.892 | 0.918 | 0.905 |

| Gradient Boosting | 0.920 | 0.861 | 0.931 | 0.894 |



5-fold cross-validation on the training set (mean +/- std):



| Model | Accuracy | Precision | Recall | F1 |

|---|---|---|---|---|

| Random Forest | 0.916 +/- 0.007 | 0.863 +/- 0.015 | 0.912 +/- 0.005 | 0.887 +/- 0.009 |

| Gradient Boosting | 0.909 +/- 0.008 | 0.839 +/- 0.018 | 0.928 +/- 0.004 | 0.881 +/- 0.009 |



The two models perform similarly. Random Forest had higher precision, and Gradient Boosting had higher recall.



\## Findings

\- Class imbalance (4,839 false positives vs 2,748 planets): class weighting left accuracy and F1 almost unchanged (CV F1 0.887 vs 0.886) but raised recall (0.912 vs 0.891) at the cost of lower precision (0.863 vs 0.881).

\- Feature importance: planet radius (koi\_prad) ranked first under both Random Forest importance and permutation importance. The order of the other features differed between methods. Stellar surface gravity (koi\_slogg) had almost no measurable effect.

\- Error analysis (Random Forest, test set): 61 false alarms and 45 missed planets. Median signal-to-noise ratio and transit depth were lower for misclassified observations than for correct ones. This is an observation from a small group of errors, not proof of the cause.



\## Limitations

\- Classification of existing catalog entries, not discovery.

\- Labels come from NASA's vetting process.

\- The model uses 11 summary measurements, not light curves.

\- Planet scores are the share of trees voting "planet", not calibrated probabilities.

\- Results come from one dataset version, one train/test split and 5-fold cross-validation.

\- Several features are correlated, which affects individual importance scores.



\## Screenshots

!\[Overview](reports/figures/dashboard\_1\_overview.png)

!\[Prediction](reports/figures/dashboard\_2\_prediction.png)

!\[Performance](reports/figures/dashboard\_3\_performance.png)

!\[Chart](reports/figures/dashboard\_4\_chart.png)



\## Setup

&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt



Download the KOI cumulative table CSV from the NASA Exoplanet Archive into data/raw/, then run the scripts in order:



&#x20;   python src\\03\_prepare\_data.py

&#x20;   python src\\04\_split\_data.py

&#x20;   python src\\05\_train\_random\_forest.py

&#x20;   python src\\06\_train\_gradient\_boosting.py

&#x20;   python src\\07\_cross\_validation.py

&#x20;   python src\\08\_feature\_importance.py

&#x20;   python src\\09\_error\_analysis.py

&#x20;   python src\\10\_confusion\_matrix\_plot.py

&#x20;   python src\\11\_imbalance\_experiment.py

&#x20;   python src\\12\_permutation\_importance.py



Run the dashboard:



&#x20;   streamlit run app\\app.py



\## Future improvements

\- Add real Kepler light-curve data as model input.

\- Test on TESS data.

\- Calibrate predicted probabilities.

\- Repeat experiments with multiple random splits.



Author: Sheetal vishwakarma

