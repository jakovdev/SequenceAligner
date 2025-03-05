# Steps to run:

## Windows
```sh
python -m venv .venv
.venv\Scripts\activate
.venv\Scripts\pip install -e .
.venv\Scripts\python main.py
```

## Linux
```sh
python -m venv .venv  
source .venv/bin/activate
.venv/bin/pip install -e . 
.venv/bin/python main.py  
```

## Results Summary
- [analysis_summary.json](results/analysis_summary.json)

## Results Visualizations
- [similarity_heatmap.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/similarity_heatmap.html)
- [similarity_distribution.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/similarity_distribution.html)
- [property_stats.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/property_stats.html)
- [feature_pca.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_pca.html)
- [feature_dist_percent_positive.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_percent_positive.html)
- [feature_dist_percent_polar.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_percent_polar.html)
- [feature_dist_percent_negative.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_percent_negative.html)
- [feature_dist_percent_hydrophobic.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_percent_hydrophobic.html)
- [feature_dist_percent_charged.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_percent_charged.html)
- [feature_dist_percent_aromatic.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_percent_aromatic.html)
- [feature_dist_molecular_weight.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_molecular_weight.html)
- [feature_dist_length.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_length.html)
- [feature_dist_hydrophobicity.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_hydrophobicity.html)
- [feature_dist_charge.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_dist_charge.html)
- [feature_correlations.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_correlations.html)
- [feature_importance.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/feature_importance.html)
- [aa_composition.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/aa_composition.html)
- [roc_curve.html](https://htmlpreview.github.io/?https://github.com/jakovdev/SequenceAligner/blob/dev-python/results/visualizations/roc_curve.html)

## Large Data
- [analysis_results.h5](https://myhdf5.hdfgroup.org/view?url=https%3A%2F%2Fgithub.com%2Fjakovdev%2FSequenceAligner%2Fblob%2Fdev-python%2Fresults%2Fanalysis_results.h5)