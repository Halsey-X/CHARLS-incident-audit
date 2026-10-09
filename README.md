# CHARLS Incident Multimorbidity Prediction: A Measurement-Stability Audit

Code and data analysis for the paper:

> **A gradient-boosting model for incident chronic conditions in CHARLS adds no decision value over a zero-cost rule: a measurement-stability audit**
>
> Jie Huang†, Yuan Chen†, Jiaqi Li, Hao Xu*
> School of Information Engineering, Hunan Applied Technology University
>
> † Equal contribution, * Corresponding author: imxiho@163.com

## Key finding

A LightGBM model trained on 29 baseline predictors for incident chronic multimorbidity in CHARLS reached an AUC of 0.65 internally and 0.64 in temporal validation — but did **not** outperform a zero-cost rule that simply reverses the baseline condition count (AUC 0.68 internal, 0.63 validation). A measurement-stability audit showed why: self-reported diagnoses are unstable between waves, so apparent "incidence" is largely artefact rather than true disease onset.

## Repository structure

```
code/          Python analysis scripts (data processing, model training, figure generation)
figures/       Main-text figures (PNG)
```

## Data

The analysis uses publicly available CHARLS data, downloadable from:
https://charls.pku.edu.cn/

## Requirements

- Python 3.12
- lightgbm, scikit-learn, pandas, numpy, matplotlib, shap

## Citation

If you use this code or findings, please cite our paper (submitted to PeerJ).

## License

MIT
