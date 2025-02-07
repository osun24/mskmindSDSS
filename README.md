# SDSS 2025: Ensemble Learning for Survival Analysis of Clinical and Genomic Biomarkers in Advanced Non-Small Cell Lung Cancer
Conference: https://ww3.aievolution.com/AMSTATevents/index.cfm?do=ev.viewEv&ev=4337

## Abstract: 
Lung cancer is the leading cause of cancer-related deaths in the U.S., with non-small cell lung cancer (NSCLC) comprising approximately 85% of cases. Survival analysis for NSCLC is essential for identifying clinical and genomic biomarkers influencing progression-free survival (PFS), time until progression or death due to NSCLC. Such biomarkers enable personalized treatment and prognosis prediction for NSCLC, improving patient outcomes and advancing precision oncology. In this study, we analyze a cohort of 216 U.S. patients with advanced NSCLC using two ensemble learning survival methods, random survival forests (RSF) and a gradient-boosted machine (GBM), and a stratified Cox proportional hazards models. All models accounted for censoring. RSF employs multiple decision trees to estimate hazards, with overall hazard predictions derived by averaging outputs from all trees. GBM uses regression trees as base learners, optimized with the Cox proportional hazards model's log-likelihood function. The models' PFS prediction performance was evaluated using the concordance index (C-index). All models demonstrated better-than-random prediction. GBM (C-index: 0.733) had the highest predictive capability followed by RSF (C-index: 0.732) and the stratified Cox proportional hazards model (C-index: 0.726). Key biomarkers were identified using permutation- and impurity-based feature importance and the effects of these biomarkers on PFS were characterized with hazard ratios. The models identified several significant biomarkers, including circulating albumin, derived neutrophil-to-lymphocyte ratio (dNLR), PD-L1 expression, and tumor mutational burden (TMB). Albumin and dNLR, markers of systemic inflammation, were linked to survival outcomes, reflecting the role of inflammation in cancer progression. PD-L1 and TMB, key immunotherapy biomarkers, showed modest protective effects, consistent with immunotherapy benefits for certain NSCLC patients.
