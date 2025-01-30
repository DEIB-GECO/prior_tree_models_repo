# Prior knowledge-guided tree-based models
Integration of prior domain downledge into tree-based models.


## Description 
We developed a robust framework that improves tree-based models for high-dimensional, noisy data by integrating feature selection, tree construction, and weighting with prior knowledge, combining data-driven insights and established domain understanding.

## Application Use Case
We compared the performance of the standard tree-based models and our proposed approaches on an application use case concerning the cancer-related subtype prediction of patients based on gene expression data. The use case concerns the classification of Breast Invasive Carcinoma (BRCA) patients patients in their corresponding cancer subtypes. We also performed two distinct sensitivity analyses to evaluate the impact of incorporating prir knowledge into tree-based models. We used a controlled dataset with limited correlation among the features for these analyses, considering publicly available RNA-seq profiles of Kidney Renal Clear Cell Carcinoma patients from The Cancer Genome Atlas (TCGA) project. The preprocessed dataset is available [here](), along with the list of features considered in the controlled dataset.

## Implementation
To implement such tree-based models, we developed PkTree, a Python package that implements the proposed modifications.
First build decicated conda environment:
```bash
conda create -n env_pktree python=3.9 
conda activate env_pktree
```
Install the PkTree package:
```bash
pip install pktree
```

Lastly, install the required packages from requirements.txt

## Prior doamin knowledge
In these experiments, we used the score of biological knowledge described [here](https://academic.oup.com/bioinformatics/article/40/10/btae605/7824055) and available [here](https://github.com/DEIB-GECO/GIS-weigthed_LASSO/tree/main). 

## Additional Information
All the code is available [here](https://github.com/DEIB-GECO/prior_tree_models_repo/tree/main/src). To replicate the experiments run the following scripts:
- `BRCA_dt.py` for experiments with the Decision Tree model
- `BRCA_rf_parallel.py` for experiments with the Random Forest model

The code to replicate the two sensitivity analyses is available [here](https://github.com/DEIB-GECO/prior_tree_models_repo/tree/main/src/sensitivity_analysis).
All the results from the experiments we performed can be found [here](https://github.com/DEIB-GECO/prior_tree_models_repo/tree/main/results).