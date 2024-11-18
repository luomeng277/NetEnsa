# NetEnsa: Network Ensemble Analysis

NetEnsa is an algorithm designed to identify key species through a multistage analysis of co-occurrence network data ensemble, algorithmic ensemble, and decision ensemble.

## Environment Dependencies

### R (4.2.3)
- `igraph` 2.2.1
- `psych` 2.4.6.26
- `readxl` 2.1.5
- `dplyr` 1.1.4
- `NetMoss2` 0.2.0

### Python (3.7)
```python
pandas 1.3.5
scipy.stats 1.7.3
matplotlib.pyplot 3.5.3
numpy 1.21.6
networkx 2.6.3
xgboost 1.5.0
shap 0.41.0
```
## Directory Structure

```
├── ReadMe.md // Documentation for the project
├── data // Folder containing test data, including disease data, health data, and label files
├── code // Folder containing the scripts for running the analysis
###########Stage 1 Data Integration################  
│ ├── 1.MIC.py // Script for calculating MIC correlations
│ ├── 1.Pearson.R // Script for calculating Pearson correlations
│ ├── 1.Spearman.R // Script for calculating Spearman correlations
###########Stage 2 Algorithm Integration#############
│ ├── 2_Univariate_Weighting.py // Script for univariate weighting
│ └── 3_Matrix.py // Script for transforming correlation matrices into network data format
│ ├── 4_Centrality.py // Script for calculating network centrality node values
###########Stage 3 Decision Integration#############
│ ├── 5_Node_Select.R // Script for calculating the difference in centrality between disease and health nodes, and selecting differential microbes
│ ├── 6_Decision.py // Script for feature selection of differential microbes, selecting compact and disease-specific nodes
│ └── 7_Select.R // Script for selecting key microbes
│ ├── 8_ROC.py // Script for calculating the AUC values of nodes
│ └── 9_Control.py // Script for validating node control attributes
└── result // Folder for storing result files
```

## Description of Scripts

### Stage 1: Data Ensemble
- **1.MIC.py**: Calculates the Minimum Inhibitory Concentration (MIC) correlations.
- **1.Pearson.R**: Calculates the Pearson correlation coefficients.
- **1.Spearman.R**: Calculates the Spearman rank correlation coefficients.
- **1.Sparcc(R)**：
Building sparc network data with NetMoss2 -> netBuild function

```
library(rsparcc)
netBuild(case\_dir = case\_dir,
control\_dir = control\_dir,
method = "sparcc")

```

### Stage 2: Algorithm Ensemble
- **2_Univariate_Weighting.py**: Performs univariate weighting to prioritize features.
- **3_Matrix.py**: Converts correlation matrices into a format suitable for network analysis.
- **4_Centrality.py**: Computes centrality measures for nodes within the network.

### Stage 3: Decision Ensemble
- **5_Node_Select.R**: Calculates the centrality differences between disease and health states and identifies differential microbes.
- **6_Decision.py**: Selects features of differential microbes, focusing on compact and disease-specific nodes.
- **7_Select.R**: Selects key microbes based on the analysis.
- **8_ROC.py**: Computes the Area Under the Curve (AUC) values for the nodes.
- **9_Control.py**: Validates the control attributes of the nodes.

## Usage

To run the analysis, follow these steps:
1. Place your data in the `data` directory.
2. Execute the scripts in the `code` directory in the order of the stages.
3. Check the `result` directory for the output of the analysis.

## Contributing

If you have improvements or bug fixes, feel free to contribute by submitting a pull request.

## License

[Insert License Information Here]

---

Please adjust the sections as needed to fit the specifics of your project, such as adding the actual license information or any additional details about the usage or contributing guidelines.
