# Risk-aware Temporal Cascade Reconstruction to Detect Asymptomatic Cases

__Conference Paper__: H. Jang, S. Pai, B. Adhikari, S.V. Pemmaraju, "Risk-aware Temporal Cascade Reconstruction to Detect Asymptomatic Cases," _IEEE International Conference on Data Mining (ICDM)_, 2021

__Journal Paper__: H. Jang, S. Pai, B. Adhikari, S.V. Pemmaraju, "Risk-aware temporal cascade reconstruction to detect asymptomatic cases." _Knowledge and Information Systems (KAIS)_, 2022

## Software

We use Python3. Install the following Python libraries:

```
gurobipy # for LP. Need a license to load our large graph
networkx # version >= 2.5
sklearn
pandas
numpy
matplotlib
tdqm
```

## Data

We provide synthetic data in the `data/` directory.
All the necessary information is saved as node attributes and edge attributes in the graph.
```
data/G_synthetic_step2_beta1_x1_v3.graphml
data/G_synthetic_step2_beta2_x1_v3.graphml
data/G_synthetic_step2_beta4_x1_v3.graphml
```

## Experiment on the Synthetic Data

Helper functions are defined in `utils/`

### Baselines

Get asymptomatic cases per baseline method. Then compute the performance.
```
python EXP1_get_ASYMP_for_baseline.py -beta 1
python EXP1_get_ASYMP_for_baseline.py -beta 2
python EXP1_get_ASYMP_for_baseline.py -beta 4
python EXP1_baseline.py
```

### Directed PCST

Run four methods to get asymptomatic cases.
Evaluation of the asymptomatic cases for MCA and GREEDY\_i are done in the scripts `EXP1_MCA_month1_v3.py` and `EXP1_d_steiner_month1_v3.py`, respectively.
For LP, we prepare another script `python EXP1_LP_get_asymp.py` to get asymptomatic cases.
```
python EXP1_MCA_month1_v3.py
python EXP1_d_steiner_month1_v3.py -i 1
python EXP1_d_steiner_month1_v3.py -i 2
python EXP1_LP_month1_v3.py
python EXP1_LP_get_asymp.py
```

### Generate tables and figures

Following script generates tables and figures in the paper for the experiment on the synthetic data.
```
python EXP1_network_statistics.py
python EXP1_tables_v3.py
python EXP1_barchart.py
python EXP1_linegraph.py
```

## Cite
Please cite our paper if you use the code or the dataset.
```
@inproceedings{jang2021directedPCST,
  title={Risk-aware Temporal Cascade Reconstruction to Detect Asymptomatic Cases},
  author={Jang, Hankyu and Pai, Shreyas and Adhikari, Bijaya and Pemmaraju, Sriram V.},
  booktitle={2021 IEEE International Conference on Data Mining (ICDM)},
  year={2021},
  organization={IEEE}
}
```
