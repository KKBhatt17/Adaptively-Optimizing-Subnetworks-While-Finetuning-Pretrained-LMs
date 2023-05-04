# Adaptively-Optimizing-Subnetworks-While-Finetuning-Pretrained-LMs
This Repository Contains Codebase for Optimization in Machine Learning Project.

Dependencies:

python version: 3.8.10

transformers = 1.25.4


## instruction to run the code:

For Baseline:

1. cd sentiment
2. python data_preparation.py
3. python baseline_model.py
4. python evaluate.py


For DPS

1. cd DPS
2. cd script
3. bash run_glue.sh
4. cd sentiment
5. python evaluate.py


The finetuned model is saved in DPS/scripts/trained_models/output directory.