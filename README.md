# DeepLINK
Deep Large-Scale Inference UsingKnockoffs

## Simulation studies

Codes used in the simulation studies are in folder `Simulations/`. An example python script, `example.py`, for the simulation setting with linear factor model and linear link function is provided. Simulation results of other settings can be obtained by running this script with changes of setting parameters: `x_design`, `y_design`, etc. Choices of factor model design `x_design`: `linear`, `add_quad`, `logistic`. Choices of link function design `y_design`: `linear`, `nonlinear`. Note that the l1 regularization factor (`l1`) and the learning rate (`lr`) in the mlp training needs to be tuned for optimal performance.

## Real data analyses

Data matrices and codes used in the real data analyses are in folder `Real_data_analyses/`. Preprocessed data matrices are given as `csv` files. 
