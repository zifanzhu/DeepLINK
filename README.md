# DeepLINK: Deep Learning INference using Knockoffs

DeepLINK is a variable selection algorithm that guarantees the false discovery rate (FDR) control in high-dimensional settings. It consists of two major parts: an autoencoder network for the knockoff variable construction and a multilayer perceptron network for feature selection with the FDR control. More details of DeepLINK are provided in the following paper:

[Zhu, Z., Fan, Y., Kong, Y., Lv, J., & Sun, F. (2021). DeepLINK: Deep learning inference using knockoffs with applications to genomics. *Proceedings of the National Academy of Sciences*, 118(36).](https://www.pnas.org/content/118/36/e2104683118.short)

## Dependencies

DeepLINK requires Python 3 (>= 3.7.6) with the following packages:

- numpy >= 1.18.5
- keras >= 2.4.3
- tensorflow >= 2.3.0

## Installation

Clone the github repository and enter DeepLINK directory with
    
    $ git clone https://github.com/zifanzhu/DeepLINK.git
    $ cd DeepLINK
    
## Usage

The inputs of DeepLINK are sample-by-feature data matrix and the corresponding response vector. Assume the dimension of the data matrix is n-by-p. Then the response should an n-by-1 vector. Also, they should be in '.npy' format and without row and column names. To convert your data into '.npy' format, you can simply load it to python by `numpy.loadtxt`, trim the row and column names, and then save it with `numpy.save`. You need to have 'numpy' installed to do so. Another required input is the output directory. The outputs of DeepLINK are two space delimited files with selected feature indices using knockoff and knockoff+ threshold respectively, with names `selected_variable_ko` and `selected_variable_ko+`.  Notice that feature index is 0-based, so the first feature (first column of the data matrix) has index 0.

### Options

    -h, --help            show this help message and exit
    -X DATA, --data=DATA  data matrix in '.npy' format (row sample, column
                          feature)
    -y RESPONSE, --response=RESPONSE
                          response vector in '.npy' format
    -o OUTPUT_DIR, --out=OUTPUT_DIR
                          output directory
    -s                    use this flag if you do not want to center and scale
                          data (center/scale makes every feature column have
                          mean 0/sd 1)
    -l L1, --l1=L1        l1 regularization coefficient used in the feature
                          selection MLP [default: 0.001]
    -r LR, --lr=LR        learning rate used in the feature selection MLP
                          [default: 0.001]
    -a ACT, --act=ACT     activation function used in the feature selection MLP
                          [default: elu]
    -L LOSS, --loss=LOSS  loss function used in the feature selection MLP
                          [default: mean_squared_error]
    -q Q, --fdr_level=Q   fdr level [default: 0.2]
    
### A toy example

Run the following code and the outputs are in folder `test/out`:
    
    $ python deeplink.py -X test/X.npy -y test/y.npy -o test/out

## Replicate studies in the paper

### Simulation studies

Codes used in the simulation studies are in folder `Simulations/`. An example python script, `example.py`, for the simulation setting with linear factor model and linear link function is provided. Simulation results of other settings can be obtained by running this script with changes of setting parameters: `x_design`, `y_design`, etc. Choices of factor model design `x_design`: `linear`, `add_quad`, `logistic`. Choices of link function design `y_design`: `linear`, `nonlinear`. Note that the l1 regularization factor (`l1`) and the learning rate (`lr`) in the mlp training needs to be tuned for optimal performance.

### Real data analyses

Data matrices and codes used in the real data analyses are in folder `Real_data_analyses/`. Preprocessed data matrices are given as `csv` files under the corresponding `data/` folder. For the microbiome data analysis, we used an independent dataset `yu_CRC_common.csv` for screening. For the other two scRNA-seq datasets, the screening was done using 50% of the dataset itself. The `comparison` folder contains the codes for a prediction comparison analysis between DeepLINK, IPAD and Random Forests.

## Citation

If you use DeepLINK, please cite the following paper:

[Zhu, Z., Fan, Y., Kong, Y., Lv, J., & Sun, F. (2021). DeepLINK: Deep learning inference using knockoffs with applications to genomics. *Proceedings of the National Academy of Sciences*, 118(36).](https://www.pnas.org/content/118/36/e2104683118.short)

## Copyright and License Information

Copyright (C) 2021 University of Southern California

Authors: Zifan Zhu, Yingying Fan, Yinfei Kong, Jinchi Lv, Fengzhu Sun

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

Commercial users should contact Dr. Fengzhu Sun (<fsun@usc.edu>) or Dr. Yingying Fan (<fanyingy@usc.edu>), copyright at University of Southern California.
