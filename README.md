# bionSBM

Integration of single-cell multi-omic data with graph-based topic modelling

Gabriele Malagoli, Filippo Valle, Andreina Tirabassi, Annalisa Marsico, Loredana Martignetti, Michele Caselle & Maria Colomé-Tatché

https://doi.org/10.64898/2026.02.25.707947



## Installation

```bash

conda create --name gt -c conda-forge graph-tool

conda activate gt

pip install git+https://github.com/gmalagol10/bionsbm

```

## Usage

```python
import bionsbm
from muon import read_h5mu

mdata = read_h5mu("Test_data.h5mu")

model = bionsbm.model.bionsbm(obj=mdata, saving_path="results/mybionsbm")

model.fit(n_init=1, verbose=False)


```



## Version History

* 0.1
    * Initial Release


## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details


