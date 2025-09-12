# bionsbm
### Graph-based topic modelling for the integration single-cell multi-omica data

## Installation

```bash

conda create --name gt -c conda-forge graph-tool

conda activate gt

pip install git+https://github.com/gmalagol10/bionsbm

```

## Usage

```python
import bionsbm
import muon as mu

mdata=mu.read_h5mu("Test_data.h5mu")

model = bionsbm.model.bionsbm(mdata)

model.fit()


```



## Version History

* 0.1
    * Initial Release


## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details


