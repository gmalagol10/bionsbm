import warnings
warnings.filterwarnings('ignore')

import muon as mu
import bionsbm
import time
import os

print(f"nSBM script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

cm=sys.argv[1]
path_to_save=sys.argv[2]

folder=os.path.dirname(path_to_save)
name=path_to_save.split("/")[-1]

names=["Script", "CM", "Path to save"]
for nm,arg in zip(names,sys.argv):
	print(nm,":", arg, flush=True)

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Reading count matrix", flush=True)
mdata=mu.read_h5mu(cm)


for run in range(0, 10):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run {run}/{runs} Fitting bionsbm model", flush=True)

	model = bionsbm.model.bionsbm(obj=mdata, saving_path="{folder}/Runs/Run{run}/{name}_")

	model.fit(n_init=7, verbose=False)
