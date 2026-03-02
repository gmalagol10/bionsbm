import warnings
warnings.filterwarnings('ignore')

import muon as mu
import bionsbm
import time
import sys
import os

print(f"bionSBM script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

cm = sys.argv[1]
path_to_save = sys.argv[2]
start = int(sys.argv[3])

name=path_to_save.split("/")[-1]

names=["Script", "CM", "Path to save"]
for nm,arg in zip(names,sys.argv):
	print(nm,":", arg, flush=True)

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Reading count matrix", flush=True)
obj=mu.read(cm)

#Run 0 creates the graph
if os.path.isfile(f"{path_to_save}/{name}_graph.xml.gz") == False:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run 0/25 Fitting bioSBM model", flush=True)
	model = bionsbm.model.bionsbm(obj=obj, saving_path=f"{path_to_save}/Runs/Run0/{name}", save_graph_path=f"{path_to_save}/{name}_graph.xml.gz")
	model.fit(n_init=7, verbose=False)
else:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"First run already done, graph already exist --> skipping graph creation", flush=True)

### The other runs use the same graph
for run in range(start,25):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run {run}/25 Fitting bioSBM model", flush=True)
	model = bionsbm.model.bionsbm(obj=obj, saving_path=f"{path_to_save}/Runs/Run{run}/{name}", load_graph_path=f"{path_to_save}/{name}_graph.xml.gz")
	model.fit(n_init=7, verbose=False)
