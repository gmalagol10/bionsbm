#!/usr/bin/env python3
from pathlib import Path
import importlib
import importlib.metadata as metadata
import inspect
import shutil
import subprocess
import sys
import tempfile

ROOT = Path('Datasets/GSE268630')
ATAC_SCRIPT = Path('03_process_ATAC.py')
failures = []


def ok(message):
    print(f'[OK]   {message}', flush=True)


def fail(message):
    failures.append(message)
    print(f'[FAIL] {message}', flush=True)


def warn(message):
    print(f'[WARN] {message}', flush=True)


print('Python:', sys.executable)
print('Version:', sys.version.replace('\n', ' '))

modules = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'h5py': 'h5py',
    'anndata': 'anndata',
    'scanpy': 'scanpy',
    'snapatac2': 'snapatac2',
    'muon': 'muon',
    'skmisc': 'scikit-misc',
}

loaded = {}
for module_name, distribution_name in modules.items():
    try:
        loaded[module_name] = importlib.import_module(module_name)
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = getattr(loaded[module_name], '__version__', 'unknown')
        ok(f'{module_name} {version}')
    except Exception as error:
        fail(f'cannot import {module_name}: {type(error).__name__}: {error}')

if not failures:
    import numpy as np
    import scipy.sparse as sp
    import anndata as ad
    import scanpy as sc
    import snapatac2 as snap
    import muon as mu

    try:
        from skmisc.loess import loess
        ok('from skmisc.loess import loess')
    except Exception as error:
        fail(f'scikit-misc LOESS import: {type(error).__name__}: {error}')

    required_callables = {
        'anndata.read_h5ad': ad.read_h5ad,
        'scanpy.pp.calculate_qc_metrics': sc.pp.calculate_qc_metrics,
        'scanpy.pp.normalize_total': sc.pp.normalize_total,
        'scanpy.pp.highly_variable_genes': sc.pp.highly_variable_genes,
        'scanpy.pp.log1p': sc.pp.log1p,
        'snapatac2.read': snap.read,
        'snapatac2.pp.make_peak_matrix': getattr(snap.pp, 'make_peak_matrix', None),
        'muon.MuData': mu.MuData,
        'muon.read_h5mu': mu.read_h5mu,
    }
    for name, obj in required_callables.items():
        if callable(obj):
            ok(name)
        else:
            fail(f'{name} is missing or not callable')

    if hasattr(snap.genome, 'hg38'):
        ok('snapatac2.genome.hg38')
    else:
        fail('snapatac2.genome.hg38 is missing')

    source = ATAC_SCRIPT.read_text() if ATAC_SCRIPT.exists() else ''
    if 'snap.pp.import_fragments' in source:
        import_name = 'import_fragments'
    elif 'snap.pp.import_data' in source:
        import_name = 'import_data'
    else:
        import_name = None
        warn(f'Could not detect the SnapATAC2 import call in {ATAC_SCRIPT}')

    if import_name:
        import_fn = getattr(snap.pp, import_name, None)
        if callable(import_fn):
            ok(f'snapatac2.pp.{import_name}')
        else:
            fail(f'the script calls snap.pp.{import_name}, but it is unavailable')

    make_peak_matrix = getattr(snap.pp, 'make_peak_matrix', None)
    if callable(make_peak_matrix):
        try:
            signature = inspect.signature(make_peak_matrix)
            parameters = signature.parameters
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
            for argument in ['peak_file', 'file', 'chunk_size', 'counting_strategy']:
                if argument in parameters or has_kwargs:
                    ok(f'make_peak_matrix supports {argument!r}')
                else:
                    fail(f'make_peak_matrix does not support {argument!r}: {signature}')
        except (TypeError, ValueError) as error:
            warn(f'Could not inspect make_peak_matrix signature: {error}')

    try:
        rng = np.random.default_rng(1)
        test = ad.AnnData(sp.csr_matrix(rng.poisson(1, (200, 100)).astype(np.float32)))
        sc.pp.highly_variable_genes(
            test,
            flavor='seurat_v3',
            n_top_genes=20,
            subset=False,
            check_values=False,
        )
        if 'variances_norm' not in test.var:
            raise RuntimeError('variances_norm was not created')
        ok('Seurat-v3 HVG smoke test')
    except Exception as error:
        fail(f'Seurat-v3 HVG smoke test: {type(error).__name__}: {error}')

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.h5mu'
            test = ad.AnnData(sp.eye(3, format='csr', dtype=np.float32))
            mdata = mu.MuData({'test': test})
            mdata.write_h5mu(path)
            reloaded = mu.read_h5mu(path)
            if reloaded['test'].shape != (3, 3):
                raise RuntimeError(f"unexpected shape {reloaded['test'].shape}")
        ok('MuData write/read smoke test')
    except Exception as error:
        fail(f'MuData write/read smoke test: {type(error).__name__}: {error}')

for executable in ['gzip', 'awk', 'macs3']:
    path = shutil.which(executable)
    if path:
        ok(f'{executable}: {path}')
    else:
        fail(f'executable not found: {executable}')

pigz = shutil.which('pigz')
if pigz:
    ok(f'pigz: {pigz}')
else:
    warn('pigz not found; the script can fall back to gzip')

macs3 = shutil.which('macs3')
if macs3:
    try:
        result = subprocess.run([macs3, 'callpeak', '-h'], text=True, capture_output=True, check=False)
        help_text = result.stdout + result.stderr
        for option in ['FRAG', '--barcodes', '--max-count']:
            if option in help_text:
                ok(f'MACS3 callpeak supports {option}')
            else:
                fail(f'MACS3 callpeak help does not contain {option}')
    except Exception as error:
        fail(f'MACS3 API check: {type(error).__name__}: {error}')

required_files = [
    ROOT / 'GSE268630_merged_fragments.tsv.gz',
    ROOT / 'GSE268630_ATAC_called_barcodes.txt',
    ROOT / 'peaks' / '.macs3_complete',
    ROOT / 'peaks' / 'GSE268630_all_peaks_peaks.narrowPeak',
    ROOT / 'peaks' / 'GSE268630_all_peaks.bed',
    ROOT / 'GSE268630_fragments.snap.h5ad',
    ROOT / 'GSE268630_ATAC_raw_peak_matrix.h5ad',
    ROOT / 'GSE268630_GEX_processed.h5ad',
]
for path in required_files:
    if path.is_file() and path.stat().st_size > 0:
        ok(f'{path} ({path.stat().st_size / 1e9:.2f} GB)')
    else:
        fail(f'missing or empty file: {path}')

if not failures:
    try:
        atac = loaded['anndata'].read_h5ad(ROOT / 'GSE268630_ATAC_raw_peak_matrix.h5ad', backed='r')
        gex = loaded['anndata'].read_h5ad(ROOT / 'GSE268630_GEX_processed.h5ad', backed='r')
        common = atac.obs_names.intersection(gex.obs_names)
        ok(f'raw ATAC matrix readable: {atac.shape}')
        ok(f'processed GEX matrix readable: {gex.shape}')
        if len(common):
            ok(f'matching GEX/ATAC barcodes: {len(common)}')
        else:
            fail('GEX and ATAC matrices have no matching barcodes')
        atac.file.close()
        gex.file.close()
    except Exception as error:
        fail(f'backed matrix/barcode test: {type(error).__name__}: {error}')

try:
    usage = shutil.disk_usage(ROOT)
    print(f'[INFO] Free space under {ROOT}: {usage.free / 1e9:.1f} GB')
except Exception as error:
    warn(f'Could not inspect free space: {error}')

print()
if failures:
    print(f'PRE-FLIGHT FAILED: {len(failures)} problem(s)')
    for problem in failures:
        print(f' - {problem}')
    sys.exit(1)

print('PRE-FLIGHT PASSED')
