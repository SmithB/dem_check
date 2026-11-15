#! /usr/bin/env python
"""Write a pboss queue of check_dem runs for a test-set csv (interleaving the sets)."""
import sys, os
import pandas as pd
csv, out_dir, queue = sys.argv[1], sys.argv[2], sys.argv[3]
extra = ' '.join(sys.argv[4:])
py = os.environ.get('LDCHECK_PYTHON', sys.executable)
env = 'NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4'
d = pd.read_csv(csv)
d['k'] = d.groupby('set').cumcount()
d = d.sort_values(['k', 'set'])
with open(queue, 'w') as fh:
    for name in d.DEM:
        fh.write(f'{env} {py} -m ldcheck.check_dem {name} --out_dir {out_dir} {extra}\n')
print(f'{len(d)} lines in {queue}')
