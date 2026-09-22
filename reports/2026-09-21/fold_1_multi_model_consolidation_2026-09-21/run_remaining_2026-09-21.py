"""Continue numerical selection, fitting, and evaluation after global review."""
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
while not (HERE/'global_review_complete.json').exists():
    time.sleep(5)
for phase,script,done in [('size_selection','choose_size_2026-09-21.py','selection_frozen_2026-09-21.json'),
                          ('fitting','fit_2026-09-21.py','predictions_frozen_2026-09-21.json'),
                          ('evaluation','evaluate_2026-09-21.py','report_complete_2026-09-21.json'),
                          ('final_validation','verify_final_2026-09-21.py',None)]:
    if done and (HERE/done).exists():
        print(f'{phase} already complete; downstream integrity checks will verify its hashes',flush=True)
        continue
    print(f'Starting {phase}',flush=True)
    with (HERE/f'{phase}.log').open('a') as out:
        subprocess.run([sys.executable,'-u',str(HERE/script)],stdout=out,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {phase}',flush=True)
