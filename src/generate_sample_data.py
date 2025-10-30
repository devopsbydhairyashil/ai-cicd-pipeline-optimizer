import pandas as pd
import numpy as np
import os
os.makedirs('data', exist_ok=True)
N = 1000
rng = np.random.default_rng(42)
df = pd.DataFrame({
    'commit_id': [f'c{i:04d}' for i in range(N)],
    'duration': rng.normal(300, 120, size=N).clip(30,2000).astype(int),
    'tests_run': rng.integers(10, 1000, size=N),
    'failures_last_24h': rng.integers(0,5,size=N),
    'changed_files': rng.integers(1,50,size=N),
})
# synthetic label: fail if long duration and many tests and random noise
df['failed'] = ((df['duration']>500) & (df['tests_run']>400) & (rng.random(N)>0.6)).astype(int)
df.to_csv('data/ci_sample.csv', index=False)
print('Generated data/data/ci_sample.csv')
