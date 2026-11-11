import os,subprocess,sys
# nvidia-smi is a read-only query; no CUDA context or allocation.
query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
if query.returncode:print(query.stderr);sys.exit(1)
rows=[tuple(x.strip() for x in row.split(',')) for row in query.stdout.splitlines() if row.strip()]
# The queue uses the default first GPU on this single-A100 host.
if not rows:sys.exit(1)
index,uuid,free=rows[0]
print(f'GPU {index} {uuid}: {free} MiB free (required 48000 MiB).',flush=True)
sys.exit(0 if int(free)>=48000 else 1)
