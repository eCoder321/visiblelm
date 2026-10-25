"""Run the T-A0 experiments on Modal GPUs, one container per (job, seed).

Setup (each fresh session container): pip install 'modal[api-proxy-support]'. The proxy extra is what lets the
client reach Modal through this environment's HTTPS proxy. Credentials come from MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.

    modal deploy lm/modal_app.py                       # (re)build the image with the current lm/ code
    python lm/modal_app.py data                        # build the corpus on the volume (once; skipped if present)
    python lm/modal_app.py launch [job ...] [--smoke]  # spawn a GPU call per (job, seed); ids -> results/lm/modal_calls.json
    python lm/modal_app.py status                      # done / running / failed per call
    python lm/modal_app.py collect                     # copy result JSONs and logs from the volume into results/lm/
    python lm/modal_app.py analyze [job ...]           # mechanism check (attention spread, unsampled pressure) -> results/lm/ta0s1_mechanism.json

Calls run on Modal, independent of this machine. Checkpoints are committed to the volume at every save, so a
preempted or retried call resumes where it stopped (run.py resumes from checkpoints; finished seeds are skipped).
"""
import os, sys, json, time
import modal

APP, VOLUME = 'visiblelm-ta0', 'visiblelm-data'
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
STAGE1 = ['peer_sig_lam0', 'peer_sig_lam0.03', 'peer_sig_lam0.1', 'peer_sig_lam0.3', 'dense_d256', 'dense_d192',
          'dense_d128', 'peer_softmax', 'dense_d256_lam0.1']

vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.13').apt_install('curl')
         .pip_install('jax[cuda12]==0.12.0.1', 'optax==0.2.8', 'numpy')
         # XLA's Triton GEMM fusion miscompiled the frontier replay on an L4 (jit diverged from eager by up to 2.0 in
         # logits at layer 1's attention output; exact again with Triton GEMMs off). Disabled for every GPU run,
         # training included, so no result depends on it.
         .env({'XLA_FLAGS': '--xla_gpu_enable_triton_gemm=false'})
         .add_local_dir(HERE, remote_path='/root/lm', ignore=['__pycache__']))
app = modal.App(APP, image=image)

@app.function(volumes={'/vol': vol}, timeout=3600, cpu=2)
def build_data():
    import subprocess
    vol.reload()
    if not os.path.exists('/vol/data/lm/meta.json'):
        subprocess.run([sys.executable, '/root/lm/data.py', '--out', '/vol/data/lm', '--cache', '/tmp/cache'], check=True)
        vol.commit()
    m = json.load(open('/vol/data/lm/meta.json'))
    return {s: {k: v[k] for k in ['bytes', 'docs', 'sha256']} for s, v in m['splits'].items()}

@app.function(gpu='L4', volumes={'/vol': vol}, timeout=8 * 3600, cpu=4, memory=16384,
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def run_seed(job: str, seed: int, smoke: bool = False):
    import shutil
    vol.reload()
    os.makedirs('/tmp/data/lm', exist_ok=True)                 # local copy: memmapped random reads stay off the volume
    for f in ['train.bin', 'val.bin', 'test.bin', 'meta.json']:
        shutil.copy(f'/vol/data/lm/{f}', f'/tmp/data/lm/{f}')
    sys.path.insert(0, '/root/lm')
    import run
    save = run.save
    def save_and_commit(path, obj):
        save(path, obj); vol.commit()
    run.save = save_and_commit
    sys.argv = ['run.py', f'/root/lm/jobs/ta0s1/{job}.json', '--data', '/tmp/data/lm', '--out', f'/vol/results/lm/seed{seed}',
                '--seeds', str(seed)] + (['--smoke'] if smoke else [])
    t0 = time.time(); run.main(); vol.commit()
    return {'job': job, 'seed': seed, 'smoke': smoke, 'seconds': time.time() - t0}

@app.function(gpu='L4', volumes={'/vol': vol}, timeout=3600, cpu=4, memory=32768)
def analyze(jobs: list):
    import shutil
    vol.reload(); os.makedirs('/tmp/data/lm', exist_ok=True)
    for f in ['val.bin', 'meta.json', 'train.bin', 'test.bin']: shutil.copy(f'/vol/data/lm/{f}', f'/tmp/data/lm/{f}')
    sys.path.insert(0, '/root/lm')
    import analyze_ta0
    return {'mechanism': analyze_ta0.run('/vol/results/lm', '/tmp/data/lm', jobs),
            'cancellation': analyze_ta0.run_cancel('/vol/results/lm', '/tmp/data/lm', jobs)}

@app.function(volumes={'/vol': vol}, timeout=600)
def collect_files():
    vol.reload(); out = {}
    for root, _, files in os.walk('/vol/results/lm'):
        for f in files:
            if f.endswith(('.json', '.txt')):
                p = os.path.join(root, f); out[os.path.relpath(p, '/vol/results/lm')] = open(p).read()
    return out

def _fn(name): return modal.Function.from_name(APP, name)

def main(argv):
    calls_path = os.path.join(ROOT, 'results', 'lm', 'modal_calls.json'); os.makedirs(os.path.dirname(calls_path), exist_ok=True)
    calls = json.load(open(calls_path)) if os.path.exists(calls_path) else []
    cmd, args = argv[0], argv[1:]
    if cmd == 'data':
        print(json.dumps(_fn('build_data').remote(), indent=1))
    elif cmd == 'launch':
        smoke = '--smoke' in args; jobs = [a for a in args if not a.startswith('--')] or STAGE1
        fn = _fn('run_seed')
        for j in jobs:
            for s in json.load(open(os.path.join(HERE, 'jobs', 'ta0s1', j + '.json'))).get('seeds', [0]):
                fc = fn.spawn(j, s, smoke)
                calls.append({'job': j, 'seed': s, 'smoke': smoke, 'id': fc.object_id, 'launched': time.strftime('%F %T')})
                print('launched', j, 'seed', s, fc.object_id)
        json.dump(calls, open(calls_path, 'w'), indent=1)
    elif cmd == 'status':
        for c in calls:
            try:
                r = modal.FunctionCall.from_id(c['id']).get(timeout=0); st = f"done in {r['seconds'] / 60:.1f} min"
            except TimeoutError:
                st = 'running'
            except Exception as e:
                st = f'FAILED: {type(e).__name__}: {str(e)[:200]}'
            print(f"{c['job']:20s} seed {c['seed']} {'smoke ' if c['smoke'] else ''}{c['launched']}  {st}")
    elif cmd == 'analyze':
        res = _fn('analyze').remote(args or STAGE1)
        p = os.path.join(ROOT, 'results', 'lm', 'ta0s1_mechanism.json'); json.dump(res, open(p, 'w'), indent=1)
        print('wrote', p)
    elif cmd == 'collect':
        files = _fn('collect_files').remote(); dst = os.path.join(ROOT, 'results', 'lm')
        for rel, text in files.items():
            p = os.path.join(dst, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'w').write(text)
        print(f'collected {len(files)} files into {dst}')
    else:
        print(__doc__)

if __name__ == '__main__':
    main(sys.argv[1:] or ['help'])
