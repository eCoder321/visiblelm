"""Byte-level real-text corpus for the T-A0 experiments (docs/redesign/DESIGNS.md §11).

Deterministic: every source is pinned, and its sha256 is recorded in meta.json, so a GPU machine elsewhere rebuilds
the identical corpus. Sources (reachable through this project's egress policy as of 2026-10-24):
  prose  NLTK corpora mirrored on GitHub (gutenberg, brown [POS tags stripped], reuters, abc, state_union, webtext)
         and tiny-shakespeare
  code   .py files from the pinned PyPI sdist sympy==1.13.3, capped at CODE_CAP bytes
Large files are cut into <= DOC_MAX-byte documents at blank lines; documents are split train/val/test 96/2/2 by a hash
of (source, path, chunk index), so no document straddles splits. Documents are joined with a 0x00 byte.

    python lm/data.py --out data/lm            # downloads (~20 MB) and builds train.bin / val.bin / test.bin / meta.json
"""
import os, io, re, sys, json, hashlib, zipfile, tarfile, argparse, subprocess, tempfile
import numpy as np

NLTK = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip'
SOURCES = [('gutenberg', NLTK.format('gutenberg')), ('brown', NLTK.format('brown')), ('reuters', NLTK.format('reuters')),
           ('abc', NLTK.format('abc')), ('state_union', NLTK.format('state_union')), ('webtext', NLTK.format('webtext')),
           ('shakespeare', 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')]
CODE_PKG = 'sympy==1.13.3'
CODE_CAP = 12_000_000
DOC_MAX = 64_000
SPLIT = (0.96, 0.02, 0.02)

def fetch(url, dst):
    if not os.path.exists(dst):
        subprocess.run(['curl', '-sSfL', '--retry', '4', '-o', dst, url], check=True)
    return open(dst, 'rb').read()

def sha(b): return hashlib.sha256(b).hexdigest()

def chunk(text, cap=DOC_MAX):
    """Cut a large text into <= cap-byte pieces at blank lines (or hard cuts if a paragraph is huge)."""
    if len(text) <= cap: return [text]
    out, cur = [], b''
    for para in re.split(rb'(\n\s*\n)', text):
        if len(cur) + len(para) > cap and cur:
            out.append(cur); cur = b''
        while len(para) > cap:
            out.append(para[:cap]); para = para[cap:]
        cur += para
    if cur.strip(): out.append(cur)
    return out

def prose_docs(name, blob):
    """Yields (path, bytes) for each text file of a source."""
    if name == 'shakespeare':
        yield 'input.txt', blob; return
    z = zipfile.ZipFile(io.BytesIO(blob))
    for info in sorted(z.infolist(), key=lambda i: i.filename):
        fn = info.filename
        if info.is_dir() or '/.' in fn or fn.endswith(('README', 'cats.txt', 'stopwords', 'CONTENTS')): continue
        if name == 'brown' and not re.search(r'/c[a-r]\d\d$', fn): continue
        if name == 'reuters' and not re.search(r'/(training|test)/\d+$', fn): continue
        if name != 'brown' and name != 'reuters' and not fn.endswith('.txt'): continue
        b = z.read(info)
        if name == 'brown':  # "The/at Fulton/np-tl ..." -> "The Fulton ..."
            b = re.sub(rb'/[^\s]+', b'', b)
            b = re.sub(rb'[ \t]+', b' ', b)
        yield fn, b

def code_docs(cache):
    d = os.path.join(cache, 'sdist')
    os.makedirs(d, exist_ok=True)
    if not any(f.endswith('.tar.gz') for f in os.listdir(d)):
        subprocess.run([sys.executable, '-m', 'pip', 'download', CODE_PKG, '--no-deps', '--no-binary', ':all:', '-d', d,
                        '-q'], check=True)
    tgz = os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith('.tar.gz'))[0])
    blob = open(tgz, 'rb').read()
    tf = tarfile.open(fileobj=io.BytesIO(blob))
    total, docs = 0, []
    for m in sorted((m for m in tf.getmembers() if m.isfile() and m.name.endswith('.py')), key=lambda m: m.name):
        b = tf.extractfile(m).read()
        if total + len(b) > CODE_CAP: break
        docs.append((m.name, b)); total += len(b)
    return os.path.basename(tgz), sha(blob), docs

def split_of(key):
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 16**8
    return 'train' if h < SPLIT[0] else ('val' if h < SPLIT[0] + SPLIT[1] else 'test')

def build(out, cache=None):
    cache = cache or os.path.join(out, 'cache')
    os.makedirs(cache, exist_ok=True)
    meta = {'sources': {}, 'splits': {}}
    buckets = {s: [] for s in ['train', 'val', 'test']}
    def add(source, path, b, cat):
        b = b.replace(b'\r\n', b'\n').replace(b'\x00', b'')
        for i, c in enumerate(chunk(b)):
            if c.strip(): buckets[split_of(f'{source}/{path}/{i}')].append((c, cat))
    for name, url in SOURCES:
        blob = fetch(url, os.path.join(cache, os.path.basename(url)))
        meta['sources'][name] = {'url': url, 'sha256': sha(blob)}
        for path, b in prose_docs(name, blob): add(name, path, b, 'prose')
    fn, h, docs = code_docs(cache)
    meta['sources']['code'] = {'pkg': CODE_PKG, 'file': fn, 'sha256': h}
    for path, b in docs: add('code', path, b, 'code')
    for split, docs in buckets.items():
        parts, spans, pos = [], [], 0
        for c, cat in docs:
            parts.append(c + b'\x00'); spans.append((pos, pos + len(c), cat)); pos += len(c) + 1
        arr = np.frombuffer(b''.join(parts), np.uint8)
        arr.tofile(os.path.join(out, f'{split}.bin'))
        meta['splits'][split] = {'bytes': int(arr.size), 'docs': len(docs), 'sha256': sha(arr.tobytes()),
                                 'bytes_by_cat': {k: int(sum(e - s for s, e, c in spans if c == k)) for k in ['prose', 'code']},
                                 'spans': spans if split != 'train' else None}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    return meta

class Corpus:
    """Memory-mapped splits. batch() samples training windows; eval_windows() gives fixed windows per category."""
    def __init__(self, path):
        self.meta = json.load(open(os.path.join(path, 'meta.json')))
        self.data = {s: np.memmap(os.path.join(path, f'{s}.bin'), np.uint8, 'r') for s in ['train', 'val', 'test']}

    def batch(self, rng, B, T):
        d = self.data['train']; st = rng.integers(0, d.size - T - 1, B)
        return np.stack([d[s:s + T + 1] for s in st]).astype(np.int32)

    def eval_windows(self, split, cat, n, T, seed=0):
        """n windows of T+1 bytes, each lying inside one document of category cat (deterministic)."""
        spans = [(s, e) for s, e, c in self.meta['splits'][split]['spans'] if c == cat and e - s > T + 1]
        rng = np.random.default_rng(seed); d = self.data[split]; out = []
        w = np.array([e - s - T - 1 for s, e in spans], float); w /= w.sum()
        for i in rng.choice(len(spans), n, p=w):
            s, e = spans[i]; o = s + rng.integers(0, e - s - T - 1)
            out.append(d[o:o + T + 1])
        return np.stack(out).astype(np.int32)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='data/lm'); ap.add_argument('--cache')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m = build(a.out, a.cache)
    for s, v in m['splits'].items(): print(s, v['bytes'], 'bytes', v['docs'], 'docs', v['bytes_by_cat'])
