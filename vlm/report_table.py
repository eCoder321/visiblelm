"""Aggregate results/*/summary.json into markdown tables for REPORT.md.  python report_table.py [name ...]"""
import os, sys, json, glob

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def g(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d

def fmt(v, digits=3):
    if v is None: return '—'
    if isinstance(v, dict) and 'mean' in v:
        return f"{v['mean']:.{digits}f}" + (f" ±{v['std']:.{digits}f}" if v['n'] > 1 else '')
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

def rows(names):
    out = []
    for n in names:
        f = os.path.join(ROOT, n, 'summary.json')
        if not os.path.exists(f): continue
        s = json.load(open(f)); S = s['summary']; ns = s['n_seeds']
        an = g(S, 'analysis', 'main') or g(S, 'analysis', 'sae') or {}
        e = g(S, 'eval') or {}
        out.append({
            'name': n, 'seeds': ns, 'model': s['job']['cfg']['model'],
            'iid': fmt(g(e, 'iid', 'nll')), 'shift_sw': fmt(g(e, 'shift', 'nll_heldout_sw')), 'shift_arg': fmt(g(e, 'shift', 'nll_heldout_arg')),
            'acc_cpy2': fmt(g(e, 'shift', 'acc_heldout_cpy2'), 2), 'acc_cpy3': fmt(g(e, 'shift', 'acc_heldout_cpy3'), 2), 'acc_succ': fmt(g(e, 'shift', 'acc_heldout_succ'), 2),
            'long': fmt(g(e, 'long', 'nll')),
            'compl': fmt(g(an, 'completeness_gap')), 'mag': fmt(g(an, 'magnitude_audit', 'gap')),
            'suff': fmt(g(an, 'faithfulness_iid', 'sufficiency_retained_frac'), 2), 'suff_rand': fmt(g(an, 'faithfulness_iid', 'random_retained_frac'), 2),
            'argmax_ok': fmt(g(an, 'faithfulness_iid', 'sufficiency_argmax_ok'), 2), 'circ': fmt(g(an, 'faithfulness_iid', 'circuit_size_median'), 1),
            'circ_clean': fmt(g(an, 'faithfulness_iid', 'circuit_clean_frac'), 2), 'dl': fmt(g(an, 'description_length'), 0),
        })
    return out

def table(rs, cols):
    head = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    return head + ''.join('| ' + ' | '.join(str(r.get(c, '—')) for c in cols) + ' |\n' for r in rs)

if __name__ == '__main__':
    names = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(ROOT, '*', 'summary.json')))
    rs = rows(names)
    print("Generalization (NLL; oracle floors: iid 0.62, held-out switch 0.00, held-out argument 0.32)\n")
    print(table(rs, ['name', 'model', 'seeds', 'iid', 'shift_sw', 'shift_arg', 'acc_cpy2', 'acc_cpy3', 'acc_succ', 'long']))
    print("\nInterpretability (completeness gap, magnitude gap, circuit sufficiency vs random, argmax kept, circuit size, clean fraction, description length)\n")
    print(table(rs, ['name', 'compl', 'mag', 'suff', 'suff_rand', 'argmax_ok', 'circ', 'circ_clean', 'dl']))
