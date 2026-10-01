import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import paired_vs_blind as lower
import paired_vs_upper as upper


def check(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            check(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            check(a, e, f'{path}[{i}]')
    elif isinstance(expected, (float, int)):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def main():
    manifest = json.loads((ROOT / 'figures/sources.json').read_text())
    for name, record in manifest['sources'].items():
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
    previous = json.loads((ROOT / 'figures/data.json').read_text())
    subprocess.run([sys.executable, str(ROOT / 'figures/build.py')], cwd=ROOT, check=True)
    check(json.loads((ROOT / 'figures/data.json').read_text()), previous, 'figure data')
    subprocess.run(['node', str(ROOT / 'figures/export.js'), '--check'], check=True)
    lo = json.loads((ROOT / 'data/references/vs-blind.json').read_text())
    hi = json.loads((ROOT / 'data/references/vs-upper.json').read_text())
    for role in ['seller', 'buyer']:
        check(lower.paired(ROOT / 'data/frontier', role, lower.WINNERS['ext'],
                           lo['bootstrap']['seed'], lo['bootstrap']['B']),
              lo['results']['ext'][role], f'{role} lower comparison')
        check(upper.paired(ROOT / 'data/frontier', role,
                           hi['bootstrap']['seed'], hi['bootstrap']['B']),
              hi['results'][role], f'{role} upper comparison')
    print('Source hashes, all figures, step-50 replay, and both frontier comparisons verified.')


if __name__ == '__main__':
    main()
