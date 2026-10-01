"""Protect scientific qualifications in the public artifact."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_preview_identifier_is_not_presented_as_a_dated_snapshot():
    provenance = json.loads((ROOT / 'data/frontier/provenance.json').read_text())
    readme = (ROOT / 'README.md').read_text()
    assert provenance['model'] == 'gpt-5.6-terra'
    assert provenance['model_snapshot'] is None
    assert provenance['accessed_date'] == '2026-08-05'
    assert 'preview model' in readme
    assert 'not pinned to a dated snapshot' in readme


@pytest.mark.parametrize('module,args', [
    ('market.best_response', ['--n', '2']),
    ('market.best_response', ['--n', '2', '--draw', 'wide']),
    ('market.blind_search', ['--screen', '1', '--episodes', '2', '--top', '1', '--draw', 'wide']),
])
def test_search_is_presented_as_a_lower_reference(module, args):
    result = subprocess.run([sys.executable, '-m', module, *args], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert 'searched comparator' in result.stdout
    assert 'ceiling' not in result.stdout
    assert 'gap ' not in result.stdout
