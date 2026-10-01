"""Build offline blog figures from archived evidence. No network or experiment runs."""
from pathlib import Path
import hashlib
import json
import random
import re
import sys
import subprocess
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from market.env import Draw, Env, parse_action
from market.vf_env import make_dataset, BOT_FACTORIES

sources = {}
def read(path):
    p = ROOT / path
    sources[path] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
    return json.loads(p.read_text())

archive = 'data/training'
metrics = read(archive + '/metrics.json')['metrics']
keys = ['reward/all/mean'] + [f'{prefix}/wide-{role}/{metric}' for role in ['seller', 'buyer'] for prefix, metric in [('reward', 'mean'), ('filters', 'zero_advantage'), ('metrics', 'deal_rate'), ('metrics', 'timeout_rate'), ('metrics', 'walk_rate'), ('metrics', 'forfeit_rate')]]
curves = [{k: row[k] for k in ['step'] + keys} for row in metrics]
datasets = {role: make_dataset(1536, seed=seed, opponent='mix', seat=role, c_lo=40, c_hi=60, v_lo=90, v_hi=110, t_lo=2, t_hi=6) for role, seed in [('seller', 0), ('buyer', 1)]}
openings = []
archived = []
endings = []
for step in [1, 5, 25, 50]:
    samples = read(archive + f'/rollouts-{step}.json')['samples']
    assert len(samples) == 64
    outcome_counts = Counter()
    excluded = Counter()
    for sample in samples:
        role = sample['env_name'].split('-')[-1]
        messages = json.loads(sample['completion'])
        first = next(m for m in messages if m['role'] == 'assistant')
        own = int(re.search(r'Your (?:cost c|value v) = (\d+)', messages[0]['content'])[1])
        action = parse_action(first['content'], None, 0, 150)
        if action and action[0] == 'OFFER':
            openings.append({'step': step, 'role': role, 'own': own, 'price': action[1]})
        else:
            excluded[role] += 1
        m = json.loads(sample['metrics'])
        found = [k for k in ['deal', 'timeout', 'walk', 'forfeit', 'breakdown'] if m[k + '_rate'] == 1]
        assert len(found) == 1, found
        outcome_counts[found[0]] += 1
        if step == 50:
            info = datasets[role][sample['problem_id']]['info']
            assert info['c' if role == 'seller' else 'v'] == own
            assert datasets[role][sample['problem_id']]['prompt'][0]['content'] == messages[0]['content']
            draw = Draw(info['c'], info['v'], (info['horizon']['mode'], info['horizon']['param']))
            env = Env(draw, random.Random(info['seed']), first_mover=role)
            opposite = 'buyer' if role == 'seller' else 'seller'
            bot = BOT_FACTORIES[info['opponent']](opposite, info['v'] if opposite == 'buyer' else info['c'], info['opp_anchor'], info['opp_e'])
            assistant = iter(m['content'] for m in messages if m['role'] == 'assistant')
            while not env.done:
                text = next(assistant) if env.turn == role else bot.act(env)
                actor = env.turn
                env.step(text)
            actual_reward = env.rewards()[0 if role == 'seller' else 1]
            assert abs(actual_reward - sample['reward']) < 1e-10
            assert env.outcome[0] == found[0]
            # Every stored opponent offer must be present in the replay.
            observed = [re.search(r'Opponent: (.*)', msg['content'])[1] for msg in messages if msg['role'] == 'user' and 'Opponent: ' in msg['content']]
            replayed = [text for who, text in env.transcript if who == opposite]
            assert replayed[:len(observed)] == observed
            assert next(assistant, None) is None
            endings.append({'family': info['opponent'], 'role': role, 'outcome': found[0], 'actor': 'learner' if actor == role else 'opponent', 'reward': sample['reward'], 'price': env.outcome[1] if found[0] == 'deal' else None})
    archived.append({'step': step, 'n': len(samples), 'outcomes': dict(outcome_counts), 'excluded_openings': dict(excluded), 'unique_setups': len({(s['env_name'], s['problem_id']) for s in samples})})
assert Counter(e['family'] for e in endings) == {'boulware': 31, 'conceder': 10, 'random_threshold': 23}
assert Counter(e['outcome'] for e in endings) == {'deal': 29, 'timeout': 35}
assert all(e['actor'] == 'opponent' for e in endings if e['outcome'] == 'deal')
assert all(o['price'] == (95 if o['role'] == 'seller' else 55) for o in openings if o['step'] == 50)

bounds = read('data/references/bounds.json')
lower = read('data/references/vs-blind.json')['results']['ext']
upper = read('data/references/vs-upper.json')['results']
data = {'curves': curves, 'archived': archived, 'openings': openings, 'endings': endings, 'bounds': bounds, 'frontier': {'lower': lower, 'upper': upper}}
(HERE / 'data.json').write_text(json.dumps(data, indent=2) + '\n')
for path in ['configs/training.toml', 'market/env.py', 'market/bots.py', 'market/vf_env.py', 'data/frontier/episodes-seller.json', 'data/frontier/episodes-buyer.json', 'data/frontier/provenance.json']:
    p = ROOT / path
    sources[path] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
(HERE / 'sources.json').write_text(json.dumps({'sources': sources, 'reconstruction': '1536 rows per role, seeds seller 0 / buyer 1, config training.toml. Every step-50 prompt, reward, ending and observed opponent offer verified against reconstruction.', 'scope': 'Training run two in blog. Metric step labels preserved verbatim, 0 through 51. Archived pages at 1, 5, 25, 50 contain 64 of 512 rollouts each. Repeated setups retained. Sample-level zero advantage, not group share.', 'omissions': ['No frozen evaluation for the trained 4B model: omitted from reference comparison.', 'No hypothetical profitability classification for timeout/walk, where no agreed price exists.', 'No extrapolated archive data between steps. No independent-episode confidence intervals for repeated training rollouts.', 'Only the archived exploratory frontier comparison shown.']}, indent=2) + '\n')
html = (HERE / 'template.html').read_text().replace('/*DATA*/', 'const DATA=' + json.dumps(data) + ';').replace('/*CODE*/', (HERE / 'figures.js').read_text())
(HERE / 'index.html').write_text(html)
subprocess.run(['node', str(HERE / 'export.js')], check=True)
print('Built data, manifest and self-contained HTML; reconstructed all 64 step-50 rollouts.')
