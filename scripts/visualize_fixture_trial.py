"""Build an offline animation from a recorded trial and a separate manual review."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='reports/structural-cable-first-run')
    args = parser.parse_args()
    run = ROOT / args.run
    session = json.loads((run / 'session.json').read_text())
    payload = {
        'fixture': session['public_input'], 'history': session['history'],
        'run': {k: session.get(k) for k in ('protocol', 'created_at', 'status', 'reports',
                                           'evaluation_data_sent_to_actor')},
        'attempts': [{k: a.get(k) for k in ('phase', 'kind', 'status', 'diagnostic', 'usage')}
                     for a in session['attempts']],
        'reports': {phase: json.loads((run / (phase + '-report.json')).read_text())
                    if (run / (phase + '-report.json')).exists() else None
                    for phase in ('base', 'new_source')},
        'comparison': json.loads((run / 'manual-review.json').read_text()),
    }
    template = (ROOT / 'scripts/fixture_trial_template.html').read_text()
    assert template.count('__TRIAL_DATA__') == 1
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    output = ROOT / 'docs/stalker-fixture-trial.html'
    output.write_text(template.replace('__TRIAL_DATA__', data))
    print(output)


if __name__ == '__main__':
    main()
