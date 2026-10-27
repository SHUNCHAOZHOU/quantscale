"""Plot completed open-code results without converting failures to zero."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    root = Path(root)
    payload = json.loads((root / 'open-report-data.json').read_text())
    runs = payload['runs']
    order = ['single', 'independent', 'centralized', 'decentralized', 'hybrid']
    labels = ['Single', 'Independent', 'Centralized', 'Decentralized', 'Hybrid']
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    colors = ['#888888', '#22658a']
    for i, topology in enumerate(order):
        selected = [r for r in runs if r['topology'] == topology]
        valid = [r for r in selected if r['status'] == 'evaluated']
        if valid:
            for j, field in enumerate(['validation_sharpe', 'hidden_sharpe']):
                values = [r[field] for r in valid]
                xpos = i + (j - .5) * .3
                axes[0].bar(xpos, np.mean(values), width=.27, color=colors[j],
                            label=['Validation 2022-2023', 'Test 2024-2025'][j] if i == next(k for k,t in enumerate(order) if any(r['topology']==t and r['status']=='evaluated' for r in runs)) else None)
                axes[0].scatter([xpos]*len(values), values, color='black', s=20, zorder=3)
        else:
            axes[0].text(i, .02, 'No test', ha='center', va='bottom', rotation=90)
        axes[1].bar(i-.16, sum(r['trials'] for r in selected), width=.3, color=colors[0], label='Attempted candidates' if i==0 else None)
        axes[1].bar(i+.16, sum(r['valid'] for r in selected), width=.3, color=colors[1], label='Candidates with train + validation' if i==0 else None)
    axes[0].set_title('Open Python strategies: net Sharpe by topology')
    axes[0].set_ylabel('Annualized net Sharpe (252; cash baseline 0)')
    axes[0].axhline(0, color='black', linewidth=.7)
    axes[0].set_xticks(range(5), [labels[i]+'\n%d/2 tested'%sum(r['status']=='evaluated' for r in runs if r['topology']==t) for i,t in enumerate(order)])
    axes[0].set_xlabel('Topology; bars = successful-test means, dots = individual runs')
    axes[1].set_title('Research completion: failures retained in candidate counts')
    axes[1].set_ylabel('Candidates (count across two planned runs)')
    axes[1].set_xlabel('Topology')
    axes[1].set_xticks(range(5), labels)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        handles, names = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, names, frameon=False, fontsize=9)
    figure.suptitle('QuantScale / DeepSeek Flash / 500 current S&P constituents', fontsize=14)
    figure.text(.01, -.035, 'Source: local open pilot journals and one-time evaluations. Train 2016-2021; historical test reused from previous pilot.\n'
                '170k token / 12 candidate cap per run. Missing tests excluded from Sharpe means; not zero-filled.', fontsize=9)
    for extension in ['png', 'svg']:
        figure.savefig(root / ('open-topology-comparison.'+extension), dpi=160, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    plot(parser.parse_args().root)
