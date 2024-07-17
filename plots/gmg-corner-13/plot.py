"""
Generate plots demonstrating goal misgeneralisation.

Cheese in corner 13x13 boolean 20M steps
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


def load_row(metric):
    metric_path = f'data/{metric.replace("/","-")}.csv'
    with open(metric_path) as file:
        final_row = list(csv.DictReader(file))[-1]
        try:
            return [float(final_row[f'cs{i} - {metric}']) for i in range(1, 12)]
        except KeyError as e:
            print(final_row.keys())
            raise e

corner_sizes = np.arange(11)+1

fig, ax = plt.subplots()
ax.set(xlim=(0, 12), ylim=(0, 1))
ax.set_xlabel('corner size')
ax.set_xticks(corner_sizes)
ax.set_ylabel('return, avg. over some levels')
ax.set_title('Goal misgeneralisation(?)')

# training return
ax.plot(
    corner_sizes,
    load_row('env/train/avg_return_by_level'),
    label='train (last batch)',
    color='blue',
    marker='o',
)
ax.plot(
    corner_sizes,
    load_row('env/eval/on-distribution/avg_return_by_level'),
    label='eval (on distribution)',
    color='orange',
    marker='o',
)
ax.plot(
    corner_sizes,
    load_row('env/eval/off-distribution/avg_return_by_level'),
    label='eval (off distribution)',
    color='green',
    marker='o',
)

# benchmark baselines
ax.plot(
    corner_sizes,
    load_row('env/train/avg_benchmark_return_by_level'),
    label='optimal (last batch)',
    color='blue',
    linestyle='dashed',
)
ax.plot(
    corner_sizes,
    load_row('env/eval/on-distribution/avg_benchmark_return_by_level'),
    label='optimal (on distribution)',
    color='orange',
    linestyle='dashed',
)
ax.plot(
    corner_sizes,
    load_row('env/eval/off-distribution/avg_benchmark_return_by_level'),
    label='optimal (off distribution)',
    color='green',
    linestyle='dashed',
)

# proxy return
ax.plot(
    corner_sizes,
    load_row('env/train/proxy_corner/avg_return_by_level'),
    label='proxy (last batch)',
    color='blue',
    linestyle='dotted',
)
ax.plot(
    corner_sizes,
    load_row('env/eval/on-distribution/proxy_corner/avg_return_by_level'),
    label='proxy (on distribution)',
    color='orange',
    linestyle='dotted',
)
ax.plot(
    corner_sizes,
    load_row('env/eval/off-distribution/proxy_corner/avg_return_by_level'),
    label='proxy (off distribution)',
    color='green',
    linestyle='dotted',
)

plt.legend(loc='lower right')
plt.savefig('gmg-corner-13.png')

