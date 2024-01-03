#!/bin/env python3

import math
import sys
import json
import graphviz
from graphviz import Source
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

args = sys.argv

if len(args) != 3:
    myname = args[0]
    print(f'''Usage: {myname} input_path output_image_path

(Example)
cd tamago
echo 'tamago-read_sgf (;SZ[9]KM[7];B[fe];W[de];B[ec])
lz-genmove_analyze 10000
tamago-dump_tree' \\
| python3 main.py --model model/model.bin --strict-visits 100 \\
| grep parent_index > tree.json
{myname} tree.json tree_graph
display tree_graph.png

# Use "tamago-dump_tree aroundPV true" to focus on the principal variation.
''')
    sys.exit(1)

input_path = args[1]
output_image_path = args[2]

with open(input_path, 'r') as file:
    data = json.load(file)

to_move = data["to_move"]
node_by_index = data["node_by_index"]
node_list = list(node_by_index.values())

# colormap = plt.cm.get_cmap('coolwarm_r')
colormap = plt.cm.get_cmap('Spectral')
# colormap = plt.cm.get_cmap('RdYlBu')
# colormap = plt.cm.get_cmap('viridis')

def get_color(value):
    emphasis = 1.5
    v = 0.5 + (value - 0.5) * emphasis
    return mcolors.to_hex(colormap(v))

def get_size(visits, shape):
    size0 = 0.5 + math.log10(visits)
    size = size0 if shape == 'square' else size0 * 2 / (math.pi ** 0.5)
    return f'{size}'

def odd(x):
    return x % 2 == 1

def flip_maybe(v, lvl):
    return v if odd(level) else 1.0 - v

dot = graphviz.Digraph(comment='Visualization of MCTS Tree')

node_list.sort(key=lambda node: node['visits'], reverse=True)

for node in node_list:
    index = node['index']
    parent_index = node['parent_index']
    move = node['gtp_move']
    visits = node['visits']
    level = node['level']
    winrate = flip_maybe(node['mean_value'], level)
    raw_winrate = flip_maybe(node['value'], level)
    node_color = get_color(winrate)
    border_color = get_color(raw_winrate)
    text_color = 'black' if abs(winrate - 0.5) < 0.25 else 'white'
    shape = 'square' if odd(level) else 'circle'
    wr = int(winrate * 100)
    raw_wr = int(raw_winrate * 100)
    label = f"{move}\n{wr}%" if visits < 10 else f"{move}\n{wr}% (raw {raw_wr}%)\n{visits} visits"
    dot.node(
        str(index),
        label=label,
        color=border_color,
        fillcolor=node_color,
        fontcolor=text_color,
        style='filled',
        penwidth='5.0',
        height=get_size(visits, shape),
        fixedsize='true',
        shape=shape,
    )

    penwidth = max(0.5, node['policy'] * 10)
    dot.edge(str(parent_index), str(index), penwidth=f"{penwidth}")

dot.render(output_image_path, format='png', view=False, cleanup=True)
