# TODO : better distance measurement (hausdorf)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from deepberry.src.openalea.deepberry.detection_and_segmentation import ellipse_interpolation

PALETTE = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [204, 121, 167], [0, 158, 115],
     [0, 114, 178], [230, 159, 0], [140, 86, 75], [0, 255, 255], [255, 0, 100], [0, 77, 0], [100, 0, 255],
     [100, 0, 0], [0, 0, 100], [100,100, 0], [0, 100,100], [100, 0, 100], [255, 100, 100]])
PALETTE = np.array(20 * list(PALETTE) + [[0, 0, 0]])

# G.add_node('1_1', demand=0)
# G.add_edge('1_1', '2_1', weight=2, capacity=1)  # weight, capacity = attribute names recognised by some algorithms
# G.add_edge('1_1', '2_2')
# G.add_edge('2_2', '3_1')
#
# nx.draw(G, with_labels=True, font_weight='bold')

PATH = 'data/grapevine/results/'

df = pd.read_csv(PATH + 'full_results.csv')
index = pd.read_csv('data/grapevine/image_index.csv')

# plantid, angle = 7236, 120
plantid, angle = 7243, 120
exp = 'DYN2020-05-15'

selec = df[(df['exp'] == exp) & (df['plantid'] == plantid) & (df['angle'] == angle)]

# for loading images
selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

all_d = []

selec = selec[~selec['task'].isin([2377, 2379, 2494, 2554])]

# pb 2380, 2382, 2384, 2385
selec = selec[~selec['task'].isin([2380, 2382, 2384])]

# 2520 -> 2527 legere rotation, donc des baies disparaissent (raparaissent 3-4 frames plus tard)

THRESHOLD = 30
C_ex = 0

C_en = 0

N_TASK = 10
N_BERRY = 999

selec['berry_id'] = -1

tasks = list(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
# tasks = tasks[7:40]

G = nx.DiGraph()  # directed graph
G.add_edge('t', 's', weight=0, capacity=99999)

for i_task, task in enumerate(tasks[:N_TASK]):

    # i = 0
    # i_task, task = i, tasks[i]

    s1 = selec[selec['task'] == tasks[i_task]].iloc[:N_BERRY]
    s2 = selec[selec['task'] == tasks[i_task + 1]].iloc[:N_BERRY]

    # distance matrix
    D = np.zeros((len(s1), len(s2)))
    centers1, centers2 = np.array(s1[['ell_x', 'ell_y']]), np.array(s2[['ell_x', 'ell_y']])
    for i, c1 in enumerate(centers1):
        D[i] = np.sqrt(np.sum((c1 - centers2) ** 2, axis=1))

    if i_task == 0:
        for i in range(len(s1)):
            G.add_edge('s', '{}_{}_a'.format(i_task, i), weight=C_en, capacity=1)
            G.add_edge('{}_{}_a'.format(i_task, i), '{}_{}_b'.format(i_task, i), weight=0, capacity=1)
            G.add_edge('{}_{}_b'.format(i_task, i), 't', weight=C_ex, capacity=1)

    for i in range(len(s2)):
        G.add_edge('s', '{}_{}_a'.format(i_task + 1, i), weight=C_en, capacity=1)
        G.add_edge('{}_{}_a'.format(i_task + 1, i), '{}_{}_b'.format(i_task + 1, i), weight=0, capacity=1)
        G.add_edge('{}_{}_b'.format(i_task + 1, i), 't', weight=C_ex, capacity=1)

    for i1, i2 in np.argwhere(D < THRESHOLD):
        G.add_edge('{}_{}_b'.format(i_task, i1), '{}_{}_a'.format(i_task + 1, i2), weight=int(D[i1, i2] - THRESHOLD), capacity=1)


res = nx.min_cost_flow(G)

# # flow cost
# cost = 0
# for node1 in res.keys():
#     for node2 in res[node1].keys():
#         if res[node1][node2]:
#             cost += G.get_edge_data(node1, node2)['weight']

# trajectories
flow = []
for node1 in res.keys():
    for node2 in res[node1].keys():
        if res[node1][node2] == 1 and node1 not in ['s', 't'] and node2 not in ['s', 't']:
            flow.append([node1, node2])
trajs = []
while flow:
    node = flow[0][0]
    traj = [node]
    neighbors = [n for edge in flow if node in edge for n in edge if n not in traj]
    while neighbors:
        traj = traj + neighbors
        new_neighbors = []
        for node in neighbors:
            new_neighbors += [n for edge in flow if node in edge for n in edge if n not in traj]
        neighbors = new_neighbors
    trajs.append(traj)
    flow = [f for f in flow if not(f[0] in [k for t in trajs for k in t] or f[1] in [k for t in trajs for k in t])]  # update flow
trajs = [[t for t in traj if 'b' not in t] for traj in trajs] # remove a/b redundancy

print('{} trajs'.format(len(trajs)))

# assign berry_id in dataframe
for i_traj, traj in enumerate(trajs):
    for node in traj:
        i_task, i, _ = node.split('_')
        row = selec[selec['task'] == tasks[int(i_task)]].iloc[int(i)]
        selec.loc[row.name, 'berry_id'] = i_traj

# visu berry colors
for i_task, task in enumerate(tasks[:N_TASK]):
    s = selec[selec['task'] == task].iloc[:N_BERRY]
    # s = s[s['berry_id'] != -1]
    plt.figure(i_task)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(min(selec['ell_x']), max(selec['ell_x']))
    plt.ylim(min(selec['ell_y']), max(selec['ell_y']))
    for _, row in s.iterrows():
        x, y, w, h, a = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=30)
        plt.plot(lsp_x, lsp_y, '-', color=PALETTE[row['berry_id']] / 255.)


# visu trajs
for i, traj in enumerate(trajs):
    plt.plot([int(traj[0].split('_')[0]), int(traj[-1].split('_')[0])], [i, i],  'k.-', linewidth=0.5)

# for plot
node_xy = {}
for node in G.nodes:
    if node == 's':
        node_xy[node] = -1, -1
    elif node == 't':
        tmax = max([int(n.split('_')[0]) for n in G.nodes if n not in ['s', 't']])
        node_xy[node] = tmax + 1, -1
    else:
        t, i, type = node.split('_')
        if type == 'a':
            node_xy[node] = int(t) - 0.05, int(i)
        else:
            node_xy[node] = int(t) + 0.05, int(i)

# for node in G.nodes:
#     x, y = node_xy[node]
#     plt.plot(x, y, 'ko')
# for node1, node2 in G.edges:
#     (x1, y1), (x2, y2) = node_xy[node1], node_xy[node2]
#     plt.plot([x1, x2], [y1, y2], 'k-')

plt.figure()
for node1 in res.keys():
    for node2 in res[node1].keys():

        (x1, y1), (x2, y2) = node_xy[node1], node_xy[node2]
        if res[node1][node2] == 1:
            col = 'grey' if node1 in ['s', 't'] or node2 in ['s', 't'] else 'red'
            plt.plot([x1, x2], [y1, y2], '-', color=col)
        else:
            pass
            #plt.plot([x1, x2], [y1, y2], 'k--')



print(set([res[k][k2] for k in res.keys() for k2 in res[k].keys()]))







