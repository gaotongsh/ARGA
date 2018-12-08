import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset):
    filename = 'data/facebook/' + dataset

    g = np.loadtxt(filename + '.edges')
    graph = nx.Graph()
    for i in g:
        graph.add_edge(int(i[0]), int(i[1]))
    adj = nx.adjacency_matrix(graph)
    node_list = list(graph.nodes)
    i = 0
    node_map = {}
    for n in node_list:
        node_map[int(n)] = int(i)
        i += 1

    node_num = len(graph.nodes)

    f = np.loadtxt(filename + '.feat')
    feature_number = f.shape[1] - 1
    features = np.ndarray((node_num, feature_number))
    for i in f:
        if int(i[0]) in node_map:
            features[node_map[int(i[0])]] = i[1:]

    features = sp.lil_matrix(features)

    y_dict = {}
    prior = []
    with open(filename + '.circles') as f:
        lines = f.readlines()
        i = 0
        for l in lines:
            words = l.split()
            prior.append(len(words) - 1)
            for w in words[1:]:
                if int(w) in node_map:
                    if not int(w) in y_dict:
                        y_dict[int(w)] = [i]
                    else:
                        y_dict[int(w)].append(i)
            i += 1

    ordered_y = []
    for k, v in node_map.items():
        if k not in y_dict:
            ordered_y.append([-1])
        else:
            ordered_y.append(y_dict[k])

    for v in ordered_y:
        if v[0] != -1:
            v.sort(key=lambda k: prior[k], reverse=True)

    print(ordered_y)

    with open(dataset + '.n.labels.pkl', 'wb') as f:
        pkl.dump(ordered_y, f)

    exit()

    # with open('label.txt', 'w') as f:
    #     for v in ordered_y:
    #         for l in v:
    #             f.write(str(l) + ' ')
    #         f.write('\n')

    return adj, features, None, None, None, None, None

    # Previous result
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_test, tx, ty, test_mask, np.argmax(labels,1)


def load_alldata(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, np.argmax(labels, 1)
