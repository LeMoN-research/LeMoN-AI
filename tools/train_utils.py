import numpy as np
from processing import positions_to_moves


def get_data(npy_path):
    data_matrix = np.load(npy_path)
    return np.array([np.array([y[0] for y in x]) for x in data_matrix]),\
        np.array([np.array([y[1] for y in x]) for x in data_matrix])


def iterate_minibatches(points, batch_size=100, block_size=20):
    blocks_left = [int(p.shape[0]/block_size) for p in points]
    blocks_total = blocks_left
    moves = np.array([positions_to_moves(x) for x in points])
    batch_count = 0
    train_moves = []
    pred_move = []
    start_pos = []
    out_spec = []

    while np.any(blocks_left != 0):
        inst = np.random.choice(range(len(blocks_left)))
        start = -blocks_left[inst]*block_size
        fin = start+block_size            
        
        if fin < moves.shape[0]:
            train_moves.append(moves[inst][start:fin-1])
            pred_move.append(moves[inst][fin])
            start_pos.append(points[inst][start])

            our_spec = np.load("spectr/{}.{}.spec".format(inst, blocks_total[inst]-blocks_left[inst]))
            out_spec.append(our_spec)

        blocks_left[inst] -= 1

        if len(train_moves) == batch_size:
            yield np.array(out_spec), np.array(train_moves), np.array(start_pos), np.array(pred_move)
            train_moves = []
            pred_move = []
            start_pos = []
            out_spec = []
