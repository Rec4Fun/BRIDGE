import jax.lax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from config import *
from torch.utils.data import Dataset, DataLoader

import os
import scipy.sparse as sp
from jax.experimental import sparse


def get_pairs(file_path):
    xy = pd.read_csv(file_path, sep="\t", names=["x", "y"])
    xy = xy.to_numpy()
    return xy


def get_size(file_path):
    nu, nb, ni = pd.read_csv(file_path, sep="\t", names=["u", "b", "i"]).to_numpy()[0]
    return nu, nb, ni


def list2jax_sp_graph(list_index, shape):
    values = np.ones(list_index.shape[0])
    sp_graph = sp.coo_matrix(
        (values, (list_index[:, 0], list_index[:, 1])),
        shape=shape
    )
    jax_sp_graph = sparse.BCOO.from_scipy_sparse(sp_graph)
    return jax_sp_graph


def list2graph(list_index, shape):
    graph = np.zeros(shape)
    for i in list_index:
        graph[i[0], i[1]] = 1
        # important
        # graph[i[0], i[1]] += 1 # load repeat or not
    return graph


def list2csr_sp_graph(list_index, shape):
    """
    list indices to scipy.sparse csr
    """
    sp_graph = sp.coo_matrix(
        (np.ones(list_index.shape[0]), (list_index[:, 0], list_index[:, 1])),
        shape=shape
    ).tocsr()
    return sp_graph


def graph2list(graph):
    idx = np.stack(graph.nonzero(), axis=0)
    idx = idx.T  # [[row, col], ...]
    return idx


def jax_sp_graph2list(graph):
    idx = graph.indices
    idx = idx.T
    return idx

def csr_sp_graph2list(graph):
    graph = graph.tocoo()
    indices = np.array([graph.row, graph.col]).T
    return indices


def make_sp_diag_mat(n):
    ids = np.arange(0, n)
    vals = np.ones(n, dtype=float)
    diag_mat = sp.coo_matrix(
        (vals, (ids, ids)),
        shape=(n, n)
    )
    jax_sp_diag_mat =  sparse.BCOO.from_scipy_sparse(diag_mat)
    return jax_sp_diag_mat


class TestDataVer2():
    """
    this data class return item index = real index + 3
    """

    def __init__(self, conf, task="test"):
        super().__init__()
        self.conf = conf
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]
        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_{task}.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")
        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))
        self.test_uid = self.ub_graph.sum(axis=1).nonzero()[0]  # only test for user in test set
        # n_bundle_test = self.bi_graph.sum(axis=1).nonzero()[0]
        # print(f"number of testing bundle: {len(n_bundle_test)}")
        print(f"number of testing users: {len(self.test_uid)}")

    def __getitem__(self, index):
        """
        return all test bundle of 1 user id
        """
        uid = self.test_uid[index]
        print("user id:", uid)
        bid = self.ub_graph[uid].nonzero()[1]
        bundles = []
        for bun_id in bid:
            bundles.append(self.bi_graph[bun_id].nonzero()[1]) # for test +0
        # for bun in bundles:
        #     bun += 3  # plus 3 pad sos eos !!! important
        item_inp = self.ui_graph[uid].nonzero()[1]
        iids = np.array(item_inp)
        iids += 3 # generator input +3 due to 3 extra tokens

        if len(iids) > self.conf["seq_len"]:
            iids = iids[:self.conf["seq_len"]]
        elif len(iids) < self.conf["seq_len"]:
            pad_len = self.conf["seq_len"] - len(iids)
            pads = pad_len * [self.conf["pad"]]
            iids = np.concatenate([iids, pads])
        else:
            pass
        bundles = []
        return iids, bundles

    def __len__(self):
        return len(self.test_uid)

class TestDataVer3():
    """
    this data class return item index = real index + 3
    """

    def __init__(self, conf, task="test"):
        super().__init__()
        self.conf = conf
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]
        
        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_{task}.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")
        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))

        self.ub_mask_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
        self.ub_mask_graph = list2csr_sp_graph(self.ub_mask_pairs, (self.num_user, self.num_bundle))
        self.test_uid = self.ub_graph.sum(axis=1).nonzero()[0]
        self.uibi_graph = self.ub_mask_graph @ self.bi_graph
        print(f"number of testing users: {len(self.test_uid)}")

    def __getitem__(self, index):
        """
        return all test bundle of 1 user id
        """
        uid = self.test_uid[index]
        print("user id:", uid)
        bid = self.ub_graph[uid].nonzero()[1]
        bundles = []
        for bun_id in bid:
            bundles.append(self.bi_graph[bun_id].nonzero()[1]) # for test +0
        # for bun in bundles:
        #     bun += 3  # plus 3 pad sos eos !!! important
        item_inp = self.uibi_graph[uid].nonzero()[1]
        iids = np.array(item_inp)
        iids.sort()
        iids += 3 # generator input +3 due to 3 extra tokens

        if len(iids) > self.conf["seq_len"]:
            iids = iids[:self.conf["seq_len"]]
        elif len(iids) < self.conf["seq_len"]:
            pad_len = self.conf["seq_len"] - len(iids)
            pads = pad_len * [self.conf["pad"]]
            iids = np.concatenate([iids, pads])
        else:
            pass
        bundles = []
        # print(uid)
        # print(iids - 3)
        # exit()
        return iids, bundles

    def __len__(self):
        return len(self.test_uid)
    

class TrainDataVer9(Dataset):
    """
    this data class return item id = real item id + 3
    if user in train not interacted with any bundles -> try pseudo bundles
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # we use bundle id to easily link bundle to user for train and test purpose 
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]
        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")
        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))
        # self.item_pretrain_feat = np.load(os.path.join(conf['data_path'], conf['dataset'], "item_feat.npy"))
        # self.ii_score = item_pretrain_feat @ item_pretrain_feat.T # it must propagate to I-Igraph
        self.iui_mat = self.ui_graph.T @ self.ui_graph
        # self.ii_score = self.ii_score - np.diag(np.diag(self.ii_score))
        self.iui_graph = (self.ui_graph.T @ self.ui_graph)
        self.iui_pairs = csr_sp_graph2list(self.iui_graph)
        self.item_feat = None # this change over time
        self.ii_score = None
        self.uibi_graph = self.ui_graph + self.ub_graph @ self.bi_graph

    def __getitem__(self, index):
        ## LOAD TRAIN DATA FOR TRANSFORMER ##
        # TODO: sync csr type and dense type matrix
        # self.ii_score =  (self.item_feat @ self.item_feat.T) * 0.2 + 0.8 * self.iui_graph
        self.ii_score = self.iui_graph
        u_row = self.uibi_graph[index]
        uid = index
        iids = u_row.nonzero()[1]
        iids.sort()
        bid_train = self.ub_graph[uid]
        """
        this dataload is very slow
        with steam return iids_shuffle would be better
        """
        if self.conf['dataset'] == 'clothing' or self.conf['dataset'] == 'food' or self.conf['dataset'] == 'electronic':
            if np.random.choice(np.arange(0, 4)) == 0 and bid_train.sum() != 0: # 25% + 75% augmented bundle
                interacted_bun = bid_train.nonzero()[1] # get nonzero columns
                rand_bun = np.random.choice(interacted_bun)
                iids_shuffle = self.bi_graph[rand_bun].nonzero()[1] # get item of that bundle (of bi graph)
            else:
                iids_shuffle = iids.copy()
                random_pseudo_bundle_item_id = np.random.choice(iids_shuffle)
                _, pseudo_bundle = jax.lax.top_k(self.ii_score[random_pseudo_bundle_item_id].todense(), k=iids_shuffle.shape[0])
                iids_shuffle_joint = np.concatenate([pseudo_bundle.copy().squeeze().sort(), iids])
                iids_shuffle = jnp.intersect1d(iids_shuffle_joint, iids_shuffle)
        else:
            iids_shuffle = iids.copy() # normal load is very slow with Steam
        # +=3 due to token sos eos pad
        iids += 3
        iids_shuffle += 3

        if len(iids) > self.conf["seq_len"]:
            iids = iids[:self.conf["seq_len"]]
        elif len(iids) < self.conf["seq_len"]:
            pad_len = self.conf["seq_len"] - len(iids)
            pads = pad_len * [self.conf["pad"]]
            iids = np.concatenate([iids, pads])
        else:
            pass

        if len(iids_shuffle) > self.conf["seq_len"] - 2:
            iids_shuffle = iids_shuffle[:(self.conf["seq_len"] - 2)]
            target = np.concatenate([iids_shuffle, [self.conf["eos"]], [self.conf["pad"]]])
            iids_shuffle = np.concatenate([[self.conf["sos"]], iids_shuffle, [self.conf["eos"]]])
        elif len(iids_shuffle) < self.conf["seq_len"] - 2:
            pad_len = self.conf["seq_len"] - 2 - len(iids_shuffle)
            pads = pad_len * [self.conf["pad"]]
            target = np.concatenate([iids_shuffle, [self.conf["eos"]], [self.conf["pad"]], pads])
            iids_shuffle = np.concatenate([[self.conf["sos"]], iids_shuffle, [self.conf["eos"]], pads])
        else:
            target = np.concatenate([iids_shuffle, [self.conf["eos"]], [self.conf["pad"]]])
            iids_shuffle = np.concatenate([[self.conf["sos"]], iids_shuffle, [self.conf["eos"]]])


        ## LOAD TRAIN DATA FOR ITEM_ITEM MAT ##
        rint = np.random.choice(len(self.iui_pairs))
        piid, ppid = self.iui_pairs[rint]
        while True:
            pnid = np.random.randint(self.num_item)
            if self.iui_graph[piid, pnid] == 0 and pnid != piid:
                break

        # iids: enc inp, iids_shuffle: dec inp
        return [iids, iids_shuffle], target, [piid, ppid, pnid]

    def __len__(self):
        return self.num_user
    

class TrainDataVer10(Dataset):
    """
    this data class return item id = real item id + 3
    if user in train not interacted with any bundles -> try pseudo bundles
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # we use bundle id to easily link bundle to user for train and test purpose 
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]
        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")
        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))
        # self.item_pretrain_feat = np.load(os.path.join(conf['data_path'], conf['dataset'], "item_feat.npy"))
        # self.ii_score = item_pretrain_feat @ item_pretrain_feat.T # it must propagate to I-Igraph
        self.iui_mat = self.ui_graph.T @ self.ui_graph
        # self.ii_score = self.ii_score - np.diag(np.diag(self.ii_score))
        self.iui_graph = (self.ui_graph.T @ self.ui_graph)
        self.iui_pairs = csr_sp_graph2list(self.iui_graph)
        self.item_feat = None # this change over time
        self.ii_score = None
        self.uibi_graph = self.ub_graph @ self.bi_graph

    def __getitem__(self, index):
        ## LOAD TRAIN DATA FOR TRANSFORMER ##
        # TODO: sync csr type and dense type matrix
        # self.ii_score =  (self.item_feat @ self.item_feat.T) * 0.2 + 0.8 * self.iui_graph
        self.ii_score = self.iui_graph
        u_row = self.uibi_graph[index]
        uid = index
        iids = u_row.nonzero()[1]
        iids.sort()
        # print(iids)
        bid_train = self.ub_graph[uid]
        """
        this dataload is very slow
        with steam return iids_shuffle would be better
        """
        if self.conf['dataset'] == 'clothing' or self.conf['dataset'] == 'food' or self.conf['dataset'] == 'electronic':
            if np.random.choice(np.arange(0, 4)) == 0 and bid_train.sum() != 0: # 25% + 75% augmented bundle
                interacted_bun = bid_train.nonzero()[1] # get nonzero columns
                rand_bun = np.random.choice(interacted_bun)
                iids_shuffle = self.bi_graph[rand_bun].nonzero()[1] # get item of that bundle (of bi graph)
            else:
                iids_shuffle = iids.copy()
                random_pseudo_bundle_item_id = np.random.choice(iids_shuffle)
                _, pseudo_bundle = jax.lax.top_k(self.ii_score[random_pseudo_bundle_item_id].todense(), k=iids_shuffle.shape[0])
                iids_shuffle_joint = np.concatenate([pseudo_bundle.copy().squeeze().sort(), iids])
                iids_shuffle = jnp.intersect1d(iids_shuffle_joint, iids_shuffle)
        else:
            iids_shuffle = iids.copy() # normal load is very slow with Steam
        # +=3 due to token sos eos pad
        iids += 3
        iids_shuffle += 3

        if len(iids) > self.conf["seq_len"]:
            iids = iids[:self.conf["seq_len"]]
        elif len(iids) < self.conf["seq_len"]:
            pad_len = self.conf["seq_len"] - len(iids)
            pads = pad_len * [self.conf["pad"]]
            iids = np.concatenate([iids, pads])
        else:
            pass

        if len(iids_shuffle) > self.conf["seq_len"] - 2:
            iids_shuffle = iids_shuffle[:(self.conf["seq_len"] - 2)]
            target = np.concatenate([iids_shuffle, [self.conf["eos"]], [self.conf["pad"]]])
            iids_shuffle = np.concatenate([[self.conf["sos"]], iids_shuffle, [self.conf["eos"]]])
        elif len(iids_shuffle) < self.conf["seq_len"] - 2:
            pad_len = self.conf["seq_len"] - 2 - len(iids_shuffle)
            pads = pad_len * [self.conf["pad"]]
            target = np.concatenate([iids_shuffle, [self.conf["eos"]], [self.conf["pad"]], pads])
            iids_shuffle = np.concatenate([[self.conf["sos"]], iids_shuffle, [self.conf["eos"]], pads])
        else:
            target = np.concatenate([iids_shuffle, [self.conf["eos"]], [self.conf["pad"]]])
            iids_shuffle = np.concatenate([[self.conf["sos"]], iids_shuffle, [self.conf["eos"]]])


        ## LOAD TRAIN DATA FOR ITEM_ITEM MAT ##
        rint = np.random.choice(len(self.iui_pairs))
        piid, ppid = self.iui_pairs[rint]
        while True:
            pnid = np.random.randint(self.num_item)
            if self.iui_graph[piid, pnid] == 0 and pnid != piid:
                break

        # iids: enc inp, iids_shuffle: dec inp
        # print(uid)
        # print(iids, iids_shuffle)
        # exit()
        return [iids, iids_shuffle], target, [piid, ppid, pnid]

    def __len__(self):
        return self.num_user