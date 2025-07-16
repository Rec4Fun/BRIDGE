import pickle

import jax.lax
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import list2graph, get_pairs

INF = 1e8


def eval(all_gen_buns_batch, uids_test_batch, ub_mask_graph_batch, 
         ub_mat, bi_mat, 
         item_feat, topk, alpha):
    
    recall_cnt, pre_cnt, ndcg_cnt, cnt = 0, 0, 0, 0
    gen_bun_mat_batch = np.zeros((len(all_gen_buns_batch), ni + 3)) # +3 due to 3 extra tokens
    for bun in all_gen_buns_batch:
        for iid in bun:
            gen_bun_mat_batch[cnt][iid+3] = 1 # iid+3 due to while generating i -3 in every token to match the item id
        cnt+=1
    gen_bun_mat_batch = gen_bun_mat_batch[:, 3:]
    jaccard_score = cal_jaccard(gen_bun_mat_batch, bi_mat)

    gen_bun_feat = gen_bun_mat_batch @ item_feat / (gen_bun_mat_batch.sum(axis=1).reshape(-1, 1) + 1e-8)
    pre_bun_feat = bi_mat @ item_feat / (bi_mat.sum(axis=1).reshape(-1, 1) + 1e-8)
    cosine_score = cal_cosine(gen_bun_feat, pre_bun_feat)

    com_score = jaccard_score * alpha + cosine_score * (1-alpha)
    
    for i in range(len(all_gen_buns_batch)):
        temp_cnt = 0
        uid = uids_test_batch[i]
        score = com_score[i] + (ub_mask_graph_batch[i] * -INF)

        _, most_sim_ids = jax.lax.top_k(score, k=topk)  # recommend top-k most similar with fake bundle
        real_buns_id = ub_mat[uid].nonzero()[0] # for dense matrix if sprase matrix index must be 1
        # print(uid, real_buns_id)
        num_pos = len(real_buns_id)
        real_buns = []
        for bun_id in real_buns_id:
            real_buns.append(bi_mat[bun_id].nonzero()[0])

        is_hit = np.in1d(most_sim_ids, real_buns_id)
        temp_cnt += sum(is_hit) # Recall
        dcg = is_hit / jnp.log2(jnp.arange(2, topk+2)) # NDCG

        if num_pos >= topk:
            temp_hit = jnp.ones(topk)
            idcg = temp_hit / jnp.log2(jnp.arange(2, topk+2))
        else:
            temp_hit = np.concatenate([jnp.ones(num_pos), jnp.zeros(topk-num_pos)])
            idcg = temp_hit / jnp.log2(jnp.arange(2, topk+2))


        ndcg_cnt += dcg.sum() / idcg.sum()
        recall_cnt += temp_cnt / len(real_buns_id)
        pre_cnt += temp_cnt / topk
    return recall_cnt, pre_cnt, ndcg_cnt


def cal_jaccard(mat1, mat2):
    intersect = mat1 @ mat2.T
    total1 = mat1.sum(axis=1)
    total2 = mat2.sum(axis=1)
    total = total1.reshape(-1, 1) + total2.reshape(1, -1) - intersect
    jaccard_score = intersect / (total + 1e-8)
    return jaccard_score


def cal_cosine(mat1, mat2):
    m, n = mat1.shape[0], mat2.shape[0]
    nume = mat1 @ mat2.T
    norm1 = jnp.linalg.norm(mat1, axis=1).reshape(-1, 1) #[m ,1]
    norm2 = jnp.linalg.norm(mat2, axis=1).reshape(1, -1) #[1, n]
    norm1 = jnp.broadcast_to(norm1, (m, n))
    norm2 = jnp.broadcast_to(norm2, (m, n))
    denom =  norm1 * norm2 + 1e-8
    cosine = nume / denom
    return cosine


if __name__ == "__main__":

    argp = ArgumentParser()
    argp.add_argument("--dataset", type=str, default="clothing")
    argp.add_argument("--topk", type=int, default=1)
    argp.add_argument("--batch_size", type=int, default=256)
    argp.add_argument("--task", type=str, default="test")
    # argp.add_argument("--type", type=str, default=str)
    argp.add_argument("--alpha", type=float, default=1)
    args = argp.parse_args()

    # ev_type = args.type
    dataset = args.dataset
    task = args.task
    batch_size = args.batch_size
    topk = args.topk
    alpha = args.alpha

    # assert (ev_type == "jaccard" or ev_type == "cosine"), "invalid eval type"

    # lazy-load
    if dataset == "clothing":
        nu, nb, ni = 965, 1910, 4487
    elif dataset == "electronic":
        nu, nb, ni = 888, 1750, 3499
    elif dataset == "food":
        nu, nb, ni = 879, 1784, 3767
    elif dataset == "Steam":
        nu, nb, ni = 29634, 615, 2819
    elif dataset == "iFashion":
        nu, nb, ni = 53897, 27694, 42563
    elif dataset == "NetEase":
        nu, nb, ni = 18528, 22864, 123628
    elif dataset == "meal":
        nu, nb, ni = 1575, 3817, 7280
    else:
        raise ValueError("Invalid datasets")

    item_feat = jnp.load(f"data/{dataset}/item_feat.npy")
    # item_feat = np.random.randn(ni, 16)

    with open(f"data/{dataset}/generated_bundles_list_{task}.pkl", "rb") as f: # generated bundles for user in test
        all_gen_buns = pickle.load(f)

    bi_pairs = get_pairs(f"data/{dataset}/bundle_item.txt")
    bi_mat = list2graph(bi_pairs, (nb, ni))

    ub_mask_pair = get_pairs(f"data/{dataset}/user_bundle_train.txt")
    ub_mask_graph = list2graph(ub_mask_pair, (nu, nb))

    ub_pairs = get_pairs(f"data/{dataset}/user_bundle_{task}.txt")
    ub_mat = list2graph(ub_pairs, (nu, nb)) # all dense graph


    n_gen_buns = len(all_gen_buns)
    uids_test = ub_pairs[:, 0]
    uids_test = np.unique(uids_test)
    uids_test.sort()
    num_batch = int(len(uids_test) / batch_size)
    batch_idx = np.arange(0, len(uids_test))
    test_batch_loader = DataLoader(batch_idx, batch_size=batch_size, shuffle=False, drop_last=False)
    

    recall_cnt = 0
    pre_cnt = 0
    ndcg_cnt = 0


    for batch in test_batch_loader:
        # print(batch)
        start = batch[0]
        end = batch[-1]
        # print(start, end)
        all_gen_buns_batch = all_gen_buns[start:end+1] # -> get data from start to end
        uids_test_batch = uids_test[start:end+1]
        ub_mask_graph_batch = ub_mask_graph[uids_test_batch]
        
        r_cnt, p_cnt, n_cnt = eval(all_gen_buns_batch, uids_test_batch, 
                                   ub_mask_graph_batch, ub_mat, bi_mat, 
                                   item_feat, topk, alpha)
        recall_cnt+=r_cnt
        pre_cnt+=p_cnt
        ndcg_cnt+=n_cnt


    print(f"Recall@{topk}: {recall_cnt / len(uids_test)}")
    print(f"Precision@{topk}: {pre_cnt / len(uids_test)}")
    print(f"NDCG@{topk}: {ndcg_cnt / len(uids_test)} ")
