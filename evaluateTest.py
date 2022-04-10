import numpy as np
import sys


def hit(rank, ground_truth):
    # HR is equal to Recall when dataset is loo split.
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0
    return result


def precision(rank, ground_truth):
    # Precision is meaningless when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / np.arange(1, len(rank) + 1)
    return result


def recall(rank, ground_truth):
    # Recall is equal to HR when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result


def map(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    # relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    relevant_num = np.cumsum([min(idx + 1, len(ground_truth)) for idx, _ in enumerate(rank)])
    result = [p / r_num if r_num != 0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg_len = min(len(ground_truth), len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg / idcg
    return result


def mrr(rank, ground_truth):
    # MRR is equal to MAP when dataset is loo split.
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0 / (last_idx + 1)
    return result


def top_k_eval(ranks, ground_truths, k):
    hit_k_list = []
    precision_k_list = []
    recall_k_list = []
    map_k_list = []
    ndcg_k_list = []
    mrr_k_list = []
    ranks_k = [rank[:k] for rank in ranks]
    for i in range(0, len(ranks)):
        rank_i = ranks_k[i]
        hit_i_k = hit(rank_i, ground_truths[i])[-1]
        precision_i_k = precision(rank_i, ground_truths[i])[-1]
        recall_i_k = recall(rank_i, ground_truths[i])[-1]
        map_i_k = map(rank_i, ground_truths[i])[-1]
        ndcg_i_k = ndcg(rank_i, ground_truths[i])[-1]
        mrr_i_k = mrr(rank_i, ground_truths[i])[-1]
        hit_k_list.append(hit_i_k)
        precision_k_list.append(precision_i_k)
        recall_k_list.append(recall_i_k)
        map_k_list.append(map_i_k)
        ndcg_k_list.append(ndcg_i_k)
        mrr_k_list.append(mrr_i_k)
    hit_k = np.round(np.average(np.array(hit_k_list)), 4)
    precision_k = np.round(np.average(np.array(precision_k_list)), 4)
    recall_k = np.round(np.average(np.array(recall_k_list)), 4)
    map_k = np.round(np.average(np.array(map_k_list)), 4)
    ndcg_k = np.round(np.average(np.array(ndcg_k_list)), 4)
    mrr_k = np.round(np.average(np.array(mrr_k_list)), 4)
    return hit_k, precision_k, recall_k, map_k, ndcg_k, mrr_k


if __name__ == '__main__':
    ranks = [[5, 7, 8, 9, 3], [4, 6, 2, 1, 10]]
    ground_truths = [[7, 3, 5], [4, 2, 8, 7]]
    k = 3
    hit_k, precision_k, recall_k, map_k, ndcg_k, mrr_k = top_k_eval(ranks, ground_truths, k=k)
    print('hit_rate@%d:' % k, hit_k)
    print('precision@%d:' % k, precision_k)
    print('recall@%d:' % k, recall_k)
    print('map@%d:' % k, map_k)
    print('ndcg@%d:' % k, ndcg_k)
    print('mrr@%d:' % k, mrr_k)