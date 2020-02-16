import numpy as np
def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    #print(gain)
    discounts = np.log2(np.arange(len(y_true)) + 2)
    #print(discounts)
    return np.sum(gain / discounts)

def ndcg_score(y_true,y_score,k=5):
    actual_score=dcg_score(y_true,y_score,k)
    max_score=dcg_score(y_true,y_true,k)
    return actual_score/max_score