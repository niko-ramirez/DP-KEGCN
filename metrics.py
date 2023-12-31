import tensorflow as tf
import numpy as np
import scipy.spatial
from scipy.sparse import isspmatrix
from tensorflow.compat.v1.losses import Reduction

def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name+":0")
    except:
        return tf.placeholder(tf.int32, name=name)


def align_loss(embeddings, align_labels, gamma, num_negs, names_neg, mode="L1"):
    def loss_metric(X, Y):
        if mode == "L1":
            loss = tf.reduce_sum(tf.abs(X - Y), 1)
        elif mode == "sim":
            loss = tf.nn.sigmoid(-tf.reduce_sum(X * Y, 1))
        else:
            exit("wrong loss mode")
        return loss
    def get_ranking_loss(names, num_labels):
        neg_left = get_placeholder_by_name(names[0])
        neg_right = get_placeholder_by_name(names[1])
        neg_l_x = tf.nn.embedding_lookup(embeddings, neg_left)
        neg_r_x = tf.nn.embedding_lookup(embeddings, neg_right)
        neg_value = loss_metric(neg_l_x, neg_r_x)
        neg_value = - tf.reshape(neg_value, [num_labels, num_negs])
        loss_value = neg_value + tf.reshape(pos_value, [num_labels, 1])
        loss_value = tf.nn.relu(loss_value)
        return loss_value

    left_labels = align_labels[:, 0]
    right_labels = align_labels[:, 1]
    num_labels = len(align_labels)
    left_x = tf.nn.embedding_lookup(embeddings, left_labels)
    right_x = tf.nn.embedding_lookup(embeddings, right_labels)
    pos_value = loss_metric(left_x, right_x) + gamma

    loss_1 = get_ranking_loss(names_neg[:2], num_labels)
    loss_2 = get_ranking_loss(names_neg[2:], num_labels)

    final_loss = tf.reduce_sum(loss_1) + tf.reduce_sum(loss_2)
    final_loss /= (2.0 * num_negs * num_labels)

    return final_loss


# def class_loss(embeddings, test, y):
#     y_pre = tf.nn.embedding_lookup(embeddings, test)
#     if isspmatrix(y):
#         y_test = y[test].todense()
#         y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
#     else:
#         y_test = y[test]
#         y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
#     # loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_test, y_pre, reduction=Reduction.NONE)
#     loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_test, logits=y_pre)
#     return tf.reduce_mean(loss)

def class_loss(embeddings, test, y):
    y_pre = tf.nn.embedding_lookup(embeddings, test)
    if isspmatrix(y):
        y_test = y[test].todense()
        y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    else:
        y_test = y[test]
        y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    # loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_test, y_pre, reduction=Reduction.NONE)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_test, logits=y_pre)
    cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)
    loss = cce(y_test, y_pre)
    return loss
    # return tf.reduce_mean(loss)

def label_loss(embeddings, test, y):
    y_pre = tf.nn.embedding_lookup(embeddings, test)
    if isspmatrix(y):
        y_test = y[test].todense()
    else:
        y_test = y[test]
    # loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_test, y_pre, reduction=Reduction.NONE)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test, logits=y_pre)
    # return tf.reduce_mean(loss)
    # cce = tf.keras.losses.CategoricalCrossentropy(
    #         from_logits=True, reduction=tf.losses.Reduction.NONE)
    # loss = cce(y_test, y_pre)
    return loss

def get_class(embeddings, test, y, logging):
    y_pre = embeddings[test]
    if isspmatrix(y):
        y_test = y[test].todense()
        y_true = np.argmax(y_test, 1).reshape(1,-1)[0]
    else:
        y_test = y[test]
        y_true = np.argmax(y_test, 1).reshape(1,-1)
    y_pre = np.argmax(y_pre, 1).reshape(1,-1)
    # print(np.concatenate([y_true, y_pre]))
    correct_prediction = np.equal(y_pre, y_true)
    acc = np.mean(correct_prediction)
    return acc, [acc]

def link_loss(embeddings, test, y, adj, negative_adj, names_neg):
    bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)

    # print(test)
    new_scores = []
    labels = []
    for index in test:
        if index >= 0:
            new_scores.append(tf.norm(embeddings[adj[index][0]] + embeddings[adj[index][1]] - embeddings[adj[index][2]]))
            labels.append(1)
        else:
            new_scores.append(tf.norm(embeddings[negative_adj[index][0]] + embeddings[negative_adj[index][1]] - embeddings[negative_adj[index][2]]))
            labels.append(0)

    print(len(new_scores))
    print(len(labels))
    return bce(labels, new_scores)


# def link_loss(embeddings, test, y, adj, negative_adj, names_neg):
#     bce = tf.keras.losses.BinaryCrossentropy(
#             from_logits=True, reduction=tf.losses.Reduction.NONE)

#     # print(test)
#     new_scores = []
#     labels = []
#     for index in test:
#         if index >= 0:
#             new_scores.append(tf.norm(embeddings[adj[index][0]] + embeddings[adj[index][1]] - embeddings[adj[index][2]]))
#             labels.append(1)
#         else:
#             new_scores.append(tf.norm(embeddings[negative_adj[index][0]] + embeddings[negative_adj[index][1]] - embeddings[negative_adj[index][2]]))
#             labels.append(0)


#     return bce(labels, new_scores)
    # else:
    #     score = tf.norm(embeddings[negative_adj[test][0]] + embeddings[negative_adj[test][1]] - embeddings[negative_adj[test][2]])
    #     return bce(tf.constant([0]), [score])



    # y_pre = tf.nn.embedding_lookup(embeddings, test)
    # if isspmatrix(y):
    #     y_test = y[test].todense()
    #     y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    # else:
    #     y_test = y[test]
    #     y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    # bce = tf.keras.losses.BinaryCrossentropy(
    #         from_logits=True, reduction=tf.losses.Reduction.NONE)
    # loss = bce(y_test, y_pre)
    # return loss

def get_link(embeddings, test, negatives, adj, logging, K=(1, 5, 10)):

    embeddings_true = [np.linalg.norm(embeddings[adj[index][0]] + embeddings[adj[index][1]] - embeddings[adj[index][2]]) for index in test]

    embeddings_false = [np.linalg.norm(embeddings[h]+ embeddings[r] - embeddings[t]) for h,r,t in negatives]

    total_embeddings = [(score, True) for score in embeddings_true] + [(score, False) for score in embeddings_false]
    total_embeddings.sort(key=lambda x: x[0], reverse=True)

    mrr = 0
    top_hits = [0] * len(K)
    for rank, tuple_value in enumerate(total_embeddings):
        for j in range(len(K)):
            if j >= rank+1:
                top_hits[j] += 1 
        is_true = tuple_value[1]
        if is_true:
            mrr += 1.0/(float(rank+1))

    test_metric = ["Hits@{}".format(K[i]) for i in range(len(K))]
    test_metric = " ".join(test_metric)
    result = [top_hits[i] / float(len(test)) * 100 for i in range(len(K))]
    result = [str(round(i, 5)) for i in result]
    logging.info(test_metric)
    logging.info(result)
    logging.info('MRR: %.5f' % (mrr / float(len(test))))

def get_label(embeddings, test, y, logging):
    y_pre = embeddings[test]
    if isspmatrix(y):
        y_test = y.todense()[test]
        y_test = np.squeeze(np.asarray(y_test))
    else:
        y_test = y[test]
    y_pre = np.argsort(-y_pre, 1)
    result_list = []

    for K in [1,3,5]:
        precision = 0
        NDCG = 0
        y_pre_K = y_pre[:, :K]
        coeff = 1./np.log(np.arange(1,K+1) + 1)
        for i,each in enumerate(y_pre_K):
            if np.sum(y_test[i]) <= 0:
                continue
            precision += np.sum(y_test[i, each])/K
            DCG_i = np.sum(y_test[i, each]*coeff)
            norm = np.sum(1./np.log(np.arange(1,min(K, np.sum(y_test[i]))+1) + 1))
            NDCG_i = DCG_i/norm
            NDCG += NDCG_i
        precision = precision/len(y_pre_K)
        NDCG = NDCG/len(y_pre_K)
        logging.info("Classification Precision %d: %.3f" % (K, precision * 100))
        logging.info("Classification NDCG %d: %.3f" % (K, NDCG * 100))
        result_list.append(round(precision, 4))
        result_list.append(round(NDCG, 4))
    return result_list[4], result_list


def get_align(embeddings, test_pair, logging, metric="cityblock", K=(1, 5, 10, 50, 100)):
    def get_metrics(sim, pos=0):
        top_hits = [0] * len(K)
        mrr = 0
        for ind in range(sim.shape[pos]):
            rank_ind = np.where(sim[(slice(None),) * pos + (ind,)].argsort() == ind)[0][0]
            mrr += 1/(rank_ind+1)
            for j in range(len(K)):
                if rank_ind < K[j]:
                    top_hits[j] += 1
        return top_hits, mrr

    embeddings_left = np.array([embeddings[e1] for e1, _ in test_pair])
    embeddings_right = np.array([embeddings[e2] for _, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(embeddings_left, embeddings_right, metric=metric)

    top_hits_l2r, mrr_l2r = get_metrics(sim, pos=0)
    top_hits_r2l, mrr_r2l = get_metrics(sim, pos=1)

    test_metric = ["Hits@{}".format(K[i]) for i in range(len(K))]
    test_metric = " ".join(test_metric)
    left_result = [top_hits_l2r[i] / len(test_pair) * 100 for i in range(len(K))]
    right_result = [top_hits_r2l[i] / len(test_pair) * 100 for i in range(len(K))]
    all_result = [(left_result[i] + right_result[i])/2 for i in range(len(right_result))]
    left_result = [str(round(i, 3)) for i in left_result]
    right_result = [str(round(i, 3)) for i in right_result]
    all_result = [str(round(i, 3)) for i in all_result]
    logging.info(test_metric)
    logging.info("l:\t" + "\t".join(left_result))
    logging.info("r:\t" + "\t".join(right_result))
    logging.info("a:\t" + "\t".join(all_result))
    logging.info('MRR-l: %.3f' % (mrr_l2r / len(test_pair)))
    logging.info('MRR-r: %.3f' % (mrr_r2l / len(test_pair)))
    logging.info('MRR-a: %.3f' % ((mrr_l2r+mrr_r2l)/2 / len(test_pair)))
