from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import tsne
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


# 推理，在已经训练好的网络上进行验证
# def inference(loader, model, device, view, data_size, class_num):
#     """
#     :return:
#     total_pred: prediction among all modalities
#     pred_vectors: predictions of each modality, list
#     labels_vector: true label
#     Hs: high-level features
#     Zs: low-level features
#     """
#     model.eval()
#     soft_vector = []
#     pred_vectors = []
#     Hs = []
#     Zs = []
#     for v in range(view):
#         pred_vectors.append([])
#         Hs.append([])
#         Zs.append([])
#     labels_vector = []
#
#     for step, (xs, y, _) in enumerate(loader):
#         for v in range(view):
#             xs[v] = xs[v].to(device)
#         with torch.no_grad():
#             qs, preds = model.forward_cluster(xs)
#             hs, _, zs, _ = model.forward(xs)
#             # q = ds_Fusion.DS_Combin(qs, class_num)
#             q = sum(qs)/view
#         for v in range(view):
#             hs[v] = hs[v].detach()
#             zs[v] = zs[v].detach()
#             preds[v] = preds[v].detach()
#             pred_vectors[v].extend(preds[v].cpu().detach().numpy())
#             Hs[v].extend(hs[v].cpu().detach().numpy())
#             Zs[v].extend(zs[v].cpu().detach().numpy())
#         q = q.detach()
#         soft_vector.extend(q.cpu().detach().numpy())
#         labels_vector.extend(y.numpy())
#
#     labels_vector = np.array(labels_vector).reshape(data_size)
#     total_pred = np.argmax(np.array(soft_vector), axis=1)
#     for v in range(view):
#         Hs[v] = np.array(Hs[v])
#         Zs[v] = np.array(Zs[v])
#         pred_vectors[v] = np.array(pred_vectors[v])
#     return total_pred, pred_vectors, Hs, labels_vector, Zs
#
#
# def valid(model, device, dataset, view, data_size, class_num, eval_h=False):
#     test_loader = DataLoader(
#             dataset,
#             batch_size=256,
#             shuffle=False,
#         )
#     total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors = inference(test_loader, model, device, view, data_size, class_num)
#     if eval_h:
#         print("Clustering results on low-level features of each view:")
#
#         for v in range(view):
#             kmeans = KMeans(n_clusters=class_num, n_init=100)
#             y_pred = kmeans.fit_predict(low_level_vectors[v])
#             nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
#             print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
#                                                                                      v + 1, nmi,
#                                                                                      v + 1, ari,
#                                                                                      v + 1, pur))
#
#         print("Clustering results on high-level features of each view:")
#
#         for v in range(view):
#             kmeans = KMeans(n_clusters=class_num, n_init=100)
#             y_pred = kmeans.fit_predict(high_level_vectors[v])
#             nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
#             print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
#                                                                                      v + 1, nmi,
#                                                                                      v + 1, ari,
#                                                                                      v + 1, pur))
#         print("Clustering results on cluster assignments of each view:")
#         for v in range(view):
#             nmi, ari, acc, pur = evaluate(labels_vector, pred_vectors[v])
#             print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v+1, acc,
#                                                                                      v+1, nmi,
#                                                                                      v+1, ari,
#                                                                                      v+1, pur))
#
#     print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
#     nmi, ari, acc, pur = evaluate(labels_vector, total_pred)
#     print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
#     return acc, nmi, pur
def valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=None):
    test_loader = DataLoader(
            dataset,
            batch_size=data_size,
            shuffle=False,
        )
    for batch_idx, (xs, y, _) in enumerate(test_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
    labels = y.cpu().detach().data.numpy().squeeze()

    # inference
    with torch.no_grad():
        hs, qs, zs, xrs, _ = model(xs)
        # hs, qs, zs, xrs = model(xs)
        qs, preds = model.forward_cluster(xs)
        q = sum(qs) / view
        # q = ds_Fusion.DS_Combin(qs, class_num)
        q = np.argmax(q.cpu().detach().numpy(), axis=1)

    if eval_h:
        print("Clustering results on low-level features of each view:")

        for v in range(view):
            # tsne.tSNE_PLOT(xs[v], labels, name='x_v' + str(v+1))
            # tsne.tSNE_PLOT(zs[v], labels, name='z_v' + str(v+1))
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(zs[v])
            nmi, ari, acc, pur = evaluate(labels, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))

        print("Clustering results on high-level features of each view:")

        for v in range(view):
            # tsne.tSNE_PLOT(h1s[v], labels, name='h1_v' + str(v + 1))
            # tsne.tSNE_PLOT(hs[v], labels, name='h_v' + str(v + 1))
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(hs[v])
            nmi, ari, acc, pur = evaluate(labels, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))
        print("Clustering results on cluster assignments of each view:")
        for v in range(view):
            nmi, ari, acc, pur = evaluate(labels, np.array(preds[v]))
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v+1, acc,
                                                                                     v+1, nmi,
                                                                                     v+1, ari,
                                                                                    v+1, pur))

    print("Clustering results on semantic labels: " + str(labels.shape[0]))
    nmi, ari, acc, pur = evaluate(labels, q)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur