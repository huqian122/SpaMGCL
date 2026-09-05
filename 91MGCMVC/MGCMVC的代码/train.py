import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch
from scipy.stats import wasserstein_distance
import pandas as pd
import torch.nn.functional as F
import ds_Fusion
# import FCM_self
# MNIST-USPS, 20
# BDGP, 120
# CCV 20,
# Fashion, 30
# Caltech-2V, 42
# Caltech-3V, 80
# Caltech-4V
# Caltech-5V, 70
# Multi-COIL-10
# COIL20
# Prokaryotic, 40
# Synthetic3d
# DHA
# MSRC
# Scene15
# digit
Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument("--mse_epochs", default=200)
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
args.con_epochs = 100
args.seed = 70

def setup_seed(seed):
    print("seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


dataset, input_dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def compute_view_value(rs, H, view):
    w = []
    for v in range(view):
        w.append(torch.exp(-torch.tensor(wasserstein_distance(H.cpu().numpy().flatten(), rs[v].cpu().numpy().flatten()))))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        # 将数据存入CPU or GPU
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, _, xrs, _ = model(xs)
        # _, _, _, xrs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    # batch_idx 索引，xs 多视图数据，_ 对应标签，_ 对应数据索引
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, zs, xrs, H = model(xs)
        # hs, qs, zs, xrs = model(xs)
        loss_list = []
        with torch.no_grad():
            weights = compute_view_value(hs, H, view)
        # 两两视图之间进行对比学习
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append((weights[v] + weights[w]) * criterion.forward_feature(hs[v], hs[w]))
                # loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    avg_loss = tot_loss / len(data_loader)
    print(f'Epoch {epoch} Loss:{avg_loss:.6f}')
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return avg_loss

def save_to_excel(data, filename):
    df = pd.DataFrame(data, columns=['Epochs', 'Loss', 'ACC', 'NMI', 'PUR'])
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    setup_seed(args.seed)
    # low_feature_dim = 64
    # high_feature_dim = 128
    low_feature_dim = 128
    high_feature_dim = 256
    dims = [500, 500, 2000]
    model = Network(view, input_dims, low_feature_dim, high_feature_dim, dims, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    accs = []
    nmis = []
    purs = []
    results = []
    best_acc, best_nmi, best_pur = 0, 0, 0
    while epoch <= args.mse_epochs + args.con_epochs:
        loss = contrastive_train(epoch)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        results.append([epoch, loss, acc, nmi, pur])
        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
        epoch += 1

    # The final result
    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))
    save_to_excel(results, './results.xlsx')
