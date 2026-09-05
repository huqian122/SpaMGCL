import torch.nn as nn
from torch.nn.functional import normalize
import torch


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, views, input_dim, low_feature_dim, high_feature_dim, dims, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.class_num = class_num
        self.device = device
        self.views = views
        for v in range(views):
            self.encoders.append(AutoEncoder(input_dim[v], high_feature_dim).to(device))
            self.decoders.append(AutoDecoder(input_dim[v], high_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.instance_head = nn.Sequential(
            nn.Linear(high_feature_dim, low_feature_dim),
            nn.ReLU(),
            nn.Linear(low_feature_dim, high_feature_dim),
        )

        self.instance_head2 = nn.Sequential(

            nn.Linear(high_feature_dim, low_feature_dim),
            nn.ReLU(),
            nn.Linear(low_feature_dim, low_feature_dim),
        )

        self.feature_fusion_module = nn.Sequential(
            nn.Linear((high_feature_dim + high_feature_dim), high_feature_dim),
            nn.ReLU(),
            nn.Linear(high_feature_dim, low_feature_dim)
        )

        self.global_high_feature_fusion_module = nn.Sequential(
            nn.Linear(self.views * low_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, low_feature_dim)
        )

        self.cluster_head = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
            # Output: N * C matrix, which N is batch_size, C is class_num
            nn.Softmax(dim=1)
        )

    def feature_fusion(self, hs, zs_gradient):
        input = torch.cat(hs, dim=1) if zs_gradient else torch.cat(hs, dim=1).detach()
        return normalize(self.global_high_feature_fusion_module(input),dim=1)

    def forward(self, xs):
        # output: resnet features
        zs = []
        # output: instance features
        hs = []
        # output: cluster features
        qs = []
        # output: reconstructed features
        xrs = []

        h1s = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            # h = normalize(self.instance_head2(z), dim=1)
            # 细粒度特征和粗粒度特征融合
            h = normalize(self.instance_head(z), dim=1)
            h = torch.cat((h, z), dim=1)
            h = normalize(self.feature_fusion_module(h), dim=1)
            q = self.cluster_head(z)
            xr = self.decoders[v](z)
            zs.append(z)
            hs.append(h)
            qs.append(q)
            xrs.append(xr)
            # h1s.append(h1)
        H = self.feature_fusion(hs, True)
        return hs, qs, zs, xrs, H

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.instance_head(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.cluster_head(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds