from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

def resize_image(image_array, new_height, new_width):
    # 转换为灰度图像
    image_gray = image_array.squeeze()  # 去除多余的维度
    # 等比例放大图像
    resized_image = cv2.resize(image_gray, (new_width, new_height))
    # 转换为原始形状
    resized_image_array = np.expand_dims(resized_image, axis=0)
    return resized_image_array

class COIL10(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'COIL10.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'COIL10.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path+'COIL10.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'COIL10.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()
class COIL20(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'COIL20.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'COIL20.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path+'COIL20.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'COIL20.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        x1 = self.x1[idx].reshape(1024)
        x2 = self.x2[idx].reshape(1024)
        x3 = self.x3[idx].reshape(1024)
        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class RGBD(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'rgbd.mat')['gt'].astype(np.int32).reshape(500,)
        self.V1 = scipy.io.loadmat(path + 'rgbd.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'rgbd.mat')['X'][0][1].astype(np.float32)

    def __len__(self):
        return 500

    # def __getitem__(self, idx):

        # x1 = self.V1[idx].reshape(784)
        # x2 = self.V2[idx].reshape(784)
        # return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def __getitem__(self, idx):
        x1 = np.transpose(self.V1[idx], (2, 0, 1))
        x2 = np.transpose(self.V2[idx], (2, 0, 1))
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = data['X1'].astype(np.float32)
        self.view2 = data['X2'].astype(np.float32)
        self.labels = scipy.io.loadmat(path)['Y'].transpose()

    def __len__(self):
        return 483

    def __getitem__(self, idx):
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class Scene15():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Scene15.mat')
        scaler = MinMaxScaler()
        self.Y = data['Y'].astype(np.int32).reshape(4485,)
        # self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        # self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        # self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)

    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()


class MSRCv1():
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.Y = data['Y'].astype(np.int32).reshape(210,)

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        # return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
        #     self.V2[idx]), torch.from_numpy(self.V3[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()
        return [self.V1[idx], self.V2[idx], self.V3[idx]], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class digit(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'digit.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'digit.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'digit.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(1024)
        x2 = self.V2[idx].reshape(1024)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('../data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('../data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "digit":
        dataset = digit('../data/')
        dims = [1024, 1024]
        view = 2
        data_size = 5000
        class_num = 10
    elif dataset == "CCV":
        dataset = CCV('../data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('../data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('../data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('../data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('../data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('../data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "COIL10":
        dataset = COIL10('../data/')
        dims = [1024, 1024, 1024]
        view = 3
        data_size = 720
        class_num = 10
    elif dataset == "COIL20":
        dataset = COIL20('../data/')
        dims = [1024, 1024, 1024]
        view = 3
        data_size = 1440
        class_num = 20
    elif dataset == "RGBD":
        dataset = RGBD('../data/')
        dims = [4096, 4096]
        view = 2
        data_size = 500
        class_num = 50
    elif dataset == "Synthetic3d":
        dataset = synthetic3d('../data/')
        dims = [3,3,3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Prokaryotic":
        dataset = prokaryotic('../data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "DHA":
        dataset = DHA('../data/DHA.mat')
        dims = [110, 6144]
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == "Scene15":
        dataset = Scene15('../data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "MSRCv1":
        dataset = MSRCv1('../data/MSRCv1.mat')
        dims = [24, 576, 512]
        view = 3
        data_size = 210
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num


# if __name__ == '__main__':
#     dataset, dims, view, data_size, class_num = load_data('MSRCv1')
#     # dataset, dims, view, data_size, class_num = load_data('MNIST-USPS')
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=256,
#         shuffle=True,
#         drop_last=True,
#     )
#     device = 'cpu'
#     print(len(data_loader))
#     for batch_idx, (xs, _, _) in enumerate(data_loader):
#         for v in range(view):
#             xs[v] = xs[v].to(device)
#             print(xs[v])
            # if v == 0:
            #     print(xs[v][0].shape)