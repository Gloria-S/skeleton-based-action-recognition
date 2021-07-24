import numpy as np
from torch.utils.data import Dataset
import glob

class my_dataset(Dataset):
    def __init__(self,split,ifAugumentation=False):
        self.split=split
        self.skeleton_list=[]
        self.label_list=[]
        if self.split=='train':
            if ifAugumentation:  # 数据增强
                self.filename="train_label path_add.txt"
            else:
                self.filename="train_label path.txt"
        else:
            self.filename="test_label path.txt"
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split('\n')[0].split(' ')
                # print(temp)
                self.label_list.append(int(temp[0]))
                sample = np.load(temp[1]).transpose((0, 4, 2, 3, 1))  # batchsize，单双人，帧数，关节，坐标
                # print(sample.shape)
                sample = np.pad(sample, ((0, 0), (0, 0), (0, 320 - sample.shape[2]), (0, 0), (0, 0)))
                self.skeleton_list.append(sample[0][0])
                if sample[0][1].all() != 0.0:
                    self.skeleton_list.append(sample[0][1])


    def __getitem__(self, item):
        sample=self.skeleton_list[item]
        label = self.label_list[item]
        return sample, label

    def __len__(self):
        return len(self.skeleton_list)

if __name__ == '__main__':
    split = 'test'
    dataset = my_dataset(split, False)