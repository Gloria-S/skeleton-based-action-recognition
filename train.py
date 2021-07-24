from __future__ import print_function
from dataloader import my_dataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import torch
import torch.nn as nn

# Set the random seed manually for reproducibility.
seed = 100
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    print("WARNING: CUDA not available")

import opts

parser = argparse.ArgumentParser(description='pytorch action')
opts.train_opts(parser)
args = parser.parse_args()
print(args)

outputclass = 5
indim = 17 * 3
batch_size = args.batch_size
seq_len = args.seq_len
gradientclip_value = args.gradclipvalue
if args.U_bound == 0:
    U_bound = np.power(10, (np.log10(args.MAG) / args.seq_len))
else:
    U_bound = args.U_bound

import Indrnn_plainnet as Indrnn_network

model = Indrnn_network.stackedIndRNN_encoder(indim, outputclass)
model.cuda()
criterion = nn.CrossEntropyLoss()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

# Adam with lr 2e-4 works fine.
learning_rate = args.lr
param_decay = []
param_nodecay = []
for name, param in model.named_parameters():
    if 'weight_hh' in name or 'bias' in name:
        param_nodecay.append(param)
        # print('parameters no weight decay: ',name)
    elif (not args.bn_decay) and ('norm' in name):
        param_nodecay.append(param)
        # print('parameters no weight decay: ',name)
    else:
        param_decay.append(param)
        # print('parameters with weight decay: ',name)

optimizer = torch.optim.Adam([
    {'params': param_nodecay},
    {'params': param_decay, 'weight_decay': args.decayfactor}
], lr=learning_rate)

ifDataAugumentation=args.ifAugmentation  # 是否进行数据增强
split = 'train'
train_dataset = my_dataset(split, ifDataAugumentation)
train_dataloader = DataLoader(train_dataset, batch_size=10, num_workers=0, shuffle=False)
split = 'test'
test_dataset = my_dataset(split)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)



def train():
    model.train()
    tacc = 0
    count = 0
    loss_mean = 0
    start_time = time.time()
    for input, target in train_dataloader:
        input = input.to(torch.float32).numpy().transpose(1, 0, 2, 3)
        input = torch.from_numpy(input)
        # print(input.shape)
        target = target.long()
        # print(target)
        seq_len, batch_size, joints_no, _ = input.size()
        # print(input.size())
        input = input.view(seq_len, batch_size, 3 * joints_no)
        # print(input.size())
        input = input.cuda()
        target = target.cuda()

        model.zero_grad()
        if args.constrain_U:
            clip_weight(model, U_bound)

        output = model(input)
        # print(output)
        loss = criterion(output, target)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        accuracy = pred.eq(target.data).cpu().sum().numpy() / (0.0 + target.size(0))
        loss.backward()
        clip_gradient(model, gradientclip_value)
        optimizer.step()
        tacc = tacc + accuracy
        count += 1
        loss_mean+=loss.item()
    elapsed = time.time() - start_time
    # print(tacc)
    loss_mean/=(count+0.0)
    print("loss:{:.3f},training accuracy:{}".format(loss_mean,tacc / (count + 0.0)))

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def test(dataloader,num_dataset,use_bneval=False):
    model.eval()
    if use_bneval==True:
        model.apply(set_bn_train)
    tacc=0
    count=0
    start_time=time.time()
    total_testdata=num_dataset
    total_ave_acc=np.zeros((total_testdata,outputclass))
    testlabels=np.zeros((total_testdata,))
    for input,target in dataloader:
        testlabels[count]=target
        input = input.to(torch.float32).numpy().transpose(1, 0, 2, 3)
        input = torch.from_numpy(input)
        target = target.long()
        # print(target)
        seq_len, batch_size, joints_no, _ = input.size()
        # print(input.size())
        input = input.view(seq_len, batch_size, 3 * joints_no)
        # print(input.size())
        input = input.cuda()
        target = target.cuda()
        output=model(input)
        pred=output.data.max(1)[1]
        accuracy=pred.eq(target.data).cpu().sum().numpy()/target.data.size(0)
        # print(output.data.cpu().numpy()[0],output.data.cpu().numpy()[0].shape)
        total_ave_acc[count]+=output.data.cpu().numpy()[0]
        tacc+=accuracy
        count+=1
    top = np.argmax(total_ave_acc, axis=-1)
    print("预测分类:",top)
    print("实际标签：",testlabels)
    eval_acc = np.mean(np.equal(top, testlabels))
    elapsed = time.time() - start_time
    # print("test_Acc",eval_acc)
    # print("test accuracy: ", tacc / (count + 0.0), eval_acc)
    # print ('test time per batch: ', elapsed/(count+0.0))
    return tacc / (count + 0.0)  # , eval_acc/(total_testdata+0.0)

def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip, clip)


def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
        if 'weight_hh' in name:
            param.data.clamp_(-clip, clip)

num_train_batches=int(np.ceil(len(train_dataset)/(batch_size+0.0)))
num_testdataset=int(len(test_dataset))
batches=30

for batchi in range(0, batches):
    for i in range(num_train_batches):
        train()
    print(batchi)# ,"test,",test_Acc)

test_acc = test(test_dataloader,num_testdataset,True)
print("test accuracy:",test_acc)
save_name = 'indrnn_action_model'
with open(save_name, 'wb') as f:
    torch.save(model.state_dict(), f)




