# 基于关节数据的人体姿态估计
## 文件存储结构
--data  
　--train  
　　--000  
　　--001  
　　--002  
　　--003  
　　--004  
　--test  
　　--000  
　　--001  
　　--002  
　　--003  
　　--004  
--readdata.py  
--train.py  
--Indrnn_plainnet.py  
--cuda_IndRNN_onlyrecurrent.py  
--opts.py  
--utils.py  
--CNN.py  

## 预处理
运行readdata.py对数据集进行预处理，得到train_label path.txt，test_label path.txt，train_label path_add.txt三个文件，每一行记录了label和.npy文件的相对路径。其中train_label path_add.txt为进行了数据增强处理的文件与原文件的路径存储，进行数据增强得到的.npy文件储存于add_data文件夹中。

## 训练与测试
### IndRNN
由于视频数据存在时间序列的相关性，我们首先尝试使用了参考资料中的IndRNN模型进行训练。<br>
模型运行环境需要CUDA，可能会因为CUDA版本不一致导致报错，目前训练使用的CUDA版本为11.1。<br>
Pycharm命令行输入运行train.py：
python -u train.py --model 'plainIndRNN' --bn_location 'bn_after' --u_lastlayer_ini --constrain_U --num_layers 3 --hidden_size 512 --dropout 0.8 --batch_size 64 --ifAugmentation --seq_len 20 --use_bneval（不进行数据增强时删除ifAugmentation参数）<br>
cuda_IndRNN_onlyrecurrent.py, Indrnn_plainnet.py, opts.py, utils.py来自[Github](https://github.com/Sunnydreamrain/IndRNN_pytorch "悬停显示文字")下的action_recognition文件夹，里面储存了适用于基于关节骨架的姿态分类任务的IndRNN模型的代码。实际模型运行时进行了一定参数调整。  
我们摒弃了此模型输入数据的部分，直接运用pytorch的Dataset，Dataloader输入数据。由于输入数据集样本量较小，我们重新写了模型训练、测试的运行代码（train.py），启用了BN层，取消了原本训练模型时调整学习率的部分，并且建议bathes，batch_size取30,64以获得较好的训练效果。  
由于数据集较小，我们尝试了数据增强，应用效果较好。<br>
#### 模型参数设置
训练输入维度17 * 3，输出分类5，内部有3个隐藏层，每层512个参数，dataloader的shuffle=False。  
输入数据形式为[320,10,51]，其中320为视频帧维度，10为每次训练的batchsize大小，51为合并后的关节三维坐标数据。  
测试输入的Dataloader中batchsize=1，否则每一批的多个输入数据会输出同一个分类结果。训练时输入Dataloader的shuffle=False，否则会出现难以拟合的情况。  
代码沿用了原模型的嵌套循环调用训练函数的形式，训练次数为batches*(len(train_dataset)/batch_size)，即训练次数随训练集样本大小增加而增加。  
学习率默认为2e-4，训练中采用交叉熵损失函数和Adam优化器。<br>


### CNN
由于IndRNN模型较大且复杂，训练次数较多，而任务使用的数据集较小，因此我们同时尝试了一个CNN模型进行分类任务（CNN.py）。  
在CNN模型中，我们摒弃了数据集在时间序列上的关联，并且每一个.npy文件的数组只取前40帧的数据输入模型训练。结果显示CNN在此数据集上同样表现良好，训练拟合速度更快，预测准确率也较高，但是在应用数据增强以后会导致测试准确率的下降。<br>
#### CNN模型参数设置
第一层卷积层：输入channel：40，输出channel：80，卷积核2 * 2  
第二层卷积层：输入channel：80，输出channel：60，卷积核2 * 2  
第一层全连接层：输入900，输出200  
第二层全连接层：输入200，输出80  
激活函数为ReLu函数，学习率设置为1e-4，训练采用交叉熵函数和Adam优化器。<br>

以上训练测试结果截图在readme.doc中。
