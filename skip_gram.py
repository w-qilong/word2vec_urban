import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Counter
import numpy as np
import random
import pandas as pd
import scipy.spatial
from sklearn.manifold import TSNE
import os

# CUDA是否可用
USE_CUDA = torch.cuda.is_available()
print("CUDA: %s" % USE_CUDA)

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

# 设定一些超参数
K = 5  # 负采样个数
NUM_EPOCHS = 150  # 训练轮数
BATCH_SIZE = 128  # 每批样本的数量
LEARNING_RATE = 1e-5  # 学习率
EMBEDDING_SIZE = [40, 50, 60, 70, 80, 90, 100]  # 词向量维度
SAVE_LOG = True  # 是否保存日志
DATA_PATH = r'process_data\K_nearest_result.csv'  # POI文件路径，格式：（center,context）
hierarchy_path = r'process_data\first_second_third.csv'

# 加载(center,context)训练数据集
in_data = pd.read_csv(DATA_PATH).values
words = []
targets = [x[1] for x in in_data]
for i in range(0, len(in_data), 4):
    words.append(in_data[i][0])

# 加载层级结构csv文件
hierarchy = pd.read_csv(hierarchy_path)

# 构建词典
vocab = dict(Counter(words))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

# 计算词频，用于负采样
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
VOCAB_SIZE = len(idx_to_word)
print("VOCAB_SIZE: %s" % VOCAB_SIZE)


# 实现Dataloader,用于加载训练数据集
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, targets, words, word_to_idx, idx_to_word, word_freqs, word_counts, K):
        """
        用于组装训练数据集（center,context）
        :param targets: 对应center
        :param words: 对应包含所有（center,context）的列表
        :param word_to_idx:根据词查询对应序号的字典
        :param idx_to_word:根据序号查询词的字典
        :param word_freqs: 根据skip-gram论文中计算词频的公式计算得到的各POI类型的频率
        :param word_counts: 每个POI类型的对应的总数
        """
        super(WordEmbeddingDataset, self).__init__()
        self.targets = targets
        self.words = words

        self.words = [word_to_idx.get(t) for t in words]
        self.words = torch.Tensor(self.words)

        self.targets = [word_to_idx.get(t) for t in targets]
        self.targets = torch.Tensor(self.targets)

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        self.K = K

    def __len__(self):
        """ 返回整个数据集（所有单词）的长度, 这里数据集对应的是中心词的数量"""
        return len(self.words)

    def __getitem__(self, idx):
        """这里idx对应inData数据集合中的下标索引"""
        center_word = self.words[idx]  # 中心词对应的idx
        pos_words = self.targets[idx * 4:idx * 4 + 4]  # 与中心词对应的4个背景词
        neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)  # 随机抽取负采样样本
        return center_word, pos_words, neg_words


# 定义采用负采样的Skip-gram模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        """ 初始化输出和输出embedding"""

        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        """
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        return: loss, [batch_size]
        """

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        """
        将训练得到的词嵌入向量迁移到CPU上，转换为numpy数组
        :return:
        """
        return self.in_embed.weight.data.cpu().numpy()

    def cal_similarity_between_words(self, first_word, second_word):
        """
        计算两个词之间的余弦相似度
        :param first_word:
        :param second_word:
        :return:
        """
        embedding_weights = self.input_embeddings()
        vector_of_first_word = embedding_weights[word_to_idx[first_word]]
        vector_of_second_word = embedding_weights[word_to_idx[second_word]]
        return vector_of_first_word.dot(vector_of_second_word) / (
                np.linalg.norm(vector_of_first_word) * np.linalg.norm(vector_of_second_word))

    def sort_poi(self, word):
        """
        计算word与其它POI类型语义向量之间的相似性，根据余弦距离返回排序后的POI列表,用于计算后续的MAP得分
        :param word:
        :return:
        """
        index = word_to_idx[word]
        embedding_weights = self.input_embeddings()
        word_embedding = embedding_weights[index]
        cos_dis = np.array([scipy.spatial.distance.cosine(e, word_embedding) for e in
                            embedding_weights])
        return cos_dis.argsort().tolist()

    def cal_MAP_score(self, hierarchy):
        """
        计算MAP得分
        :param hierarchy:POI分类等级对应表格路径
        :return:返回MAP得分
        """
        group = hierarchy.groupby(by='second')
        same_third = []
        for i in list(group):
            same_third.append(i[1]['third'].to_list())
        MAP = []
        # 对于每一三级分类
        for third in hierarchy['third'].values:
            for tmp_group in same_third:
                if third in tmp_group:
                    sorted_poi_sim = self.sort_poi(third)  # 得到相似性排序的idx
                    tmp_group_index = [word_to_idx[item] for item in tmp_group]  # 找到当前同级分类POI类型的idx
                    tmp_group_rank = [sorted_poi_sim.index(idx) for idx in tmp_group_index]
                    tmp_group_rank_sorted = sorted(tmp_group_rank)
                    tmp_score_list = [(tmp_group_rank_sorted.index(poi) + 1) / (poi + 1) for poi in
                                      tmp_group_rank_sorted]  # 计算MAP得分
                    cur_score = np.mean(tmp_score_list)
            MAP.append(cur_score)
        return np.mean(MAP)


# 构建训练过程中的data_loader
dataset = WordEmbeddingDataset(targets, words, word_to_idx, idx_to_word, word_freqs, word_counts, K)
data_loader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print('The data set has been built!')

# 以下为模型训练过程
for dim in EMBEDDING_SIZE:
    # 结果保存路径
    model_path = 'Result/negativeNum_%s_epoch_%s_embeddingSize_%s' % (K, NUM_EPOCHS, dim)
    if os.path.exists(model_path):
        print("Folder is exist!")
    else:
        os.mkdir(model_path)

    # 训练日志路径
    LOG_FILE = os.path.join(model_path, 'log.txt')  # 包含训练epoch，iteration，loss
    LOG_FILE_MAP = os.path.join(model_path, 'MAP_SCORE.txt')  # 记录每轮MAP score

    # 定义模型并将模型迁移到GPU上
    model = EmbeddingModel(VOCAB_SIZE, dim)
    if USE_CUDA:
        model = model.cuda()
        print('GPU IS USING!')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 定义优化器

    # 加载模型参数
    for e in range(0, NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(data_loader):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
            if USE_CUDA:
                input_labels = input_labels.cuda()
                pos_labels = pos_labels.cuda()
                neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                if SAVE_LOG:
                    with open(LOG_FILE, "a") as f_out:
                        f_out.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))

        # 每训练一轮进行一次评估
        print("epoch: %s, 火锅店，中餐厅：%f" % (e, model.cal_similarity_between_words('火锅店', '中餐厅')))
        print("epoch: %s, 火锅店，购物相关场所：%f" % (e, model.cal_similarity_between_words('火锅店', '购物相关场所')))

        # 计算MAP得分
        MAP_score = model.cal_MAP_score(hierarchy)
        with open(LOG_FILE_MAP, "a") as f_out:
            f_out.write("epoch: %s, MAP_score: %f\n" % (e, MAP_score))
        print("epoch: %s, MAP_score: %f" % (e, MAP_score))

    # 保存模型
    np.save(os.path.join(model_path, 'embedding'), model.input_embeddings())
    np.save(os.path.join(model_path, 'word_to_idx'), word_to_idx)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e, 'MAP_SCORE': MAP_score}
    torch.save(state, os.path.join(model_path, 'embedding.th'))
    print('Model save complete!')

    # 使用T-SNE降维词向量到2维和3维,便于后续可视化
    label_list = list(word_to_idx.keys())
    for dimension in [2, 3]:
        tsne = TSNE(n_components=dimension, init='pca', random_state=0)
        y = tsne.fit_transform(model.input_embeddings())
        if dimension == 2:
            tx = y[:, 0]  # 获取x
            ty = y[:, 1]  # 获取y
            outdict = {'third': label_list, 'x': tx, 'y': ty}  # 将降维后的向量使用字典存储
        else:
            tx = y[:, 0]  # 获取x
            ty = y[:, 1]  # 获取y
            tz = y[:, 2]  # 获取z
            outdict = {'third': label_list, 'x': tx, 'y': ty, 'z': tz}  # 将降维后的向量使用字典存储
        out_dataframe = pd.DataFrame(data=outdict)
        result = out_dataframe.set_index('third').join(hierarchy.set_index('third'), on='third')  # 添加一级分类和二级分类到特征向量
        result.to_csv(os.path.join(model_path, 'TSNE_%s.csv' % dimension))
    print('TSNE processing is complete!')
