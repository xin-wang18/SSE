import os
import re
from time import time
import random
import Constants
import sys
import torch
import json
import numpy as np
import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(word2index, path, myembed_path=None):
    # weights = [word for word, idx in word2index.items()]
    index2word = [word for word, idx in word2index.items()]
    weights = [[-1]] * len(index2word)
    print("加载预训练词向量", os.path.abspath(path))
    # doc = open(path, "r", encoding="utf-8").read().splitlines()
    f = open(path, "r", encoding="utf-8")
    embeds = []
    embed_dim = Constants.embedding_dim
    for line in f:
        if len(line) < 300:
            print("预训练词向量规模", line)
            continue
        if line[1] != ' ' or line[0] not in word2index:  # 词 生字
            continue
        tokens = line.strip().split()
        if tokens[0] not in word2index:
            continue
        index = word2index[tokens[0]]
        # print(tokens,len(tokens))
        # for token in tokens[1:]:
        #     print(tokens[1:])
        #     print(float(token))
        tokens = [float(token) for token in tokens[1:]]
        weights[index] = np.asarray(tokens)
        embeds.append(line)
    if myembed_path:
        myembed = open(myembed_path, "w", encoding="utf-8")
        myembed.writelines(embeds)
        myembed.close()
        print("myembed 写入 ", myembed_path)
    strange_words = []
    for i in range(len(weights)):
        if len(weights[i]) == embed_dim:
            continue
        else:
            weights[i] = np.random.randn(embed_dim)
            strange_words.append(index2word[i])

    print(len(strange_words), "个生词随机初始化", " ".join(strange_words))
    return torch.Tensor(weights)

def count_word(counter, word, n=1):  # 统计词频  累加n
    if word not in counter:
        counter[word] = n
    else:
        counter[word] += n

def sort_counter(counter, reverse=True):  # 词频降序
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=reverse)
    counter = dict(items)
    return counter

def counter2frequency(counter):
    sum = 0
    for word, num in counter.items():
        sum += num
    frequency = {}
    for word, num in counter.items():
        frequency[word] = num / sum
    return frequency


def counter2dict(counter, word2index=Constants.Default_Dict, min_freq=2, max_token=10000):  # 生成字典
    ignored_word_count = 0
    for word, count in counter.items():
        if len(word2index) >= max_token:
            print("词典已满")
            break
        if word not in word2index:
            if count >= min_freq:
                word2index[word] = len(word2index)
            else:
                ignored_word_count += 1
    print('[Info] 频繁字典大小 = {},'.format(len(word2index)), '最低频数 = {}'.format(min_freq))
    print("[Info] 忽略罕词数 = {}".format(ignored_word_count))
    index2word = [k for k, v in word2index.items()]
    assert len(index2word) == len(word2index)
    return word2index, index2word


def get_index2word(word2index):
    index2word = []
    for word, count in word2index.items():
        index2word.append(word)
    return index2word


def sentence2indices(line, word2index, max_len=None, padding_index=None, unk=None, began=None, end=None):
    result = [word2index.get(word, unk) for word in line if word in word2index ]
    if max_len is not None:
        result = result[:max_len]
    if began is not None:
        result.insert(0, began)
    if end is not None:
        result.append(end)
    if padding_index is not None and len(result) < max_len:
        result += [padding_index] * (max_len - len(result))
    if not result:
        a=0
    # assert len(result) == max_len
    return result


def indices2sentence(index2word, indices):
    sentence = "".join(index2word[index] for index in indices)
    return sentence


def split_train(x, rate=0.90, shuffle=True):
    if shuffle:
        random.shuffle(x)
    index = int(len(x) * rate)
    train = x[:index]
    test = x[index:]
    index = int(len(test) * 0.9)
    valid = test[:index]
    test = test[index:]
    return train, valid, test


def write_splits(x, dir="data", shuffle=True):
    if shuffle:
        random.shuffle(x)
    left = int(len(x) * 0.9)
    right = left + int(0.9 * (len(x) - left))

    with open(dir + "/train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(x[:left]))
    with open(dir + "/valid.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(x[left:right]))
    with open(dir + "/test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(x[right:]))
    print("训练集、验证集、测试集已写入", dir, "目录下")


def count_doc(doc, counter={}):
    # for index, line in enumerate(iter(lambda: read_start_of_line(reade_file), '')):
    for line in doc:
        # words = split_lans(line)
        words = list(line)
        for word in words:
            if word:
                count_word(counter, word)
    return sort_counter(counter)


def merge_counter(counter1, counter2):
    if len(counter1) > 0:
        for word, num in counter1.items():
            count_word(counter2, word, num)
    return sort_counter(counter2)


def make_batches(list, batch_size, vocab, max_len=20, shuffle=True):
    for i in range(0, len(list), batch_size):
        batch = list[i:i + batch_size]
        if shuffle:
            random.shuffle(batch)
        x, y = [], []
        for j in range(len(batch)):
            batch[j] = batch[j][:max_len]
            x.append(sentence2indices(line=batch[j], word2index=vocab, max_len=20, padding_index=Constants.PAD))
            y.append(batch[j][1])
        yield torch.LongTensor(x), torch.LongTensor(y)


def doc2tensor(examples, word2index, max_len):
    data = []
    for line in examples:
        example = sentence2indices(line, word2index=word2index, max_len=max_len, padding_index=Constants.PAD)
        data.append(example)
    return torch.LongTensor(data)


def few_data(dict_data, n_class, n_support, n_batch, max_len, word2index=None, index2word=None):  # [y][x,x,x,]
    classes = random.sample(list(dict_data.keys()), n_class)
    labels = np.array(range(n_class))
    labels = dict(zip(classes, labels))

    support_x, support_y = [], []
    batch_x, batch_y = [], []

    for c in classes:
        # examples = dict_data[c][:n_support + n_batch]
        examples = dict_data[c]
        # while n_support + n_batch>len(examples):
        #     examples+=examples
        examples = random.sample(examples, n_support + n_batch)
        support_x += examples[:n_support]
        support_y += [labels[c]] * n_support
        batch_x += examples[n_support:n_support + n_batch]
        batch_y += [labels[c]] * n_batch

    samples = doc2tensor(support_x, word2index, max_len)
    sample_labels = torch.LongTensor(support_y)
    # batches = torch.LongTensor(batch_x)
    batches = doc2tensor(batch_x, word2index, max_len)
    batch_labels = torch.LongTensor(batch_y)

    return samples, sample_labels, batches, batch_labels, classes
    # return support_x, support_y, batch_x, batch_y, labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def json_dict(path):
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    counter = count_doc(doc)
    with open("data/counter.json", "w", encoding="utf-8") as f:
        json.dump(counter, f, ensure_ascii=False)
    dict = counter2dict(counter)
    print(dict)

    labels = {}
    for line in doc:
        line = line.split("\t")[0]
        if line not in labels:
            labels[line] = len(labels)
    print(len(labels))

def arguement(doc,vocab,freq=None,tokenize=list):
    # insert del replace swap
    # line -line
    doc1=[]
    for line in doc:
        # tokens=tokenize(line)
        length=len(line)
        if length<=0:
            continue
        random_words=random.sample(vocab,2)
       
        ex_place=np.random.randint(0,length+1,1)
        in_place=np.random.randint(0,length,4)

        ## insert
        pos=ex_place[0]
        word=random_words[0]
        if pos>=length:
            l1=line+word
        else:
            l1=line[:pos]+word+line[pos:]

        pos=in_place[0]
        if pos>=length-1:
            l2=line[:pos]
        else:
            l2=line[:pos]+line[pos+1:]

        pos=in_place[1]
        word=random_words[1]
        if pos>=length-1:
            l3=line[:-1]+word
        else:
            l3=line[:pos]+word+line[pos+1:]

        i,j=in_place[2],in_place[3]
        if i>j:
            i,j=j,i
        if j+1>=length:
            l4 = line[0:i] + line[j] + line[i + 1:j] + line[i]
        else:
            l4=line[0:i]+line[j]+line[i+1:j]+line[i]+line[j+1:]
        doc1+=[line,l1,l2,l3,l4]
        # doc1+=[line,l2,l3,l4]
        # print(line,doc1)
    return doc1
def genData(path, outpath):
    f = open(path, "r", encoding="utf-8")
    # doc = open(path, "r", encoding="utf-8").read().splitlines()
    # print(path, "  样本数量 ", len(doc))
    doc2 = []
    count = {}
    # 6552400379030536455_!_101_!_news_culture_!_上联：老子骑牛读书，下联怎么对？_!_
    for line in f:
        if "\t" in line:
            continue
        words = line.split("|,|");
        if len(words) != 5:
            continue
        catogory = words[1].strip()
        sentence = words[2].strip()
        if ',' in catogory:
            catogory = catogory.split(',')[0]
        if '/' in catogory:
            catogory = catogory.split('/')[-1]
        if not catogory or not sentence:
            continue
        if catogory not in count:
            count[catogory] = 1
        else:
            if count[catogory] >= 1000:
                continue
            count[catogory] += 1
        doc2.append([catogory, sentence])

    print(count)
    f = open(outpath, "w", encoding="utf-8")
    for pair in doc2:
        if count[pair[0]] < 100:
            continue
        line = '\t'.join(pair)
        f.write(line + '\n')
    # f.writelines("\n".join(doc2))
    f.close()


if __name__ == "__main__":
    t0 = time()
    path0 = "../../data/toutiao-text-classfication-dataset/toutiao_cat_data.txt"
    path1 = "../../data/toutiao-multilevel-text-classfication-dataset/mlc_dataset.txt"
    path = "data/toutiao.txt"
    genData(path1, path)
    # json_dict(path)
    print("耗时", time() - t0)
