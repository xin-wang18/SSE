import os
from torch.utils.data import Dataset
import numpy as np
import random
import json
import torch
import time
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
import Constants
from Util import count_doc, counter2dict, load_weights, sentence2indices

class Classification_Dataset(Dataset):

    def __init__(self,root):
        self.root=root
        class_folders = Constants.class_folders
        labels = np.array(range(len(class_folders)))
        # print(labels)
        self.labels = dict(zip(class_folders, labels))
        # print(labels)
        self.user_roots = []
        self.user_tensor_lists = []
        all_tweet_list = []
        user_tweet_list = []
        # self.batch_roots = []

        samples = dict()
        # batch_num= 5 if self.split == "train" else 5
        for c in class_folders:
            temp = [os.path.join(root+c, x) for x in os.listdir(root+c)]
            # samples[c] = random.sample(temp, len(temp))
            self.user_roots += temp
            for user in temp:
                # print(user)
                week_data_file = user+"/week.txt"
                # print(week_data_file)
                f = open(week_data_file,'r')
                # tweets = f.read().splitlines()
                tweets = f.readlines()
                # print(tweets)
                user_tweets = []
                for tweet in tweets:
                    tweet=tweet.replace("\xa0","")
                    tweet=tweet.replace(" \n","")
                    tweet=tweet.replace("\n","")
                    # print(tweet)
                    all_tweet_list.append(tweet)
                    user_tweets.append(tweet)
                f.close()
                user_tweet_list.append(user_tweets)
        # print(all_tweet_list,len(all_tweet_list))
        # print(len(self.user_roots))
        # print(user_tweet_list,len(user_tweet_list))
        counter = count_doc(all_tweet_list)
        word2index, index2word = counter2dict(counter=counter, min_freq=0)
        weights = load_weights(word2index, Constants.embedding_root)
        word_embeddings = torch.nn.Embedding(weights.size(0), weights.size(1))
        word_embeddings.weight = torch.nn.Parameter(weights)
        total_tweet_num=0
        for i in range(len(user_tweet_list)):
            user_tensor=torch.zeros([1,128,Constants.embedding_dim])
            tweet_num=len(user_tweet_list[i])
            total_tweet_num+=tweet_num
            if tweet_num<Constants.MAX_TWEET_NUM: #小于则补齐
                for j in range(len(user_tweet_list[i])):
                    post=user_tweet_list[i][j]
                    index=sentence2indices(post, word2index, Constants.max_tweet_len, Constants.PAD)
                    post_embedding=torch.unsqueeze(word_embeddings(torch.tensor(index)),dim=0)
                    user_tensor=torch.cat((user_tensor,post_embedding),dim=0)
                    # print(post)
                    # print(index)
                    # print(post_embedding.shape)
                user_tensor=user_tensor[1:,:,:]
                # print(user_tensor.shape)
                pad_tensor=torch.zeros([Constants.MAX_TWEET_NUM-tweet_num,128,Constants.embedding_dim])
                user_tensor=torch.cat((user_tensor,pad_tensor),dim=0)
            else:
                for j in range(Constants.MAX_TWEET_NUM):
                    post=user_tweet_list[i][j]
                    index=sentence2indices(post, word2index, Constants.max_tweet_len, Constants.PAD)
                    post_embedding=torch.unsqueeze(word_embeddings(torch.tensor(index)),dim=0)
                    user_tensor=torch.cat((user_tensor,post_embedding),dim=0)
                    # print(post)
                    # print(index)
                    # print(post_embedding.shape)
                user_tensor=user_tensor[1:,:,:]
                # user_tensor=user_tensor.detach().numpy()
                # print(type(user_tensor))
                # user_tensor = torch.autograd.Variable(user_tensor,requires_grad=False)
                # user_tensor = user_tensor.data

                # print(user_tensor.shape)    
            # print(user_tensor.shape)
            user_tensor=user_tensor.detach().numpy() 
            # print(type(user_tensor))           
            self.user_tensor_lists.append(user_tensor)
        print(len(self.user_tensor_lists))
        print("总推文数",total_tweet_num)

    def __len__(self):
        return len(self.user_roots)

    def __getitem__(self,index):
        user_label = self.labels[self.get_class(self.user_roots[index])]
        user_feature= self.user_tensor_lists[index]
        # user_feature = torch.from_numpy(user_feature)
        # user_feature = torch.autograd.Variable(user_feature,requires_grad=False)
        # user_feature = user_feature.data

        return user_feature,user_label
    
    def get_class(self,sample):
        return os.path.join(sample.split('/')[-2])

class Proportion_Classification_Dataset(Dataset):

    def __init__(self,root,proportion):
        self.root=root
        self.proportion=proportion
        class_folders = Constants.class_folders
        labels = np.array(range(len(class_folders)))
        # print(labels)
        self.labels = dict(zip(class_folders, labels))
        # print(labels)
        self.user_roots = []
        self.user_tensor_lists = []
        all_tweet_list = []
        user_tweet_list = []
        # self.batch_roots = []

        samples = dict()
        # batch_num= 5 if self.split == "train" else 5
        for c in class_folders:
            temp = [os.path.join(root+c, x) for x in os.listdir(root+c)]
            # samples[c] = random.sample(temp, len(temp))
            temp = temp[:int(len(temp)*self.proportion)]
            self.user_roots += temp
            # print(temp)
            for user in temp:
                # print(user)
                week_data_file = user+"/week.txt"
                # print(week_data_file)
                f = open(week_data_file,'r')
                # tweets = f.read().splitlines()
                tweets = f.readlines()
                # print(tweets)
                user_tweets = []
                for tweet in tweets:
                    tweet=tweet.replace("\xa0","")
                    tweet=tweet.replace(" \n","")
                    tweet=tweet.replace("\n","")
                    # print(tweet)
                    all_tweet_list.append(tweet)
                    user_tweets.append(tweet)
                f.close()
                user_tweet_list.append(user_tweets)
        # print(all_tweet_list,len(all_tweet_list))
        # print(len(self.user_roots))
        # print(user_tweet_list,len(user_tweet_list))
        counter = count_doc(all_tweet_list)
        word2index, index2word = counter2dict(counter=counter, min_freq=0)
        weights = load_weights(word2index, Constants.embedding_root)
        word_embeddings = torch.nn.Embedding(weights.size(0), weights.size(1))
        word_embeddings.weight = torch.nn.Parameter(weights)
        total_tweet_num=0
        for i in range(len(user_tweet_list)):
            user_tensor=torch.zeros([1,128,Constants.embedding_dim])
            tweet_num=len(user_tweet_list[i])
            total_tweet_num+=tweet_num
            if tweet_num<Constants.MAX_TWEET_NUM: #小于则补齐
                for j in range(len(user_tweet_list[i])):
                    post=user_tweet_list[i][j]
                    index=sentence2indices(post, word2index, Constants.max_tweet_len, Constants.PAD)
                    post_embedding=torch.unsqueeze(word_embeddings(torch.tensor(index)),dim=0)
                    user_tensor=torch.cat((user_tensor,post_embedding),dim=0)
                    # print(post)
                    # print(index)
                    # print(post_embedding.shape)
                user_tensor=user_tensor[1:,:,:]
                # print(user_tensor.shape)
                pad_tensor=torch.zeros([Constants.MAX_TWEET_NUM-tweet_num,128,Constants.embedding_dim])
                user_tensor=torch.cat((user_tensor,pad_tensor),dim=0)
            else:
                for j in range(Constants.MAX_TWEET_NUM):
                    post=user_tweet_list[i][j]
                    index=sentence2indices(post, word2index, Constants.max_tweet_len, Constants.PAD)
                    post_embedding=torch.unsqueeze(word_embeddings(torch.tensor(index)),dim=0)
                    user_tensor=torch.cat((user_tensor,post_embedding),dim=0)
                    # print(post)
                    # print(index)
                    # print(post_embedding.shape)
                user_tensor=user_tensor[1:,:,:]
                # user_tensor=user_tensor.detach().numpy()
                # print(type(user_tensor))
                # user_tensor = torch.autograd.Variable(user_tensor,requires_grad=False)
                # user_tensor = user_tensor.data

                # print(user_tensor.shape)    
            # print(user_tensor.shape)
            user_tensor=user_tensor.detach().numpy() 
            # print(type(user_tensor))           
            self.user_tensor_lists.append(user_tensor)
        print(len(self.user_tensor_lists))
        print("总推文数",total_tweet_num)

    def __len__(self):
        return len(self.user_roots)
        # return int(len(self.user_roots)*self.proportion)

    def __getitem__(self,index):
        user_label = self.labels[self.get_class(self.user_roots[index])]
        user_feature= self.user_tensor_lists[index]
        # user_feature = torch.from_numpy(user_feature)
        # user_feature = torch.autograd.Variable(user_feature,requires_grad=False)
        # user_feature = user_feature.data

        return user_feature,user_label
    
    def get_class(self,sample):
        return os.path.join(sample.split('/')[-2])


def main():
    # loader=Proportion_Classification_Dataset(root=Constants.train_root,proportion=1)
    loader=Classification_Dataset(root=Constants.train_root)
    # support_text_list,support_image_list,batch_text_list,batch_image_list=loader.__getitem__(0)
    time_start=time.time()
    dataloader = DataLoader(
        loader,
        3000,
        shuffle = True,
        num_workers = 4
        )
    print("fasdf")
    for i, (text,label) in enumerate(dataloader):
        print(text.shape,label)
        # torch.save(text,'sample_feature')
        # break
    # for i, (text,label) in enumerate(dataloader):
    #     print(text.shape)
    #     # print(img.shape)
    #     # print(label)
    #     break
        # print(img.shape)
    # support_text_list,support_image_list,support_labels,batch_text_list,batch_image_list,batch_labels=loader.__getitem__(0)
    # print(len(support_text_list))
    time_end=time.time()
    sum_t=time_end-time_start
    print("time cost",sum_t,"s")
if __name__ == '__main__':
    main()