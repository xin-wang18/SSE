import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Constants

class Stess_classification(nn.Module):
    def __init__(self):
        super(Stess_classification,self).__init__()
        self.LSTM = nn.LSTM(input_size=Constants.embedding_dim,hidden_size=256,batch_first=True,bidirectional=True)
        self.att1=nn.Linear(512,1)
        self.att2=nn.Linear(512,1)
        # self.att2=nn.Linear(512,1)
        # self.fc1=nn.Linear(512,64)
        self.fc1=nn.Linear(512,Constants.CLASS_NUM)
        # self.fc2=nn.Linear(64,Constants.CLASS_NUM)
        # self.resnet18.avgpool = nn.AvgPool2d(4, stride=1)

    def feature_fusion(self,text):
        s=text # 64*20,128,512
        att_s=self.att1(s) # 64*20,128,1
        # att_s=torch.tanh(att_s)
        att_s=F.softmax(att_s,dim=1) # 64*20,128,1
        # print("att_s",att_s.shape)
        att_s_t=att_s.transpose(1,2) # 64*20,1,128
        tweet_feature=torch.bmm(att_s_t,s) # 64*20,1,512
        tweet_feature=torch.squeeze(tweet_feature,1) # 64*20,512
        tweet_feature=tweet_feature.view(-1,Constants.MAX_TWEET_NUM,512) # 64,20,512
        # print("tweet_feature",tweet_feature.shape)
        att_u=self.att2(tweet_feature) # 64,20,1
        # att_u=torch.tanh(att_u)
        att_u=F.softmax(att_u,dim=1)
        att_u_t=att_u.transpose(1,2) # 64,1,20
        user_feature=torch.bmm(att_u_t,tweet_feature) # 64,1,512
        user_feature=torch.squeeze(user_feature,1) # 64,512
        # print("user_feature",user_feature.shape)
        return user_feature

    def forward(self,text):#64,20,128,768
        # print(len(text)
        text=text.view(-1,Constants.max_tweet_len,Constants.embedding_dim) #64*20,128,768
        # print(text.shape)
        text,(hn,cn)=self.LSTM(text) ##64*20,128,512
        # print(text.shape)
        # text=text.view(Constants.batch_size,Constants.MAX_TWEET_NUM,Constants.max_tweet_len,512) #64,20,128,512
        feature=self.feature_fusion(text) # 64,512
        # print("feature",feature.shape)
        # np.save('feature',feature.cpu().detach().numpy())
        score = self.fc1(feature) # 64,64
        score = torch.tanh(score)
        # print("score",score.shape)
        # score = self.fc2(score)
        # score = torch.tanh(score) # 64,class_num
        # print("score",score.shape)
        return score
        # return F.softmax(score,dim=1)


class mullti(nn.Module):
    def __init__(self):
        super(mullti,self).__init__()
        self.LSTM = nn.LSTM(input_size=Constants.embedding_dim,hidden_size=256,batch_first=True,bidirectional=True)
        self.fc1=nn.Linear(Constants.embedding_dim,1)
        self.fc2=nn.Linear(Constants.max_tweet_len,1)
        self.fc3=nn.Linear(Constants.MAX_TWEET_NUM,Constants.CLASS_NUM)

    def forward(self,text):#64,20,128,768
        score = self.fc1(text) #64,20,128,1
        score = torch.squeeze(score,-1) #64,20,128
        score = torch.tanh(score)
        score = self.fc2(score) #64,20,1
        score = torch.squeeze(score,-1) #64,20
        score = torch.tanh(score)
        score = self.fc3(score) #64,6
        score = torch.tanh(score)
        return score