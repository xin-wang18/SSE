
import torch
import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report,confusion_matrix
import time
import os
import jieba
# from torch.nn.parallel import DistributedDataParallel
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 
import json
import re
import shutil
import datetime
from snownlp import SnowNLP
import multiprocessing
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, BertModel
from transformers import BertConfig
import torch
import random
import faiss
from utily_new import stressful_emotion_masked

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def test():
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    tokenizer, model = init_bert()

    inputs = tokenizer("我喜欢[MASK]", return_tensors="pt")
    labels = tokenizer("我喜欢你", return_tensors="pt")["input_ids"]
    # print (**inputs)
    print (inputs)
    print (labels)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_encodings):
        self.data_encodings = data_encodings
    # def __init__(self, original_encodings,positive_encodings, negative_encodings):
    #     self.positive_encodings = positive_encodings
    #     self.negative_encodings = negative_encodings
    #     self.original_encodings = original_encodings
        # self.inputs = [{key: val} for key, val in self.encodings.items()]
        # self.labels = labels

    def __getitem__(self, idx):
        # original_item=self.original_encodings[idx]
        # positive_item=self.positive_encodings[idx]
        # negative_item=self.negative_encodings[idx]
        item=[]
        for i in range(len(self.data_encodings)):
            item.append({key: torch.tensor(self.data_encodings[i][key][idx][:128]) for key in self.data_encodings[i].keys()})
        # original_item = {key: torch.tensor(self.original_encodings[key][idx][:128]) for key in self.original_encodings.keys()}
        # positive_item = {key: torch.tensor(self.positive_encodings[key][idx][:128]) for key in self.positive_encodings.keys()}
        # negative_item = {key: torch.tensor(self.negative_encodings[key][idx][:128]) for key in self.negative_encodings.keys()}

        # new_input_ids = item['input_ids']
        # for i in range(1,len(item['input_ids'])-1):
        #     if item['input_ids'][i]>=533 and random.random()<0.15:
        #         item['input_ids'][i]=103

        # item['labels'] = torch.tensor(int(self.labels[idx]))
        return item, idx

    def __len__(self):
        return len(self.data_encodings[0]["input_ids"])


batch_size = 12
max_epoch = 3
lr = 0.000003
T=0.07
lam=0.6
weight_decay = 0.01
device=7
device_for_cluster = 2
saved_name = './XIN/cluster_lam_0.6_epoch_3_new'
warmup_epoch = 0
feature_dim = 768
num_cluster = [5,10,15]
r = 1

def init_bert():
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/Jax/transformer/' + model_name)

    config = BertConfig.from_json_file('/home/ubuntu/Jax/transformer/'+ model_name+'/config_xin.json')
    model = BertForSequenceClassification(config).from_pretrained('/home/ubuntu/Jax/transformer/' + model_name,num_labels = 768)
    # model = BertModel(config).from_pretrained('/home/ubuntu/Jax/transformer/' + model_name,num_labels = 768)
    # model = BertForMaskedLM(config).from_pretrained('/home/ubuntu/Jax/transformer/' + model_name)
    print(model)
    # print(model.cls.predictions.decoder)

    # tokenizer = BertTokenizer.from_pretrained('_saved_model_directory' )
    # model = BertForMaskedLM.from_pretrained('_saved_model_directory')

    model.cuda(device)
    return tokenizer, model


def load_data():
    stress_data = open('data/tweet_for_pretrain_clean.txt','r', encoding='utf8').readlines()
    # stress_data = open('data/tweet_for_pretrain_small.txt','r', encoding='utf8').readlines()
    stress_data = [x.strip() for x in stress_data]
    # print(stress_data)
    # labels = [x.split("\t")[1] for x in stress_data]
    # # print(labels)
    # stress_data = [x.split("\t")[0] for x in stress_data]
    # print(stress_data)
    original_data,emotion_positive_data,stressor_positive_data,emotion_negative_data,emotion_replace_negative_data,stressor_negative_data=stressful_emotion_masked(stress_data)
    print("masked over")
    return [original_data,emotion_positive_data,stressor_positive_data,emotion_negative_data,emotion_replace_negative_data,stressor_negative_data]

def compute_features(eval_loader, model):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),768).cuda(device)
    for i, ((tokenized_text,), index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            tokenized_text = {key: tokenized_text[key].cuda(device) for key in tokenized_text}
            feat = model(**tokenized_text) 
            features[index] = feat.logits
    return features.cpu()

def run_kmeans(x):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    # for seed, num_cluster in enumerate(args.num_cluster):
    for seed, num in enumerate(num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20 #clustering iterations
        clus.nredo = 5  #redo clustering this many times and keep best
        clus.seed = seed
        clus.max_points_per_centroid = 5000
        clus.min_points_per_centroid = 100

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        # cfg.device = args.gpu
        cfg.device = device_for_cluster
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        # density = args.temperature*density/density.mean()  #scale the mean to temperature 
        density = T*density/density.mean()  #scale the mean to temperature
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda(device)
        # print(centroids)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        # print(centroids)
        # exit()

        im2cluster = torch.LongTensor(im2cluster).cuda(device)               
        density = torch.Tensor(density).cuda(device)
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def task_adapt_bert():

    tokenizer, model = init_bert()

    # original_data,positive_data,emotion_negative_data,stressor_negative_data = load_data()
    data=load_data()
    # print(data[1])
    # return
    data_encodings=[]
    # for i in range(len(data[2])):
    #     try:
    #         data_encodings.append(tokenizer(data[2][i],padding=True))
    #     except:
    #         print(data[2][i])
    #         print(i)       
    for i in range(len(data)):
        # print(data[i])
        try:
            data_encodings.append(tokenizer(data[i],padding=True))
        except:
            print(data[i])
            print(i)
    # print(data_encodings[0].keys())
    # print(len(data_encodings[:1]))
    # return
    print("tokenizer over")
    # print(type(train_encodings))

    # print (labels)

    # train_dataset = Dataset(original_train_encodings,positive_train_encodings, negative_train_encodings)
    train_dataset = Dataset(data_encodings)
    cluster_dataset = Dataset(data_encodings[:1])
    # print(train_dataset.__getitem__(100))
    # return 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle = True,
        # 要求按顺序获取数据
        num_workers = 0
    )
    cluster_dataloader = DataLoader(
        cluster_dataset,
        batch_size*4,
        shuffle = False,
        num_workers = 0
    )
    optimizer = torch.optim.Adam(model.parameters(),
    lr = lr,
    weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(device)
    print (len(train_dataset))
    for epoch in range(max_epoch):
        cluster_result = None
        if epoch>=warmup_epoch:
            # compute features for center-cropped texts
            features = compute_features(cluster_dataloader, model)
            # print(features.shape)
            # exit()
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num in num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(cluster_dataset.__len__(),dtype=torch.long).cuda(device_for_cluster))
                cluster_result['centroids'].append(torch.zeros(int(num),feature_dim).cuda(device_for_cluster))
                cluster_result['density'].append(torch.zeros(int(num)).cuda(device_for_cluster))
                # print(cluster_result['im2cluster'][0].shape)
                # print(cluster_result['centroids'][0].shape)
                # print(cluster_result['density'][0].shape)
            
            features = features.numpy()
            cluster_result = run_kmeans(features)
            # print(cluster_result.items())
            # exit()

        model.train()
        for i, ((o,p_e,p_s,n_e,n_e_o,n_s),index) in tqdm(enumerate(train_dataloader),total=len(train_dataset)/batch_size):
            # X = X.cuda()
            # print(index)
            # exit()
            optimizer.zero_grad()
            # print(len(o), o["input_ids"][1])
            # print(len(p), p["input_ids"][1])
            # print(len(n_e), n_e["input_ids"][1])
            # print(len(n_s), n_s["input_ids"][1])
            # print (batch, len(batch))
            o = {key: o[key].cuda(device) for key in o}
            # print(o)
            # exit()
            o_outputs = model(**o)
            # print(o_outputs.logits.shape)
            # exit()
            p_e = {key: p_e[key].cuda(device) for key in p_e}
            p_e_outputs = model(**p_e)
            p_s = {key: p_s[key].cuda(device) for key in p_s}
            p_s_outputs = model(**p_s)
            n_e = {key: n_e[key].cuda(device) for key in n_e}
            n_e_outputs = model(**n_e) 
            n_e_o = {key: n_e_o[key].cuda(device) for key in n_e_o}
            n_e_o_outputs = model(**n_e_o) 
            n_s = {key: n_s[key].cuda(device) for key in n_s}
            n_s_outputs = model(**n_s)   
            # print(n_e_o_outputs.logits.shape)

            #---------------------------------------stressful emotion contrast---------------------------------------#

            l_pos = torch.einsum('nc,nc->n', [o_outputs.logits, p_e_outputs.logits]).unsqueeze(-1)
            l_neg = torch.cat([torch.einsum('nc,nc->n', [o_outputs.logits, n_e_outputs.logits]).unsqueeze(-1),
            torch.einsum('nc,nc->n', [o_outputs.logits, n_e_o_outputs.logits]).unsqueeze(-1)],dim=1)
            # print(l_pos.shape)
            # print(l_neg.shape)
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= T
            # print(logits.shape)
            # exit()
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(device)
            loss= criterion(logits,labels)

            #---------------------------------------stressor contrast---------------------------------------#
            l_pos_2 = torch.einsum('nc,nc->n', [o_outputs.logits, p_s_outputs.logits]).unsqueeze(-1)
            l_neg_2 = torch.einsum('nc,nc->n', [o_outputs.logits, n_s_outputs.logits]).unsqueeze(-1)
            # l_neg_2 = torch.einsum('nc,ck->nk', [o_outputs.logits, n_s_outputs.logits.t()])
            # print(l_pos.shape)
            # print(l_neg.shape)
            logits_2 = torch.cat([l_pos_2, l_neg_2], dim=1)
            logits_2 /= T
            labels_2 = torch.zeros(logits_2.shape[0], dtype=torch.long).cuda(device)
            loss_2 = criterion(logits_2,labels_2)
            # print(list(o_outputs.keys()))
            # loss = outputs.loss
            # logits = outputs.logits

            #---------------------------------------prototypical contrast---------------------------------------#
            if cluster_result is not None:
                # proto_labels = []
                # proto_logits = []
                loss_proto = 0
                for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                    # get positive prototypes
                    pos_proto_ids = im2cluster[index]
                    pos_prototypes = prototypes[pos_proto_ids]
                    # print(pos_proto_id.shape)
                    # print(pos_prototypes.shape)
                    # exit()

                    # sample negative prototypes
                    all_proto_id = [i for i in range(im2cluster.max()+1)]
                    # print(all_proto_id)
                    neg_proto_ids = torch.zeros(1).cuda(device)
                    neg_prototypes = torch.zeros(1,768).cuda(device)
                    for id in pos_proto_ids:
                        # print(id.tolist())
                        neg_proto_id = set(all_proto_id)-set([id.tolist()])
                        neg_proto_id = random.sample(neg_proto_id,r)
                        # print(id,neg_proto_id)
                        neg_prototype = prototypes[neg_proto_id]
                        # print(neg_prototype)
                        neg_proto_ids = torch.cat([neg_proto_ids,torch.LongTensor(neg_proto_id).cuda(device)],dim=0)
                        neg_prototypes = torch.cat([neg_prototypes,neg_prototype],dim=0)
                    neg_prototypes = neg_prototypes[1:,:]
                    neg_proto_ids = neg_proto_ids[1:].type(torch.long)
                    # print(neg_proto_ids)
                    # print(neg_prototypes.shape)
                    # exit()

                    # compute prototypical logits
                    l_pos_proto = torch.einsum('nc,nc->n', [o_outputs.logits, pos_prototypes]).unsqueeze(-1)
                    l_neg_proto = torch.einsum('nc,nc->n', [o_outputs.logits, neg_prototypes]).unsqueeze(-1)
                    logits_proto = torch.cat([l_pos_proto,l_neg_proto], dim = 1)
                    # print(logits_proto.shape)
                    # exit()

                    # targets for prototype assignment
                    labels_proto = torch.zeros(logits_proto.shape[0], dtype=torch.long).cuda(device)
                    # print(labels_proto.shape)

                    # scaling temperatures for the selected prototypes
                    pos_density = density[pos_proto_ids].unsqueeze(-1)
                    neg_density = density[neg_proto_ids].unsqueeze(-1)
                    temp_proto = torch.cat([pos_density,neg_density],dim=1)
                    logits_proto /= temp_proto
                    # print(pos_density.shape)
                    # print(neg_density.shape)
                    # print(temp_proto.shape)
                    # print(logits_proto)
                    # proto_labels.append(labels_proto)
                    # proto_logits.append(logits_proto)
                    loss_proto += criterion(logits_proto, labels_proto)
                    # exit()
                loss_proto /= len(num_cluster)
            #---------------------------------------joint loss---------------------------------------#

            # loss=(1-lam)*loss+lam*loss_2
            loss = (1-lam)*(loss+loss_2)+lam*loss_proto
            # loss=loss_2
            print (epoch, loss)
            loss.backward()
            optimizer.step()
        #     break
        # break
    
    # exit()
    model.save_pretrained(saved_name)
    tokenizer.save_pretrained(saved_name)

    # model_to_save = model.module if hasattr(model, 'module') else model
    # #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
    # output_model_file = os.path.join('/home/ubuntu/caolei/transformer/bert-base-chinese/bert_xin.bin')
    
    # torch.save(model_to_save.state_dict(), output_model_file)



def get_word_list():
    file = open('/home/ubuntu/Jax/transformer/bert-base-chinese/vocab.txt','r', encoding='utf8').readlines()
    word_list = []
    # for x in file:
    #     # print (len(x.split()))
    #     if len(x.split()) == 301:
    #         word_list.append(x.split()[0])
    #     else:
    #         word_list.append(' ')
    for x in file:
        print(x.split(),len(x.split()))
        if len(x.split()) != 0:
            word_list.append(x.split()[0])

    # print (word_list, len(word_list))
    return word_list

def generate_embeddings():
    word_list = get_word_list()

    # names = ['bert_saved_30k_large', 'bert-base-chinese','bert_saved_5k', 'bert_saved_10k','bert_saved_20k','bert_saved_40k','bert_saved_80k']
    names = ['cluster_lam_0.6_epoch_3_new']
    # names = ['bert-base-chinese']
    for model_name in names:
        tokenizer = BertTokenizer.from_pretrained('./XIN/'+model_name)
        model = BertForSequenceClassification.from_pretrained('./XIN/'+model_name)
        model.cuda(device)
        # file = open('data/embeddings_for_xin/'+model_name + '.txt', 'w+', encoding='utf8') 
        file = open('./XIN/'+model_name + '.txt', 'w+', encoding='utf8') 
        for index, word in tqdm(enumerate(word_list)):

            sentence = tokenizer(word, truncation=True, padding=True)
            sentence = {key: sentence[key] for key in sentence.keys()}
            # print (sentence)

            input_ids = torch.tensor([sentence['input_ids']]).cuda(device)
            attention_mask = torch.tensor([sentence['attention_mask']]).cuda(device)
   
            outputs = model(input_ids, attention_mask=attention_mask,output_hidden_states=True)

            # print (len(outputs['hidden_states']))
            all_hidden = None
            for x in outputs['hidden_states']:
                if all_hidden == None:

                    all_hidden = x
                else:
                    all_hidden = torch.cat([all_hidden,x], dim=0)

            # print (all_hidden.shape)
            all_hidden = all_hidden[1:]
            emb = all_hidden.mean(axis=0,keepdim=False)
            emb = emb.mean(axis=0,keepdim=False)
            # print (emb.shape)
            emb = emb.detach().cpu().numpy()
            emb = emb.tolist()
            emb = str(emb)[1:-1].replace(',',' ')
            # print (emb)
            file.write(word + ' ' + emb + '\n')
            # print (outputs['hidden_states'][-1].shape)
        file.close()
        # print (outputs['hidden_states'][-1][:,0,:])
        
        # print (outputs['hidden_states'][-1][:,0,:].shape)
        #     break
        break


def main():
    # init_bert()
    # load_data()
    # train()
    # test()
    task_adapt_bert()
    generate_embeddings()

    print ('End')


if __name__ == '__main__':
    main()