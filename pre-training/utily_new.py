from unittest import result
from random import randint
import faiss
import torch
import torch.nn as nn
import numpy as np

def mask(text,lexcion):
    flag=0
    num=0
    masked_word=[]
    for k in range(len(lexcion)):
        if lexcion[k] in text:
            repeat_num=text.count(lexcion[k])
            text=text.replace(lexcion[k],"[MASK]")
            flag=1
            # num+=1
            for i in range(repeat_num):
                masked_word.append(lexcion[k])
                num+=1
    return text,flag,num,masked_word

def sampling_positive_word(lexcion):
    index = randint(0, len(lexcion) - 1)
    return lexcion[index]
    
def replace(text,lexcion1,lexcion2):
    flag=0
    for k in range(len(lexcion1)):
        if lexcion1[k] in text:
            text=text.replace(lexcion1[k],sampling_positive_word(lexcion2))
            flag=1
    return text,flag

def random_mask(text,masked_word,index):
    char="[MASK]"
    # begin_index=0
    # end_index=0
    # masked_index=[]
    # for k in range(len(masked_word)):
    #     if masked_word[k] in text:
    #         begin_index=text.find(masked_word[k])
    #         end_index=begin_index+len(avoid_lexcion[k])
    # print(text)
    for i in range(10000000):
        pos = randint(0, len(text)-len(masked_word[index])-1)
        # if i in range(begin_index,end_index):
        #     continue
        if text[pos]!="[" and text[pos]!="M" and text[pos]!="A" and text[pos]!="S" and text[pos]!="K" and text[pos]!="]" and text[pos+len(masked_word[index])]!="[" and text[pos+len(masked_word[index])]!="M" and text[pos+len(masked_word[index])]!="A" and text[pos+len(masked_word[index])]!="S" and text[pos+len(masked_word[index])]!="K" and text[pos+len(masked_word[index])]!="]":
            masked_text = "".join((text[:pos], char, text[pos+len(masked_word[index]):]))
            flag=1
            for k in range(len(masked_word)):
                if masked_word[k] not in masked_text:
                    flag=0
                    # print(masked_text,masked_word[k])
            if flag==1:
                return masked_text
    return text
def stressful_emotion_masked(data):
    stressful_emotion_file=open("./data/lexcion/stress_emotion_word.txt",'r')
    stressful_emotion=stressful_emotion_file.read().split("\n")
    positive_emotion_file=open("./data/lexcion/positive_emotion_word.txt",'r')
    positive_emotion=positive_emotion_file.read().split("\n")    
    stressor_file=open("./data/lexcion/stressor.txt",'r')
    stressor=stressor_file.read().split("\n")
    # print(stressor)
    # print(stressful_emotion)
    original_data=[]
    stressor_positive_data=[]
    emotion_positive_data=[]
    emotion_negative_data=[]
    emotion_replace_negative_data=[]
    stressor_negative_data=[]
    for j in range(len(data)):
        emotion_replace_negative_sentence,flag=replace(data[j],stressful_emotion,positive_emotion)
        emotion_negative_sentence,flag,emotion_masked_num,masked_emotion_word=mask(data[j],stressful_emotion)
        stressor_negative_sentence,flag,stressor_masked_num,masked_stressor_word=mask(data[j],stressor)
        stressor_positive_sentence=data[j]
        emotion_positive_sentence=data[j]
        for i in range(stressor_masked_num):
            # print(masked_stressor_word)
            stressor_positive_sentence=random_mask(stressor_positive_sentence,masked_stressor_word,i)
        for i in range(emotion_masked_num):
            emotion_positive_sentence=random_mask(emotion_positive_sentence,masked_emotion_word,i)
        if flag==1:
            emotion_replace_negative_data.append(emotion_replace_negative_sentence)
            original_data.append(data[j])
            stressor_positive_data.append(stressor_positive_sentence)
            emotion_positive_data.append(emotion_positive_sentence)
            emotion_negative_data.append(emotion_negative_sentence)
            stressor_negative_data.append(stressor_negative_sentence)
    # print(data)
    # for i in range(len(positive_data)):
    #     print(original_data[i])
    #     print(positive_data[i])
    #     print(emotion_negative_data[i])
    #     print(emotion_replace_negative_data[i])
    #     print(stressor_negative_data[i])
    # print(len(negative_data))
    stressful_emotion_file.close()
    positive_emotion_file.close()
    stressor_file.close()
    # return original_data,positive_data,emotion_negative_data,stressor_negative_data
    return original_data,emotion_positive_data,stressor_positive_data,emotion_negative_data,emotion_replace_negative_data,stressor_negative_data

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    # for seed, num_cluster in enumerate(args.num_cluster):
    for seed, num_cluster in enumerate([5]):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        # cfg.device = args.gpu
        cfg.device = 7
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
        density = 0.007*density/density.mean()  #scale the mean to temperature
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

if __name__ == '__main__':
    #----------------------------------------------------check the positve and negative examples------------------------------------------#
    # stress_data = open('data/tweet_for_pretrain_clean.txt','r', encoding='utf8').readlines()
    # stress_data = [x.strip() for x in stress_data]
    # original_data,emotion_positive_data,stressor_positive_data,emotion_negative_data,emotion_replace_negative_data,stressor_negative_data=stressful_emotion_masked(stress_data)
    # # print(emotion_negative_data,stressor_negative_data)
    # for i in range(len(original_data)):
    #     try:
    #         if len(original_data[i])==0:
    #             print(original_data[i])
    #             print(stressor_negative_data[i])
    #             print(stressor_positive_data[i])
    #             print(emotion_negative_data[i])
    #             print(emotion_replace_negative_data[i])
    #             print(emotion_positive_data[i])
    #     except:
    #         print(original_data[i])
    #         print(stressor_negative_data[i])
    #         print(stressor_positive_data[i])
    #         print(emotion_negative_data[i])
    #         print(emotion_replace_negative_data[i])
    #         print(emotion_positive_data[i])
    #----------------------------------------------------------------------------------------------------------------------#    
    #     if len(stressor_negative_data[i].replace("[MASK]",""))!=len(stressor_positive_data[i].replace("[MASK]","")):
    #         print(stressor_negative_data[i])
    #         print(stressor_positive_data[i])
    #         print(len(stressor_negative_data[i].replace("[MASK]","")),len(stressor_positive_data[i].replace("[MASK]","")))
    #     if len(emotion_negative_data[i].replace("[MASK]",""))!=len(emotion_positive_data[i].replace("[MASK]","")):
    #         print(emotion_negative_data[i])
    #         print(emotion_positive_data[i])
    #         print(len(emotion_negative_data[i].replace("[MASK]","")),len(emotion_positive_data[i].replace("[MASK]","")))
    #     if emotion_negative_data.count("[MASK]")!=emotion_positive_data.count("[MASK]"):
    #         print(1)
    #-----------------------------------calculate average-----------------------------------------#
    # num=0
    # for i in range(len(emotion_negative_data)):
    #     num+=emotion_negative_data[i].count("[MASK]")
    # print(num,num/len(emotion_negative_data))
    # num=0
    # for i in range(len(stressor_negative_data)):
    #     num+=stressor_negative_data[i].count("[MASK]")
    # print(num,num/len(stressor_negative_data))
    #-----------------------------------------------------------------------------------------------#
    #-------------------------------------test for run_kmeans funciton-------------------------------#
    x = np.random.rand(1000,768).astype('float32')
    result = run_kmeans(x,5)
    print(result)