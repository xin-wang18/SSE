
import torch
from dataset import Classification_Dataset,Proportion_Classification_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report,confusion_matrix
import time
import Constants
from model import Stess_classification,mullti

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    model = Stess_classification()
    # model = mullti()
    # model.cuda()
    model.to(device)
    print(model)

    print("Learning rate",Constants.lr)
    # for reddit dataset
    # train_data = Classification_Dataset(Constants.train_root)
    train_data = Proportion_Classification_Dataset(Constants.train_root,proportion=0.15)
    val_data = Classification_Dataset(Constants.valid_root)
    test_data = Classification_Dataset(Constants.test_root)

    train_dataloader = DataLoader(
        train_data,
        Constants.batch_size,
        shuffle = True,
        num_workers = Constants.num_workers
        )
    val_dataloader = DataLoader(
        val_data,
        Constants.batch_size,
        shuffle = False,
        num_workers = Constants.num_workers
        )
    test_dataloader = DataLoader(
        test_data,
        Constants.batch_size,
        shuffle = False,
        num_workers = Constants.num_workers
        )
    criterion = torch.nn.CrossEntropyLoss()
    lr = Constants.lr
    optimizer = torch.optim.Adam(model.parameters(),
        lr = lr,
        weight_decay = Constants.weight_decay)
    # writer = SummaryWriter()
    cnt=0

    sum_acc, sum_loss = 0.,0.
    sum_acc_, sum_loss_ = 0.,0.
   
    
  
    for epoch in range(Constants.max_epoch):
        for i, (input,target) in tqdm(enumerate(train_dataloader),total=len(train_data)/Constants.batch_size):
        
            input = input.to(device)
            target = target.to(device)

            # print(type(input))
            # print (type(target), target.shape)
            optimizer.zero_grad()
            # print (input1.type(),input2.type(),sentence_len.type(),sample_len.type(),target.type())
            score = model(input)
            # print ('\n iter {} score {} target {} '.format(i,score, target))
            # print ('score ',score.type(),score.shape)
            # print ('target ',target.type(),target.shape)

            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            y_ = torch.argmax(score,dim = 1).cpu().numpy()
            correct = sum(y_==target.cpu().numpy())/Constants.batch_size
            # print ('pred ',y_)
            # print ('target ',target.cpu().numpy())
            sum_acc += correct
            sum_loss += loss.data
            # sum_acc_ += correct
            # sum_loss_ += loss.data

            print("epoch {} iter {} sum correct {} loss {}".format(epoch, i,sum_acc/(i+1),sum_loss/(i+1)))
            # writer.add_scalar("/log/train_acc",sum_acc/(i+1),cnt)
            # writer.add_scalar("/log/train_loss",sum_loss/(i+1),cnt)
            # print(score,type(score))
            # if i%opt.print_freq == 0 and i>0:
            #     # y_ = torch.argmax(score,dim = 1).cpu().numpy()
            #     # correct = sum(y_==target.cpu().numpy())/opt.batch_size
            #     print("iter {} correct {} loss {}".format(i,sum_acc/opt.print_freq,sum_loss/opt.print_freq))
                
            #     # print (type(loss.data))
            #     writer.add_scalar("/log/acc",sum_acc,cnt)
            #     writer.add_scalar("/log/loss",sum_loss,cnt)
            #     sum_acc, sum_loss = 0., 0.
            cnt +=1
                # if cnt == 10:
            # print (embedding,embedding.weight)

        sum_acc, sum_loss = 0., 0.
        val(model,val_dataloader,len(val_data),Constants.batch_size,epoch)
        test(model,test_dataloader,len(test_data),Constants.batch_size,epoch)
        sum_acc_,sum_loss_=0.,0.
        # model.module.save()
        # break
    # writer.close()
        # model.save()



def val(model,dataloader, lenn,batch_size,epoch):
    model.eval()
    sum_acc = 0.
    sum_y_, sum_target = np.asarray([]),np.asarray([])
    for i, (input, target) in tqdm(enumerate(dataloader), total = lenn/batch_size):


        input = input.to(device)
        target = target.to(device)

        score = model(input)
  
        y_ = torch.argmax(score,dim = 1).cpu().numpy()
        correct = sum(y_==target.cpu().numpy())
        print ('pred ',y_, y_.shape)
        print ('target ',target.cpu().numpy())
        sum_acc += correct

        sum_y_ =  np.concatenate((sum_y_,y_),axis=0)
        sum_target = np.concatenate((sum_target,target.cpu().numpy()),axis=0)


    f1, p, r = metrics(sum_target,sum_y_)
  
    # writer.add_scalar("/log/val_acc",sum_acc/lenn,epoch)
    # writer.add_scalar("/log/val_f1",f1,epoch)
    # writer.add_scalar("/log/val_p", p, epoch)
    # writer.add_scalar("/log/val_r", r, epoch)
    print("val correct {} ".format(sum_acc/lenn))
    print("val f1 {} p {} r {} ".format(f1, p, r))
    # prefix = 'checkpoints/' + opt.model + '_'+str(epoch) +'_'
    # name = time.strftime(prefix + '%m_%d_%H:%M:%S.pth')
    # torch.save(model, name)
    model.train()

def test(model,dataloader,lenn,batch_size,epoch):
    model.eval()
    sum_acc = 0.
    sum_y_, sum_target = np.asarray([]),np.asarray([])
    for i, (input, target) in tqdm(enumerate(dataloader), total = lenn/batch_size):
      
        input = input.to(device)
        target = target.to(device)

        score = model(input)

        y_ = torch.argmax(score,dim = 1).cpu().numpy()
        correct = sum(y_==target.cpu().numpy())
        print ('pred ',y_, y_.shape)
        print ('target ',target.cpu().numpy())
        sum_acc += correct

        sum_y_ =  np.concatenate((sum_y_,y_),axis=0)
        sum_target = np.concatenate((sum_target,target.cpu().numpy()),axis=0)

    f1, p, r = metrics(sum_target,sum_y_)

    # writer.add_scalar("/log/test_acc",sum_acc/lenn,epoch)
    # writer.add_scalar("/log/test_f1",f1,epoch)
    # writer.add_scalar("/log/test_p", p,epoch)
    # writer.add_scalar("/log/test_r", r,epoch)
    print("test correct {} ".format(sum_acc/lenn))
    print("test f1 {} p {} r {}".format(f1, p, r))
    # if f1>0.6 and f1<0.7:
    #     torch.save(model.state_dict(),"./save/bert_six_class.pt")
    #     exit()
    model.train()


def metrics(y_true,y_pred):

    print('macro ',f1_score(y_true, y_pred, average='macro'))
    print(precision_score(y_true, y_pred, average='macro'))
    print(recall_score(y_true, y_pred, average='macro'))
    
    print('micro ', f1_score(y_true, y_pred, average='micro'))
    print(precision_score(y_true, y_pred, average='micro'))
    print(recall_score(y_true, y_pred, average='micro'))



    print (confusion_matrix(y_true, y_pred))
    target_names = Constants.class_folders
    # print(classification_report(y_true, y_pred, target_names=target_names))
    return f1_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro')

def main():
    start =time.time()
    train()
    end=time.time()
    print ('total time: ',end-start)
    print ('End')

if __name__ == '__main__':
    main()