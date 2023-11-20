'''
Define some constants
'''
# from Category_classification.main_bi_lstm import BATCH_SIZE


MAX_TWEET_NUM=20
# batch_size=64
batch_size=64
num_workers=0
lr=0.001
weight_decay=1e-4
max_epoch=500
train_root="./data_six_divide/train/"
valid_root="./data_six_divide/valid/"
test_root="./data_six_divide/test/"
CLASS_NUM=6
# class_folders = ['Financial state','School','Work hours or conditions','Fired at work','Unknown','Responsibility']
class_folders = ['Financial state','School','Work hours or conditions','Fired at work','Responsibility','Unknown']
# train_root="./data_two_divide/train/"
# valid_root="./data_two_divide/valid/"
# test_root="./data_two_divide/test/"
# CLASS_NUM=2
# class_folders = ['No stress','Stress']

# embedding_root="./data/mbert.txt"
# embedding_root="./data/equal_lam_0.6.txt"
# embedding_root="./data/contrasitve_lam_0.6.txt"
# embedding_root="./data/chinese-roberta.txt"
embedding_root="./data/cluster_lam_0.6_epoch_3_new.txt"
# embedding_root="./data/bert-base-chinese.txt"
# embedding_root="./data/sgns.weibo.word"
# embedding_dim=300
embedding_dim=768
max_tweet_len=128

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '[PAD]'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

Default_Dict0={
    PAD_WORD: PAD,
    UNK_WORD: UNK,
    BOS_WORD: BOS,
    EOS_WORD: EOS
}
Default_Dict={
    UNK_WORD: 0
}