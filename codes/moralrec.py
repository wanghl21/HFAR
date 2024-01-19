# %%
import metrics_utils as metrics_utils
import process_utils as process_utils
import generator_utils as generator_utils
from importlib import reload
from metrics_utils import acc as acc_func
from torch.utils.data import Dataset, DataLoader
import os
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import sys

# %%
from process_utils import *
from generator_utils import *
from model_utils import *
from metrics_utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MAX_SENTENCE = 30
MAX_ALL = 50

# %%

if len(sys.argv) < 3:
    print("Please provide parameters!")
    sys.exit(1)
alpha = float(sys.argv[2])
print(f"alpha is：{alpha}")
method = sys.argv[1]
print(f"method is：{method}")

# %%
# Path
data_root_path = 'MIND_Small-Release'
embedding_path = 'MIND_Small-Release/embedding_path/'

# %%
# read news content
news, news_index, category_dict, subcategory_dict, word_dict = read_news(
    data_root_path, 'docs.tsv')
news_title, news_vert, news_subvert = get_doc_input(
    news, news_index, category_dict, subcategory_dict, word_dict)


# %%
# read news morality
news_qua_dict = read_news_morality(
    data_root_path, 'docs_morality_fators.tsv', news_index)
news_qua = np.asarray(news_qua_dict)
news_qua.shape
news_title.shape

# %%
# read session and parse
train_session = read_clickhistory(data_root_path, 'train_sam2.tsv', news_index)
test_session = read_clickhistory(data_root_path, 'val_sam2.tsv', news_index)
train_user = parse_user(train_session, news_index)
test_user = parse_user(test_session, news_index)

# %%
npratio = 1
train_sess, train_user_id, train_label = get_train_input(
    train_session, npratio, news_index)
test_impressions, test_userids = get_test_input(test_session, news_index)
train_generator = get_hir_train_generator(
    news_title, train_user['click'], train_user_id, train_sess, train_label, 32)


# %%
# Data Process
NUM_CLASSES = [2, 2, 2]
NUM_CLASS = 3


class RecContrastiveTextDataset(Dataset):
    def __init__(self, news_titles, qua_labels, clicked_news, user_ids, sess_ids, rec_labels):
        self.news_titles = news_titles
        self.qua_labels = qua_labels

        self.label_to_texts = {label: []
                               for label in set(tuple(list(range(0, NUM_CLASS+1))))}
        for text, label in zip(news_titles, qua_labels):
            label = np.sum(label)
            self.label_to_texts[label].append(text)

        self.clicked_news = clicked_news
        self.user_ids = user_ids
        self.sess_ids = sess_ids
        self.rec_labels = rec_labels

    def __len__(self):
        return len(self.sess_ids)

    def __get_news(self, docids):
        news_title = self.news_titles[docids]
        return news_title

    def __get_news_qua(self, docids):

        news_title = self.news_titles[docids]
        news_qua_labels = self.news_titles[docids]

        return news_title, news_qua_labels

    def __get_news_cont(self, docids):

        neg_news_titles = []
        cont_labels = []
        news_titles = []
        news_qua_labels = []
        for docid in docids:
            news_title = self.news_titles[docid]
            news_qua_label = self.qua_labels[docid]
            news_titles.append(news_title)
            news_qua_labels.append(news_qua_label)

            label = int(np.sum(news_qua_label) > 0)

            if (label == 0):
                negative_label = random.choice(
                    list(range(1, len(self.qua_labels[docids]))))
                neg_news_title = random.choice(
                    self.label_to_texts[negative_label])
                cont_label = np.array([1])
            else:
                negative_label = 0
                cont_label = np.array([0])
                neg_news_title = random.choice(
                    self.label_to_texts[negative_label])
            neg_news_titles.append(neg_news_title)
            cont_labels.append(cont_label)
        return np.array(news_titles), np.array(neg_news_titles), np.array(news_qua_labels), np.array(cont_labels)

    def __getitem__(self, idx):

        sess_id = self.sess_ids[idx]

        news_titles, neg_news_titles, news_qua_labels, cont_labels = self.__get_news_cont(
            sess_id)
        user_id = self.user_ids[idx]
        clicked_ids = self.clicked_news[user_id]
        user_title = self.__get_news(clicked_ids)

        rec_label = self.rec_labels[idx].argmax(axis=-1)
        return (news_titles, neg_news_titles, news_qua_labels, cont_labels, user_title, rec_label)


# %%

train_dataset = RecContrastiveTextDataset(
    news_title, news_qua, train_user['click'], train_user_id, train_sess, train_label)
len(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# %%
# load embedding
title_word_embedding_matrix, have_word = load_matrix(embedding_path, word_dict)


class NewsEncoder(nn.Module):
    def __init__(self, news_encoder_mode):
        super(NewsEncoder, self).__init__()
        self.mode = news_encoder_mode

        weight = torch.FloatTensor(title_word_embedding_matrix).cuda()
        self.embedding_layer = nn.Embedding.from_pretrained(
            weight, freeze=False)

        self.cnn = nn.Conv1d(300, 256, 3, padding=1)

        self.sa = MultiHeadAttention(300, 16, 16, 16)
        self.dropout = nn.Dropout(0.2)
        self.att = AttentionPooling(256, 256, 0.2)

    def forward(self, x):
        '''
            x: batch_size, history_len, emb_dim
            xmask: batch_size, history_len
            user_subcate_mask: batch_size, history_len, subcate_len
        '''
        word_emb = self.embedding_layer(x)
        word_emb = self.dropout(word_emb)
        if self.mode == 'CNN':
            word_emb = word_emb.permute((0, 2, 1))
            word_vecs = self.cnn(word_emb)
            word_vecs = word_vecs.permute((0, 2, 1))

        elif self.mode == 'SelfAtt':
            word_vecs = self.sa(word_emb, word_emb, word_emb)
        news_vec = self.att(word_vecs)

        return news_vec


class UserEncoder(nn.Module):
    def __init__(self, user_encoder_mode, use_mask):
        super(UserEncoder, self).__init__()

        self.use_mask = use_mask
        self.mode = user_encoder_mode

        self.sa = MultiHeadAttention(256, 16, 16, 16)

        self.att = AttentionPooling(256, 256, 0.2)
        self.dropout = nn.Dropout(0.2)

        self.click_embedding = nn.Embedding(51, 128)
        self.fc = nn.Linear(128, 1)
        self.GRU = nn.GRU(256, 256, 1, batch_first=True)
        self.default_user = nn.Parameter(torch.empty(
            1, 256).uniform_(-1/np.sqrt(256), 1/np.sqrt(256))).type(torch.FloatTensor)
        # torch.nn.init.constant_(self.default_user, 0.)

    def forward(self, x):
        '''
            x: batch_size, history_len, emb_dim
            xmask: batch_size, history_len
            user_subcate_mask: batch_size, history_len, subcate_len
        '''

        mask = None
        if not self.use_mask:
            mask = None
        if self.mode == 'SelfAtt':
            user_vecs = self.sa(x, x, x, mask)
            user_vecs = self.dropout(user_vecs)
            user_vec = self.att(user_vecs, mask)

        elif self.mode == 'Att':
            user_vecs = x
            user_vecs = self.dropout(user_vecs)
            user_vec = self.att(user_vecs, mask)
        elif self.mode == 'GRU':
            user_vecs = x  # bz,50,256
            user_vec, _ = self.GRU(user_vecs)  # bz,50,256   1,50,256
            user_vec = user_vec[:, -1, :]

        return user_vec


# %%
factor_num = 3


class NewsQuaEncoder(nn.Module):
    def __init__(self, news_encoder_mode):
        super(NewsQuaEncoder, self).__init__()
        self.mode = news_encoder_mode
        weight = torch.FloatTensor(title_word_embedding_matrix).cuda()
        self.embedding_layer = nn.Embedding.from_pretrained(
            weight, freeze=False)
        self.cnn = nn.Conv1d(300, 256, 3, padding=1)

        self.sa = MultiHeadAttention(300, 16, 16, 16)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.att = AttentionPooling(256, 256, 0.2)

    def forward(self, x):
        word_emb = self.embedding_layer(x)
        word_emb = self.dropout(word_emb)
        if self.mode == 'CNN':
            word_emb = word_emb.permute((0, 2, 1))
            word_vecs = self.cnn(word_emb)
            word_vecs = word_vecs.permute((0, 2, 1))

        elif self.mode == 'SelfAtt':
            word_vecs = self.sa(word_emb, word_emb, word_emb)
            word_vecs = self.relu(word_vecs)
        news_vec = self.att(word_vecs)

        return news_vec


# %%
# model for multiDim classifer
NUM_DIMENSIONS = 3
NUM_CLASSES = [2, 2, 2]


class MultiDimTextClassifier(nn.Module):
    def __init__(self, news_encoder_mode, hidden_dim, num_classes):
        super(MultiDimTextClassifier, self).__init__()
        self.news_encoder = NewsQuaEncoder(news_encoder_mode)

        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for n_classes in num_classes
        ])

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        news_qua_embx = self.news_encoder(x)
        class_outputs = [classifier(news_qua_embx)
                         for classifier in self.classifiers]

        news_qua_embc = self.news_encoder(c)
        Sx = self.fc(news_qua_embx)
        Sc = self.fc(news_qua_embc)
        cont_outputs = self.sigmoid(Sx-Sc)
        return news_qua_embx, class_outputs, Sx, cont_outputs


# %%
# model for recommend

class NewsRecommend(nn.Module):
    def __init__(self, news_encoder_mode, user_encoder_mode, use_mask, newsqua_encoder_mode, hidden_dim, num_classes):
        super(NewsRecommend, self).__init__()
        self.news_encoder = NewsEncoder(news_encoder_mode)
        self.user_encoder = UserEncoder(user_encoder_mode, use_mask)
        self.news_classifier = MultiDimTextClassifier(
            newsqua_encoder_mode, hidden_dim, num_classes)

    def forward(self, candi_title, neg_news_title, news_qua_label, cont_label, click_title, rec_label):
      # def forward(self,candi_title,candi_neg_title,click_title,click_mask,click_num,labels):

        bz = candi_title.shape[0]  # 64
        candi_title = torch.reshape(
            candi_title, (bz*(1+npratio), 30))  # 128,30
        neg_news_title = torch.reshape(
            neg_news_title, (bz*(1+npratio), 30))  # 128,30

        click_title = torch.reshape(click_title, (bz*50, 30))  # 3200,30

        candi_vecs = self.news_encoder(candi_title)  # 128,256
        click_vecs = self.news_encoder(click_title)

        candi_vecs = torch.reshape(
            candi_vecs, (bz, 1+npratio, 256))  # 64,2,256
        click_vecs = torch.reshape(click_vecs, (bz, 50, 256))  # 64,50,256

        user_vec = self.user_encoder(click_vecs)  # 64,256

        news_qua_embx, class_outputs, Sx, cont_scores = self.news_classifier(
            candi_title, neg_news_title)  # cont_scores 128,1

        rec_scores = torch.bmm(
            candi_vecs, user_vec.unsqueeze(dim=-1)).squeeze(-1)  # 64,2

        cont_scores = torch.reshape(cont_scores, (bz, 1+npratio))  # 64,2
        Sx = torch.reshape(Sx, (bz, 1+npratio))

        final_scores = rec_scores + alpha * Sx

        return class_outputs, cont_scores, rec_scores, final_scores


# %%

if "LSTUR" in method:
    news_encoder_mode = 'CNN'
    user_encoder_mode = 'GRU'
    newsqua_encoder_mode = news_encoder_mode
elif "NAML" in method:
    news_encoder_mode = 'CNN'
    user_encoder_mode = 'Att'
    newsqua_encoder_mode = news_encoder_mode
elif "NRMS" in method:
    news_encoder_mode = 'SelfAtt'
    user_encoder_mode = 'SelfAtt'
    newsqua_encoder_mode = news_encoder_mode
elif "KRED" in method:
    news_encoder_mode = 'SelfAtt'
    user_encoder_mode = 'Att'
    newsqua_encoder_mode = news_encoder_mode

else:
    print("error!")
    exit()
use_mask = False

model = NewsRecommend(news_encoder_mode, user_encoder_mode, use_mask,
                      newsqua_encoder_mode, hidden_dim=256, num_classes=NUM_CLASSES)
model = model.cuda()
model

# %%
class_weights = []
for i in range(NUM_CLASS):
    class_count = sum(news_qua[:, i])
    class_counts = [news_qua[:, i].shape[0]-class_count, class_count]
    class_counts = [sum(class_counts) / c for c in class_counts]
    class_weight = torch.FloatTensor(
        [c / sum(class_counts) for c in class_counts]).cuda()
    class_weights.append(class_weight)
class_weights

# %%
# model evaluate

path = "/data/Quality_rec/"
num_epochs = 7


def evaluate_all_metrics(model):
    model = model.eval()
    with torch.no_grad():
        news_encoder = model.news_encoder
        user_encoder = model.user_encoder
        qua_news_encoder = model.news_classifier.news_encoder
        fc = model.news_classifier.fc

        news_scorings = get_news_scoring(news_encoder, news_title)
        user_scorings = get_user_scoring(
            user_encoder, news_scorings, test_user['click'][:20000])
        news_qua_scorings = get_news_qua_scoring(
            qua_news_encoder, fc, news_title)
        np.save(path+"scoring/"+method+"/"+str(num_epochs) +
                str(alpha)+"our_user_scoring.npy", user_scorings)
        np.save(path+"scoring/"+method+"/"+str(num_epochs) +
                str(alpha)+"our_news_scoring.npy", news_scorings)
        np.save(path+"scoring/"+method+"/"+str(num_epochs) +
                str(alpha)+"our_news_qua_scorings.npy", news_qua_scorings)

        eval_rank = evaluate_model_qua_ours(
            test_impressions[:20000], user_scorings, news_scorings, news_qua_scorings, news_qua, alpha)
        print(eval_rank)
        recall_rank = evaluate_recall_model_qua_ours(
            test_impressions[:20000], user_scorings, news_scorings, news_qua_scorings, news_qua, alpha)
        print(recall_rank)

        with open('Validation/' + method+'/alpha' + str(alpha) + method + '_ours.json', 'a+') as f:
            # f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())+'\n')
            f.write("eval_rank"+'\n')
            s = json.dumps(eval_rank)
            f.write(s+'\n')
            f.write("recall_rank"+'\n')
            s = json.dumps(recall_rank)
            f.write(s+'\n')


# %%
# model training


optimizer = optim.Adam(model.parameters(), lr=0.0001)  # lr=0.0001
criterion = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()  # nn.functional.binary_cross_entropy #

# training setting

loss = 0.0

w_class = 1
w_cont = 1
w_rec = 1

for epoch in range(num_epochs):
    NUM = len(train_dataloader)
    cnt = 0

    all_loss = 0.0
    acc_cont = 0.0
    auc_cont = 0.0
    acc_rec = 0.0
    acc_class = 0.0

    for candi_title, neg_news_title, news_qua_label, cont_label, click_title, rec_label in train_dataloader:
        candi_title, neg_news_title, news_qua_label, cont_label, click_title, rec_label = candi_title.cuda(
        ), neg_news_title.cuda(), news_qua_label.cuda(), cont_label.cuda(), click_title.cuda(), rec_label.cuda()
        optimizer.zero_grad()
        class_outputs, cont_scores, rec_scores, final_scores = model(
            candi_title, neg_news_title, news_qua_label, cont_label, click_title, rec_label)
        class_weight = [1, 1, 1]
        loss_class = 0.0
        news_qua_label = torch.reshape(
            news_qua_label, (news_qua_label.shape[0]*(1+npratio), 3))  # 128,3
        for i in range(NUM_CLASS):
            class_criterion = nn.CrossEntropyLoss(weight=class_weights[i])
            y_label = news_qua_label[:, i].to(torch.int64)

            # anchor_label[:,i]
            loss_class += class_weight[i] * \
                class_criterion(class_outputs[i], y_label)

            y_label_on_cpu = y_label.detach().cpu()
            outputs_on_cpu = class_outputs[i].detach().cpu()
            acc_score = acc_func(y_label_on_cpu, outputs_on_cpu)
            acc_class += acc_score

        cont_scores = torch.reshape(
            cont_scores, (cont_scores.shape[0]*(1+npratio), 1))  # 128,1
        cont_label = torch.reshape(
            cont_label, (cont_label.shape[0]*(1+npratio), 1))  # 128,1

        loss_cont = criterion_bce(
            cont_scores, torch.abs(cont_label.float()-10**(-5)))
        loss_rec = criterion(final_scores, rec_label)
        loss = w_class * loss_class + w_cont * loss_cont + w_rec * loss_rec

        cont_outputs_on_cpu = cont_scores.detach().cpu()
        cont_label_on_cpu = cont_label.detach().cpu()
        auc_cont += roc_auc_score(cont_label_on_cpu,
                                  cont_outputs_on_cpu)
        acc_cont += accuracy_score(cont_label_on_cpu,
                                   cont_outputs_on_cpu > 0.5)
        acc_rec += acc_func(rec_label, final_scores)
        loss.backward()
        optimizer.step()
        all_loss += loss
        per = cnt/NUM
        cnt += 1
        Log(all_loss/cnt, acc_rec/cnt, per)
    print("\n")
    print("acc_class", acc_class/cnt/3)
    print("acc_cont", acc_cont/cnt)
    print("auc_cont", auc_cont/cnt)
    print("acc_rec", acc_rec/cnt)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# %%
evaluate_all_metrics(model)


# %%


def load_evaluate_all_metrics_ours():

    news_scorings = np.load(path+"scoring/"+method +
                            "/"+str(num_epochs)+"0.0our_news_scoring.npy")
    user_scorings = np.load(path+"scoring/"+method +
                            "/"+str(num_epochs)+"0.0our_user_scoring.npy")
    news_qua_scorings = np.load(
        path+"scoring/"+method+"/"+str(num_epochs)+"0.0our_news_qua_scorings.npy")

    eval_rank = evaluate_model_qua_ours(
        test_impressions[:20000], user_scorings, news_scorings, news_qua_scorings, news_qua, alpha)
    print(eval_rank)
    recall_rank = evaluate_recall_model_qua_ours(
        test_impressions[:20000], user_scorings, news_scorings, news_qua_scorings, news_qua, alpha)
    print(recall_rank)

    with open('Validation/' + method+'/0.0alpha' + str(alpha) + method + '_ours.json', 'a+') as f:
      # f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())+'\n')
        f.write("eval_rank"+'\n')
        s = json.dumps(eval_rank)
        f.write(s+'\n')
        f.write("recall_rank"+'\n')
        s = json.dumps(recall_rank)
        f.write(s+'\n')
# load_evaluate_all_metrics_ours()
