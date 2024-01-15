import numpy as np
import os
import random
from nltk.tokenize import word_tokenize
from generator_utils import *
MAX_SENTENCE = 30
MAX_ALL = 50


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1), ratio)
    else:
        return random.sample(nnn, ratio)


def shuffle(pn, labeler, pos, npratio):
    index = np.arange(pn.shape[0])
    pn = pn[index]
    labeler = labeler[index]
    pos = pos[index]

    for i in range(pn.shape[0]):
        index = np.arange(npratio+1)
        pn[i, :] = pn[i, index]
        labeler[i, :] = labeler[i, index]
    return pn, labeler, pos


def read_news(path, filenames):
    news = {}
    category = []
    subcategory = []
    news_index = {}
    index = 1
    word_dict = {}
    word_index = 1
    with open(os.path.join(path, filenames)) as f:
        lines = f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id, vert, subvert, title = splited[0:4]
        news_index[doc_id] = index
        index += 1
        category.append(vert)
        subcategory.append(subvert)
        title = title.lower()
        title = word_tokenize(title)
        news[doc_id] = [vert, subvert, title]
        for word in title:
            word = word.lower()
            if not (word in word_dict):
                word_dict[word] = word_index
                word_index += 1
    category = list(set(category))
    subcategory = list(set(subcategory))
    category_dict = {}
    index = 1
    for c in category:
        category_dict[c] = index
        index += 1
    subcategory_dict = {}
    index = 1
    for c in subcategory:
        subcategory_dict[c] = index
        index += 1
    return news, news_index, category_dict, subcategory_dict, word_dict


def get_doc_input(news, news_index, category, subcategory, word_dict):
    news_num = len(news)+1
    news_title = np.zeros((news_num, MAX_SENTENCE), dtype='int32')
    news_vert = np.zeros((news_num,), dtype='int32')
    news_subvert = np.zeros((news_num,), dtype='int32')
    for key in news:
        vert, subvert, title = news[key]
        doc_index = news_index[key]
        news_vert[doc_index] = category[vert]
        news_subvert[doc_index] = subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE, len(title))):
            news_title[doc_index, word_id] = word_dict[title[word_id].lower()]

    return news_title, news_vert, news_subvert


def load_matrix(embedding_path, word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1, 300))
    have_word = []
    with open(os.path.join(embedding_path, 'glove.840B.300d.txt'), 'rb') as f:
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            l = l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index] = np.array(tp)
                have_word.append(word)
    return embedding_matrix, have_word


def read_clickhistory(path, filename, news_index):

    lines = []
    userids = []
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()

    sessions = []
    for i in range(len(lines)):
        _, uid, eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clicks = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click, pos, neg])
    return sessions


def parse_user(session, news_index):
    user_num = len(session)
    user = {'click': np.zeros((user_num, MAX_ALL), dtype='int32'), }
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg = session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) > MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click = [0]*(MAX_ALL-len(click)) + click

        user['click'][user_id] = np.array(click)
    return user


def get_train_input(session, npratio, news_index):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs = sess
        # 每一个用户的pos记录随机sample npratio个neg 然后带上uid组成一条sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = newsample(negs, npratio)  # 随机采样npratio个负样本
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)
    # print(len(user_id))
    sess_all = np.zeros((len(sess_pos), 1+npratio), dtype='int32')
    label = np.zeros((len(sess_pos), 1+npratio))
    # 下面是把 pos['Nxxx']转成news index
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id, 0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id, index] = news_index[neg]
            index += 1
        # index = np.random.randint(1+npratio)
        label[sess_id, 0] = 1
    user_id = np.array(user_id, dtype='int32')

    return sess_all, user_id, label


def get_test_input(session, news_index):

    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels': [],
               'docs': []}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)

    userid = np.array(userid, dtype='int32')

    return Impressions, userid,


def get_qua_input(news_train_index, news_title, news_qua):
    train_data = np.zeros((len(news_train_index), MAX_SENTENCE), dtype='int32')
    label = np.zeros((len(news_train_index), news_qua.shape[1]), dtype='int32')

    for index in range(len(news_train_index)):
        train_data[index, :] = news_title[news_train_index[index], :]
        label[index, :] = news_qua[news_train_index[index], :]

    return train_data, label


def get_news_scoring(news_encoder, news_title):
    print("get_news_scoring...")
    bz = 64
    news_scorings = []
    for i in range(int(np.ceil(len(news_title)/bz))):
        start = bz*i
        ed = bz*(i+1)
        if ed > len(news_title):
            ed = len(news_title)
        data = news_title[start:ed]
        data = torch.LongTensor(data).cuda()
        # print(data.shape)
        ns = news_encoder(data)
        ns = ns.detach().to('cpu').numpy()
        news_scorings.append(ns)
    news_scorings = np.concatenate(news_scorings, axis=0)
    return news_scorings


def get_news_qua_scoring(news_encoder, fc, news_title):
    print("get_news_scoring...")
    bz = 64
    news_qua_scorings = []
    for i in range(int(np.ceil(len(news_title)/bz))):
        start = bz*i
        ed = bz*(i+1)
        if ed > len(news_title):
            ed = len(news_title)
        data = news_title[start:ed]
        data = torch.LongTensor(data).cuda()
        # print(data.shape)
        ns = news_encoder(data)
        # 这里是fc以后的结果
        ns = fc(ns)
        ns = ns.detach().to('cpu').numpy()
        news_qua_scorings.append(ns)
    news_qua_scorings = np.concatenate(news_qua_scorings, axis=0)
    return news_qua_scorings


def get_user_scoring(user_encoder, news_scoring, user_click):
    print("get_user_scoring...")
    user_generator = get_hir_user_generator(news_scoring, user_click, 32)
    user_scorings = []
    cnt = 0
    for data in user_generator:
        us = user_encoder(data)
        us = us.detach().to('cpu').numpy()
        user_scorings.append(us)
        cnt += 1
        if (cnt == len(user_generator)):
            break
    user_scorings = np.concatenate(user_scorings, axis=0)
    return user_scorings


def get_ori_user_scoring(user_encoder, news_scoring, user_click):
    print("get_user_scoring...")
    user_generator = get_hir_user_generator_ori(news_scoring, user_click, 32)
    user_scorings = []
    cnt = 0
    for data in user_generator:
        us = user_encoder(*data)
        us = us.detach().to('cpu').numpy()
        user_scorings.append(us)
        cnt += 1
        if (cnt == len(user_generator)):
            break
    user_scorings = np.concatenate(user_scorings, axis=0)
    return user_scorings


def compute_doc_sim(news_scoring):
    print("compute_doc_sim...")
    z = np.sqrt((news_scoring**2).sum(axis=-1)
                ).reshape((news_scoring.shape[0], 1))
    norm_news_scoring = news_scoring/z

    random_index = np.random.permutation(len(norm_news_scoring))
    rand_ns = norm_news_scoring[random_index]

    scores = (norm_news_scoring*rand_ns).sum(axis=-1).mean()

    return scores


def Log(loss, acc, per, K=2):
    NUM = 100//K
    num1 = int(NUM*per)
    num2 = NUM-num1

    print("\r loss=%.3f acc=%.4f  %s%s  %.2f%s" %
          (loss, acc, '>' * num1, '#'*num2, 100*per, '%'), flush=True, end='')


def read_news_quality(path, filenames, news_index):
    print("read news quality...")
    with open(os.path.join(path, filenames)) as f:
        lines = f.readlines()
    news_num = len(lines)+1
    news_qua = np.zeros((news_num, 11), dtype='int32')
    for line in lines:
        splites = line.strip('\n').split('\t')
        doc_id = splites[0]
        doc_index = news_index[doc_id]
        for splite in splites:
            if ('fake news:1' in splite):
                fn = 1
            elif ('fake news:0' in splite):
                fn = 0
            if ('clickbait headlines:1' in splite):
                ch = 1
            elif ('clickbait headlines:0' in splite):
                ch = 0
            if ('gender discrimination:1' in splite):
                gd = 1
            elif ('gender discrimination:0' in splite):
                gd = 0
            if ('racial discrimination:1' in splite):
                rd = 1
            elif ('racial discrimination:0' in splite):
                rd = 0

            if ('violence:1' in splite):
                vi = 1
            elif ('violence:0' in splite):
                vi = 0

            if ('crime:1' in splite):
                cr = 1
            elif ('crime:0' in splite):
                cr = 0

            if ('pornographic tendency:1' in splite):
                pt = 1
            elif ('pornographic tendency:0' in splite):
                pt = 0
            if ('breaking news:1' in splite):
                bn = 1
            elif ('breaking news:0' in splite):
                bn = 0

            if ('news writing standards:1' in splite):
                nw = 1
            elif ('news writing standards:0' in splite):
                nw = 0
            if ('adult audience:1' in splite):
                adu = 1
            elif ('adult audience:0' in splite):
                adu = 0
            if ('adolescent audience:1' in splite):
                ado = 1
            elif ('adolescent audience:0' in splite):
                ado = 0
        news_qua[doc_index, :] = [fn, ch, gd, rd, vi, cr, pt, bn, nw, adu, ado]

    return news_qua
