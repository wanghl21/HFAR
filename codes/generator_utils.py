import numpy as np
import torch


class get_hir_qua_generator():
    def __init__(self, qua_data, label, batch_size):
        self.news_qua_emb = qua_data
        self.label = label

        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        # news_emb = self.news_emb[docids]
        news_qua_emb = self.news_qua_emb[docids]
        return news_qua_emb

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum
        # label = self.label[start:ed].argmax(axis=-1)
        label = self.label[start:ed]

        # doc_ids = self.doc_id[start:ed]
        # news_qua= self.__get_news(doc_ids)
        news_qua = self.news_qua_emb[start:ed]
        label = np.array(label, dtype='float')
        label = torch.FloatTensor(label).cuda()
        news_qua = torch.LongTensor(news_qua).cuda()
        return (news_qua, label)


class get_hir_qua_train_generator():
    def __init__(self, news_qua, clicked_news, user_id, news_id, label, batch_size):
        self.news_qua_emb = news_qua
        # self.news_emb = news_scoring
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label

        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        # news_emb = self.news_emb[docids]
        news_qua_emb = self.news_qua_emb[docids]
        return news_qua_emb

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed].argmax(axis=-1)

        doc_ids = self.doc_id[start:ed]
        title = self.__get_news(doc_ids)

        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_title = self.__get_news(clicked_ids)

        click_mask = clicked_ids > 0
        click_mask = np.array(click_mask, dtype='float32')

        click_num = click_mask.sum(axis=-1)
        click_num = np.array(click_num, dtype='int32')
        click_num = click_num.reshape((len(click_num), 1))

        label = np.array(label, dtype='int64')
        label = torch.LongTensor(label).cuda()
        click_num = torch.LongTensor(click_num).cuda()
        click_mask = torch.FloatTensor(click_mask).cuda()
        title = torch.LongTensor(title).cuda()
        user_title = torch.LongTensor(user_title).cuda()

        return (title, user_title, click_mask, click_num, label)


class get_hir_train_generator():
    def __init__(self, news_scoring, clicked_news, user_id, news_id, label, batch_size):
        self.news_emb = news_scoring
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label

        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        news_emb = self.news_emb[docids]

        return news_emb

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed].argmax(
            axis=-1)  # (32,)

        doc_ids = self.doc_id[start:ed]
        title = self.__get_news(doc_ids)  # (32,2,30)

        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_title = self.__get_news(clicked_ids)  # (32,50,30)

        click_mask = clicked_ids > 0
        click_mask = np.array(click_mask, dtype='float32')

        click_num = click_mask.sum(axis=-1)
        click_num = np.array(click_num, dtype='int32')
        click_num = click_num.reshape((len(click_num), 1))

        label = np.array(label, dtype='int64')
        label = torch.LongTensor(label).cuda()
        click_num = torch.LongTensor(click_num).cuda()
        click_mask = torch.FloatTensor(click_mask).cuda()
        title = torch.LongTensor(title).cuda()
        user_title = torch.LongTensor(user_title).cuda()

        return (title, user_title, click_mask, click_num, label)


class get_hir_user_generator():
    def __init__(self, news_emb, clicked_news, batch_size):
        self.news_emb = news_emb

        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        news_emb = self.news_emb[docids]

        return news_emb

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        clicked_ids = self.clicked_news[start:ed]
        user_title = self.__get_news(clicked_ids)

        click_mask = clicked_ids > 0
        click_mask = np.array(click_mask, dtype='float32')

        click_num = click_mask.sum(axis=-1)
        click_num = np.array(click_num, dtype='int32')
        click_num = click_num.reshape((len(click_num), 1))

        click_num = torch.LongTensor(click_num).cuda()
        click_mask = torch.FloatTensor(click_mask).cuda()
        user_title = torch.FloatTensor(user_title).cuda()

        return user_title


class get_hir_user_generator_ori():
    def __init__(self, news_emb, clicked_news, batch_size):
        self.news_emb = news_emb

        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        news_emb = self.news_emb[docids]

        return news_emb

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        clicked_ids = self.clicked_news[start:ed]
        user_title = self.__get_news(clicked_ids)

        click_mask = clicked_ids > 0
        click_mask = np.array(click_mask, dtype='float32')

        click_num = click_mask.sum(axis=-1)
        click_num = np.array(click_num, dtype='int32')
        click_num = click_num.reshape((len(click_num), 1))

        click_num = torch.LongTensor(click_num).cuda()
        click_mask = torch.FloatTensor(click_mask).cuda()
        user_title = torch.FloatTensor(user_title).cuda()

        return [user_title, click_mask, click_num]
