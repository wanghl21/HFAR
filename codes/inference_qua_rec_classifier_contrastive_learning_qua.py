# %%
from keras_version.metrics_utils import evaluate_rank
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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MAX_SENTENCE = 30
MAX_ALL = 50

# %%

if len(sys.argv) < 3:
    print("parameters！")
    sys.exit(1)
beta = float(sys.argv[2])
print(f"beta：{beta}")
method = sys.argv[1]
print(f"method：{method}")


# %%
# Path
data_root_path = '/data/whl/MIND_Small-Release'
embedding_path = '/data/whl/MIND_Small-Release/embedding_path/'

# %%
# read news content
news, news_index, category_dict, subcategory_dict, word_dict = read_news(
    data_root_path, 'docs.tsv')
news_title, news_vert, news_subvert = get_doc_input(
    news, news_index, category_dict, subcategory_dict, word_dict)


# %%
# read news quality
news_qua_dict = read_news_quality(
    data_root_path, 'docs_quality_fators.tsv', news_index)
news_qua = np.asarray(news_qua_dict)
news_qua.shape

news_qua = news_qua[:, [1, 3, 4]]
news_title.shape

# %%
# read session and parse

test_session = read_clickhistory(data_root_path, 'val_sam2.tsv', news_index)
test_user = parse_user(test_session, news_index)

# %%
npratio = 1
test_impressions, test_userids = get_test_input(test_session, news_index)


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


# %%
# model evaluate
# for recommand
path = "/data/whl/Quality_rec/"
num_epochs = 7


def load_evaluate_all_metrics_ours():

    news_scorings = np.load(path+"scoring/"+method +
                            "/"+str(num_epochs)+"0.13our_news_scoring.npy")
    user_scorings = np.load(path+"scoring/"+method +
                            "/"+str(num_epochs)+"0.13our_user_scoring.npy")
    news_qua_scorings = np.load(
        path+"scoring/"+method+"/"+str(num_epochs)+"0.13our_news_qua_scorings.npy")

    eval_rank = evaluate_model_qua_ours(
        test_impressions[:20000], user_scorings, news_scorings, news_qua_scorings, news_qua, beta)
    print(eval_rank)
    recall_rank = evaluate_recall_model_qua_ours(
        test_impressions[:20000], user_scorings, news_scorings, news_qua_scorings, news_qua, beta)
    print(recall_rank)

    with open('Validation/' + method+'/0.13alpha' + str(beta) + method + '_ours.json', 'a+') as f:
        # f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())+'\n')
        f.write("eval_rank"+'\n')
        s = json.dumps(eval_rank)
        f.write(s+'\n')
        f.write("recall_rank"+'\n')
        s = json.dumps(recall_rank)
        f.write(s+'\n')


load_evaluate_all_metrics_ours()
