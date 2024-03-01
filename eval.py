from scipy.linalg import norm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


from networks.model import primary_encoder_v2_no_pooler_for_con
import gc

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
import warnings

from util_visualization import get_dataset_labels, clean_tweet, create_labels, plot_tsne_pca

def predictions(df, emb_dataframe, df_lab, hard, nb):
    print("calcolo True false")
    labels_pred = []
    labs_0 = []
    labs_grou = []
    for i in range(len(df)):
        vote_uno = 0
        vote_zero = 0
        # prendo riga del dataframe riferita al tweet di test di cui voglio fare la predizione
        row = emb_dataframe.iloc[[i]]
        copy_lab = df_lab.copy()
        # Ottengo gli indici dei 3 tweet più simili e l'indice del tweet meno simile
        ind_max = np.argmax(row)
        ind_min = np.argmin(row)
        if copy_lab[ind_min + 1] == 0:
            label_reverse = 1
        else:
            label_reverse = 0
        y = np.argsort(row)
        ind_max2 = y[0][-2]
        ind_max3 = y[0][-3]
        # faccio votazione tra i 35 valori più simili
        for k in range(1, nb):  # 36
            if copy_lab[y[0][-k] + 1] == 1:
                vote_uno = vote_uno + 1
            else:
                vote_zero = vote_zero + 1
        # Se la differenza di voti è minore di hard, sono in uno stato di indecisione e predico disagreement
        # Se no faccio append normalmente
        if abs(vote_uno - vote_zero) < hard:
            labels_pred.append(0)
        elif vote_uno > vote_zero:
            labels_pred.append(1)
        else:
            labels_pred.append(0)
    return labels_pred

#Funzione che calcola la matrice di similarità tra ogni istanza di test e tutte le istanze di train restituisce in output poi un datatframe
def cal_distances(df,word_embeddings_train,word_embeddings_test):
    print("Calcolo matrice distanze embeddings")
    warnings.filterwarnings("ignore")
    matrix = []
    for i in range(len(df)):
        col = []
        for j in range(len(word_embeddings_train)):
            text1 = word_embeddings_train[j]
            text2 = word_embeddings_test[i]
            # cosine similarity
            distance = np.dot(text1, text2) / (norm(text1) * norm(text2))
            #distance = np.linalg.norm(text1 - text2) # euclidean distance
            col.append(distance)
        matrix.append(col)

    emb_dataframe = pd.DataFrame(matrix)
    return emb_dataframe

# Funzione che estrae gli embeddings dal modello
def create_embeddings(df,layers):
    word_embeddings = []
    base_dir = './Disagreement/Dataset'
    tokenizer = BertTokenizer.from_pretrained(base_dir + '/TokenzierBert')
    for idx in range(1, len(df) + 1):
        with (torch.no_grad()):
            input_dict = tokenizer(df["text"][idx], return_tensors="pt", padding='max_length', truncation=True,
                                   max_length=128)
            input_dict.to(device)
            df = df.tail(-1)
            """hidden_states, _ = model.get_cls_features_ptrnsp(input_dict["input_ids"], input_dict["attention_mask"])
            del input_dict
            word_embeddings.append(hidden_states.cpu().numpy()[0])"""
            _,_,hidden_states = model.get_cls_features_ptrnsp(input_dict["input_ids"], input_dict["attention_mask"])
            del input_dict
            if layers != 1:
                hidden_states = hidden_states[-layers:]  # Seleziona gli ultimi n layer
                layer_average = torch.mean(torch.stack(hidden_states), dim=0)  # Media dei layer
                sentence_embeddings = torch.mean(layer_average, dim=1).squeeze()  # Media dei token
            else:
                sentence_embeddings = torch.mean(hidden_states[-1], dim=1).squeeze()

            sentence_embeddings = sentence_embeddings.cpu().numpy()
            word_embeddings.append(sentence_embeddings)

    return np.array(word_embeddings)

#funzione che calcola le performance
def prediction_metrics(lab_test, labels_pred):
    matrix = confusion_matrix(lab_test, labels_pred)
    print("Confusion Matrix:\n", matrix)
    print("Classification Report:\n", classification_report(lab_test, labels_pred))
    print("Total Accuracy:\n", accuracy_score(lab_test, labels_pred))
    print("Equals Only\n")
    return accuracy_score(lab_test, labels_pred)

#funzione che crea dataframe delle similarità tra gli embeddings
def create_dataframe_distances(df_tot_train,df_test,file_name,plot_graph,save,layers):
    # Creo embeddings
    print("Inizio a creare le feature del Test")
    word_embeddings_train = create_embeddings(df_tot_train,layers)
    word_embeddings_test = create_embeddings(df_test,layers)
    col_train = create_labels(df_tot_train["hard_label"])
    col_test = create_labels(df_test["hard_label"])
    # Faccio plot embeddings sia con label di Odio che di Disagreement
    gc.collect()
    print("Plotting..............")
    if plot_graph:
        hid, hid_test = plot_tsne_pca("Total", word_embeddings_train, word_embeddings_test, col_train, col_test)
    col_train = create_labels(df_tot_train["disagreement"])
    col_test = create_labels(df_test["disagreement"])
    if plot_graph:
        _, _ = plot_tsne_pca("Total", word_embeddings_train, word_embeddings_test, col_train, col_test)

    emb_dataframe = cal_distances(df_test, word_embeddings_train, word_embeddings_test)
    if save:
        emb_dataframe.to_csv(file_name, sep=',', index=False, encoding='utf-8')
    return emb_dataframe

# Carico Modello
model = primary_encoder_v2_no_pooler_for_con(768, 2)
model.load_state_dict(
    torch.load('./models/infoNCEBest.pth'), strict=False)
model.eval()
gc.collect()

# Carico e pulisco il dataset
base_dir = './Disagreement/Dataset'

print("Load Dati")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

base_dir = './Disagreement/Dataset'
tokenizer = BertTokenizer.from_pretrained(base_dir + '/TokenzierBert')

# Read csv test
warnings.filterwarnings("ignore")
df_md_test = pd.read_json(base_dir + '/MD-Agreement_test.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_md_test = get_dataset_labels(df_md_test)

df_brexit_test = pd.read_json(base_dir + '/HS-Brexit_test.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_brexit_test = get_dataset_labels(df_brexit_test)

df_armis_test = pd.read_json(base_dir + '/ArMIS_test.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_armis_test = get_dataset_labels(df_armis_test)

df_conv_test = pd.read_json(base_dir + '/ConvAbuse_test.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_conv_test = get_dataset_labels(df_conv_test)

# Unisco in un unico dataset
frames = [df_md_test, df_armis_test, df_conv_test, df_brexit_test]
df_test = pd.concat(frames)
df_test = df_test.reset_index()
df_test = df_test.drop(['soft_label_0', 'soft_label_1'], axis=1)
original_text_test = df_test["text"]

df_brexit_test = clean_tweet(df_brexit_test)

df_md_test = clean_tweet(df_md_test)

df_conv_test = clean_tweet(df_conv_test)

frames = [df_md_test, df_armis_test, df_conv_test, df_brexit_test]
df_test = pd.concat(frames)
df_test = df_test.reset_index()
df_test = df_test.drop(['soft_label_0', 'soft_label_1'], axis=1)

# read csv train
df_md_train = pd.read_json(base_dir + '/MD-Agreement_train.json', orient='index')[['text', 'hard_label', 'soft_label']]
df_md_val = pd.read_json(base_dir + '/MD-Agreement_dev.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_md_train = get_dataset_labels(df_md_train)
df_md_val = get_dataset_labels(df_md_val)

df_brexit_train = pd.read_json(base_dir + '/HS-Brexit_train.json', orient='index')[['text', 'hard_label', 'soft_label']]
df_brexit_val = pd.read_json(base_dir + '/HS-Brexit_dev.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_brexit_train = get_dataset_labels(df_brexit_train)
df_brexit_val = get_dataset_labels(df_brexit_val)

df_armis_train = pd.read_json(base_dir + '/ArMIS_train.json', orient='index')[['text', 'hard_label', 'soft_label']]
df_armis_val = pd.read_json(base_dir + '/ArMIS_dev.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_armis_train = get_dataset_labels(df_armis_train)
df_armis_val = get_dataset_labels(df_armis_val)

df_conv_train = pd.read_json(base_dir + '/ConvAbuse_train.json', orient='index')[['text', 'hard_label', 'soft_label']]
df_conv_val = pd.read_json(base_dir + '/ConvAbuse_dev.json', orient='index')[['text', 'hard_label', 'soft_label']]

df_conv_train = get_dataset_labels(df_conv_train)
df_conv_val = get_dataset_labels(df_conv_val)

df_brexit_train = clean_tweet(df_brexit_train)
df_brexit_val = clean_tweet(df_brexit_val)

df_md_train = clean_tweet(df_md_train)
df_md_val = clean_tweet(df_md_val)

df_conv_train = clean_tweet(df_conv_train)
df_conv_val = clean_tweet(df_conv_val)

frames = [df_md_train, df_md_val, df_armis_train, df_armis_val, df_conv_train, df_conv_val, df_brexit_train,
          df_brexit_val]
""""frames = [df_md_train, df_armis_train, df_conv_train, df_brexit_train]"""
df_tot_train = pd.concat(frames)
df_tot_train = df_tot_train.reset_index()
df_tot_train = df_tot_train.drop(['soft_label_0', 'index', 'soft_label_1'], axis=1)

labels_train = df_brexit_train["disagreement"]
labels_dev = df_brexit_val["disagreement"]

frames = [labels_train, labels_dev]
df_lab = pd.concat(frames)
df_lab = df_lab.reset_index()
df_lab = df_lab.drop(['index'], axis=1)

labels_test = df_brexit_test["disagreement"]

gc.collect()

# Aggiusto indexes
df_tot_train.index = np.arange(1, len(df_tot_train) + 1)
df_test.index = np.arange(1, len(df_test) + 1)

lab_train = df_tot_train["disagreement"]

lab_test = df_test["disagreement"]

# Fase di predizione
# se ho già creato il dataframe con le distanze lo carico e basta altrimenti lo calcolo
print("INFONCE_Total Predictions.........................")
print()
print()
file_name = "contr_final.csv"
#emb_dataframe = pd.read_csv(file_name,sep=',')
emb_dataframe = create_dataframe_distances(df_tot_train,df_test,file_name,False,True,7)

# Predictions
labels_pred = predictions(df_test, emb_dataframe, lab_train, 7, 59)

prediction_metrics(lab_test,labels_pred)

# Mostro prediction per dataset
lunghezza1 = 3057
lunghezza2 = 145
lunghezza3 = 840
lunghezza4 = 168

pred_md = labels_pred[:lunghezza1]
pred_arm = labels_pred[lunghezza1:lunghezza1 + lunghezza2]
pred_conv = labels_pred[lunghezza1 + lunghezza2:lunghezza1 + lunghezza2 + lunghezza3]
pred_brexit = labels_pred[lunghezza1 + lunghezza2 + lunghezza3:]

prediction_metrics(df_brexit_test["disagreement"],pred_brexit)
prediction_metrics(df_armis_test["disagreement"],pred_arm)
prediction_metrics(df_conv_test["disagreement"],pred_conv)
prediction_metrics(df_md_test["disagreement"],pred_md)


print("INFONCE_Split Predictions.........................")
print()
print()
neibArmis = 22
neibMD = 105
neibConv = 19
neibBrexit = 50

preds = []
# se ho già creato il dataframe con le distanze lo carico e basta alttrimenti lo calcolo per ogni dataset
emb_dataframe = create_dataframe_distances(df_md_train,df_md_test,"contr_md.csv",False,True,7)
#emb_dataframe = pd.read_csv("contr_md.csv", sep=',')
# Predictions
labels_pred = predictions(df_md_test, emb_dataframe, df_md_train["disagreement"], 2, neibMD)
preds = preds + labels_pred
prediction_metrics(df_md_test["disagreement"],labels_pred)
emb_dataframe = create_dataframe_distances(df_armis_train,df_armis_test,"contr_armis.csv",False,True,7)
#emb_dataframe = pd.read_csv("contr_armis.csv", sep=',')
# Predictions
labels_pred = predictions(df_armis_test, emb_dataframe, df_armis_train["disagreement"], 2, neibArmis)
preds = preds + labels_pred
prediction_metrics(df_armis_test["disagreement"],labels_pred)
emb_dataframe = create_dataframe_distances(df_conv_train,df_conv_test,"contr_conv.csv",False,True,7)
#emb_dataframe = pd.read_csv("contr_conv.csv", sep=',')
# Predictions
labels_pred = predictions(df_conv_test, emb_dataframe, df_conv_train["disagreement"], 2, neibConv)
preds = preds + labels_pred
prediction_metrics(df_conv_test["disagreement"],labels_pred)
emb_dataframe = create_dataframe_distances(df_brexit_train,df_brexit_test,"contr_brexit.csv",False,True,7)
#emb_dataframe = pd.read_csv("contr_brexit.csv", sep=',')
# Predictions
labels_pred = predictions(df_brexit_test, emb_dataframe, df_brexit_train["disagreement"], 2, neibBrexit)
preds = preds + labels_pred
prediction_metrics(df_brexit_test["disagreement"],labels_pred)
print("prediction totale ")
acc_new = prediction_metrics(lab_test, preds)



