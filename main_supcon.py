from __future__ import print_function

import gc
import os
import re
import sys
import argparse
import time
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm


from datasets import Dataset
from torch import nn
from transformers import BertTokenizer
import torch.nn.functional as Fun

from util import  get_linear_schedule_with_warmup
from util import warmup_learning_rate
from util import set_optimizer, save_model
from networks.model import primary_encoder_v2_no_pooler_for_con
from losses import SupConLossText, InfoNCE

import nlpaug.augmenter.word as naw

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

# Clean all tweets
def clean_tweet(df):
    for i in range(1, len(df["text"]) + 1):
        clean_tweet = re.sub("<user>", "", df["text"][i])
        clean_tweet = re.sub('prev_user', '.', clean_tweet)
        clean_tweet = re.sub('"prev_agent":', '.', clean_tweet)
        clean_tweet = re.sub('"agent":', '.', clean_tweet)
        clean_tweet = re.sub('"user":', '.', clean_tweet)
        clean_tweet = re.sub('"', '', clean_tweet)
        clean_tweet = re.sub('=', '', clean_tweet)
        clean_tweet = re.sub('{', '', clean_tweet)
        clean_tweet = re.sub('}', '', clean_tweet)
        clean_tweet = re.sub(':', '', clean_tweet)
        clean_tweet = re.sub(',', '', clean_tweet)
        clean_tweet = re.sub("]", '', clean_tweet)
        clean_tweet = re.sub("r'\([^)]*\)", '', clean_tweet)
        clean_tweet = re.sub("-", '', clean_tweet)
        clean_tweet = re.sub("_", "", clean_tweet)
        clean_tweet = re.sub("RT", "", clean_tweet)
        clean_tweet = re.sub("<url>", "", clean_tweet)
        clean_tweet = clean_tweet.replace('...', " ")
        clean_tweet = clean_tweet.replace('\n', " ")
        clean_tweet = clean_tweet.replace('&amp', " ")
        clean_tweet = clean_tweet.strip()
        df["text"][i] = clean_tweet.lower()
    return df

# Load Dataset
def load_hate():
    base_dir = 'C:/Users/micof/OneDrive/Desktop/Lavoro Tesi/Disagreement/Dataset'
    df_md_test = pd.read_json(base_dir + '/MD-Agreement_test.json', orient='index')[
        ['text', 'hard_label']]

    df_brexit_test = pd.read_json(base_dir + '/HS-Brexit_test.json', orient='index')[
        ['text', 'hard_label']]

    df_armis_test = pd.read_json(base_dir + '/ArMIS_test.json', orient='index')[['text', 'hard_label']]

    df_conv_test = pd.read_json(base_dir + '/ConvAbuse_test.json', orient='index')[['text', 'hard_label']]

    df_brexit_test = clean_tweet(df_brexit_test)


    df_md_test = clean_tweet(df_md_test)

    df_conv_test = clean_tweet(df_conv_test)

    frames = [df_md_test, df_armis_test, df_conv_test, df_brexit_test]
    df_test = pd.concat(frames)
    df_test = df_test.reset_index()
    df_test = df_test.drop(['index'], axis=1)

    df_md_train = pd.read_json(base_dir + '/MD-Agreement_train.json', orient='index')[
        ['text', 'hard_label']]
    df_md_val = pd.read_json(base_dir + '/MD-Agreement_dev.json', orient='index')[['text', 'hard_label']]

    df_brexit_train = pd.read_json(base_dir + '/HS-Brexit_train.json', orient='index')[
        ['text', 'hard_label']]
    df_brexit_val = pd.read_json(base_dir + '/HS-Brexit_dev.json', orient='index')[['text', 'hard_label']]

    df_armis_train = pd.read_json(base_dir + '/ArMIS_train.json', orient='index')[['text', 'hard_label']]
    df_armis_val = pd.read_json(base_dir + '/ArMIS_dev.json', orient='index')[['text', 'hard_label']]
    #df_armis_train_aug = augment(df_armis_train)
    #df_armis_val_aug = augment(df_armis_val)

    df_conv_train = pd.read_json(base_dir + '/ConvAbuse_train.json', orient='index')[
        ['text', 'hard_label']]
    df_conv_val = pd.read_json(base_dir + '/ConvAbuse_dev.json', orient='index')[['text', 'hard_label']]

    df_brexit_train = clean_tweet(df_brexit_train)
    #df_brexit_train_aug = augment(df_brexit_train)
    df_brexit_val = clean_tweet(df_brexit_val)
    #df_brexit_val_aug = augment(df_brexit_val)

    df_md_train = clean_tweet(df_md_train)
    df_md_val = clean_tweet(df_md_val)

    df_conv_train = clean_tweet(df_conv_train)
    df_conv_val = clean_tweet(df_conv_val)

    frames = [df_md_train, df_md_val, df_armis_train, df_armis_val, df_conv_train, df_conv_val, df_brexit_train,
              df_brexit_val]
    df_tot_train = pd.concat(frames)
    df_tot_train = df_tot_train.sample(frac=1, random_state=42)
    df_tot_train = df_tot_train.reset_index()
    df_tot_train = df_tot_train.drop(['index'], axis=1)
    df_train, df_val = train_test_split(df_tot_train, test_size=0.2, shuffle=True, random_state=123)

    return df_train, df_val, df_test

# Funzione per fara data augmentation (opzionale)
def augment(df):
    aug_sub = naw.ContextualWordEmbsAug(
        model_path='bert-base-multilingual-cased', action="substitute", device='cuda')
    texts = []
    labels = []
    for i in range(len(df)):
        print(round((i / len(df)) * 100), "%", end='\r')
        t = df.iloc[i]
        trad = translate(df.iloc[i], aug_sub)
        texts.append(trad[1])
        labels.append(trad[2])
    return pd.DataFrame({"text": texts, "hard_label": labels})


def translate(x, aug_ins):
    trad = aug_ins.augment(x["text"])
    return [x["text"], trad[0], x["hard_label"]]


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--dataset', type=str, default='disagreement', help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='InfoNCE', help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


# Create data loader
def set_loader(opt, aug):
    # construct data loader
    df_train, val, test = load_hate()
    t = df_train
    if aug:
        df_aug_val = augment(val)
        df_aug_val.to_csv("C:/Users/micof/OneDrive/Desktop/Google contr/aug_val.csv", index=False)
        df_aug = augment(df_train)
        df_aug.to_csv("C:/Users/micof/OneDrive/Desktop/Google contr/aug.csv", index=False)
        df_aug = Dataset.from_pandas(df_aug)
        df_aug_val = Dataset.from_pandas(df_aug_val)

    df_train = df_train.reset_index()
    df_train = df_train.drop(['index'], axis=1)

    df_train = Dataset.from_pandas(df_train)
    df_train = df_train.map(preprocess_function, batched=True)
    if aug:
        df_aug = df_aug.map(preprocess_function, batched=True)
        df_train = df_train.add_column("input_ids_aug", df_aug["input_ids"])
        df_train = df_train.add_column("attention_mask_aug", df_aug["attention_mask"])
        # df_train = df_train.add_column("text_aug", df_aug["text"])
    labels = df_train["hard_label"]
    class_sample_count = np.array(
        [len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(labels))
    train_loader = torch.utils.data.DataLoader(
        df_train, batch_size=opt.batch_size, shuffle=True)


    val = Dataset.from_pandas(val)
    val = val.map(preprocess_function, batched=True)
    if aug:
        df_aug_val = df_aug_val.map(preprocess_function, batched=True)
        val = val.add_column("input_ids_aug", df_aug_val["input_ids"])
        val = val.add_column("attention_mask_aug", df_aug_val["attention_mask"])
    # Calcola i pesi solo per le label presenti nel dataset
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=opt.batch_size, shuffle=True)
    return train_loader, len(df_train), val_loader, test, t


def preprocess_function(examples):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=128)


def set_model(opt, training_data):
    # creazione modello lavorare con bert
    # model = BertClassifier()
    model = primary_encoder_v2_no_pooler_for_con(768, 2)
    criterion = SupConLossText(temperature=opt.temp)
    class_weights = []
    for sample in training_data["hard_label"].value_counts().values:
        class_weights.append(1 - (sample / training_data.shape[0]))
    weight_for_class_0 = 1
    weight_for_class_1 = 0.3
    weight = torch.tensor(class_weights)  # higher weight for class 1
    device = torch.device("cuda")
    weight = weight.to(device)
    # nn.BCEWithLogitsLoss(weight  = weight)
    losses = {"contrastive": SupConLossText(temperature=opt.temp,weights=weight),
              "infoNCE": InfoNCE(temperature=opt.temp, negative_mode="unpaired"), "ce_loss": nn.BCEWithLogitsLoss(weight=weight)}

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, losses


def train(train_loader, model, criterion, optimizer, epoch, opt, running_loss, loss_values, lr_scheduler, loss_function,
          val_dataloader, running_val_loss, val_loss_values, lambda_value, total_acc_train=0, total_acc_val=0):
    """one epoch training"""
    model.train()
    # con encoding dei testi
    ce = 0
    cl = 0
    va_ce = 0
    va_cl = 0
    aug = False
    total = 0
    total_val = 0
    for idx, values in enumerate(tqdm(train_loader)):
    #for values in train_loader:
        mask = torch.stack(values['attention_mask'], dim=1)
        input_id = torch.stack(values["input_ids"], dim=1).squeeze(0)
        if aug:
            aug_mask = torch.stack(values['attention_mask_aug'], dim=1)
            aug_input_id = torch.stack(values["input_ids_aug"], dim=1).squeeze(0)
        labels = values["hard_label"]
        lab = Fun.one_hot(labels, num_classes=2)
        if torch.cuda.is_available():
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            mask = mask.to(device)
            labels = labels.to(device)
            input_id = input_id.to(device)
            if aug:
                aug_mask = aug_mask.to(device)
                aug_input_id = aug_input_id.to(device)
            lab = lab.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        model = model.cuda()
        hidden, features,_ = model.get_cls_features_ptrnsp(input_id, mask)
        contr = 0
        if aug:
            _, pos = model.get_cls_features_ptrnsp(aug_input_id, aug_mask)
            pred_1 = model(hidden)
            ce_loss = (lambda_value * loss_function["ce_loss"](pred_1, lab.float()))
            nce_loss = loss_function["infoNCE"](features, pos)
            features = torch.cat([features, pos], dim=0)
            labels_aug = torch.cat([labels, labels], dim=0)
            contr_loss = loss_function["contrastive"](features, labels_aug)
            ce = ce + ce_loss
            cl = cl + ((1 - lambda_value) * (contr_loss - nce_loss))
            loss = ce_loss + ((1 - lambda_value) * (contr_loss - nce_loss))
        else:
            pred_1 = model(hidden)
            ce_loss = (lambda_value * loss_function["ce_loss"](pred_1, lab.float()))
            positives = []
            negatives = []
            for p in range(len(labels)):
                if labels[p] == 0:
                    positives.append(features[p])
                else:
                    negatives.append(features[p])
            if len(positives) < len(negatives):
                tmp = negatives
                negatives = positives
                positives = tmp
            pos_confr = []
            feat_confr = []
            # Divido gli esempi positivi tra Features ed esempi Positivi
            if len(positives) % 2 == 0:
                pos_confr = [positives[i] for i in range(len(positives)) if i % 2 == 0]
                feat_confr = [positives[i] for i in range(len(positives)) if i % 2 != 0]
            else:
                pos_confr = [positives[i] for i in range(len(positives)) if i % 2 == 0]
                feat_confr = [positives[i] for i in range(len(positives)) if i % 2 != 0]
                if len(pos_confr) > len(feat_confr):
                    pos_confr.pop(0)
                else:
                    feat_confr.pop(0)
            #Calcolo InfoNCE Caso 1 ho esempi negativi nella batch caso 2 non ho esempi negativi
            if (len(negatives) > 0):
                pos = torch.stack(pos_confr, dim=0).squeeze(0)
                neg = torch.stack(negatives, dim=0).squeeze(0)
                fea = torch.stack(feat_confr, dim=0).squeeze(0)
                if len(pos) > bsz:
                    pos = pos.unsqueeze(0)
                if len(neg) > bsz:
                    neg = neg.unsqueeze(0)
                if len(fea) > bsz:
                    fea = fea.unsqueeze(0)
                nce_loss = loss_function["infoNCE"](fea, pos, neg)
                # contr_loss = loss_function["contrastive"](features, labels)
                ce = ce + ce_loss
                cl = cl + ((1 - lambda_value) * (nce_loss+contr))
                loss = ce_loss + ((1 - lambda_value) * (nce_loss+contr))
            else:
                pos = torch.stack(pos_confr, dim=0).squeeze(0)
                fea = torch.stack(feat_confr, dim=0).squeeze(0)
                if len(pos) > bsz:
                    pos = pos.unsqueeze(0)
                if len(fea) > bsz:
                    fea = fea.unsqueeze(0)
                nce_loss = loss_function["infoNCE"](fea, pos)
                # contr_loss = loss_function["contrastive"](features, labels)
                ce = ce + ce_loss
                cl = cl + ((1 - lambda_value) * (nce_loss+contr))
                loss = ce_loss + ((1 - lambda_value) * (nce_loss+contr))

        running_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # evito gradient clipping
        optimizer.step()
        total += labels.size(0)
        acc = (pred_1.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        model.zero_grad()
        lr_scheduler.step()
        optimizer.zero_grad()
    with torch.no_grad():
        #Validation
        for val_idx, val_values in enumerate(tqdm(val_dataloader)):
            mask = torch.stack(val_values['attention_mask'], dim=1)
            input_id = torch.stack(val_values["input_ids"], dim=1).squeeze(0)
            if aug:
                aug_mask = torch.stack(val_values['attention_mask_aug'], dim=1)
                aug_input_id = torch.stack(val_values["input_ids_aug"], dim=1).squeeze(0)
            labels = val_values["hard_label"]
            lab = Fun.one_hot(labels, num_classes=2)
            if torch.cuda.is_available():
                use_cuda = torch.cuda.is_available()
                device = torch.device("cuda" if use_cuda else "cpu")
                mask = mask.to(device)
                labels = labels.to(device)
                input_id = input_id.to(device)
                if aug:
                    aug_mask = aug_mask.to(device)
                    aug_input_id = aug_input_id.to(device)
                lab = lab.to(device)
            bsz = labels.shape[0]
            hidden, features,_ = model.get_cls_features_ptrnsp(input_id, mask)
            pred_1 = model(hidden)
            ce_val_loss = (lambda_value * loss_function["ce_loss"](pred_1, lab.float()))
            #contr_val = loss_function["contrastive"](features, labels)
            contr_val = 0
            positives = []
            negatives = []
            for p in range(len(labels)):
                if labels[p] == 0:
                    positives.append(features[p])
                else:
                    negatives.append(features[p])
            if len(positives) < len(negatives):
                tmp = negatives
                negatives = positives
                positives = tmp
            if len(positives) % 2 == 0:
                pos_confr_val = [positives[i] for i in range(len(positives)) if i % 2 == 0]
                feat_confr_val = [positives[i] for i in range(len(positives)) if i % 2 != 0]
            else:
                pos_confr_val = [positives[i] for i in range(len(positives)) if i % 2 == 0]
                feat_confr_val = [positives[i] for i in range(len(positives)) if i % 2 != 0]
                if len(pos_confr_val) > len(feat_confr_val):
                    pos_confr_val.pop(0)
                else:
                    feat_confr_val.pop(0)
            if len(negatives) > 0:
                if len(pos_confr_val) > 0:
                    pos_val = torch.stack(pos_confr_val, dim=0).squeeze(0)
                    neg_val = torch.stack(negatives, dim=0).squeeze(0)
                    fea_val = torch.stack(feat_confr_val, dim=0).squeeze(0)
                    if len(pos_val) > bsz:
                        pos_val = pos_val.unsqueeze(0)
                    if len(neg_val) > bsz:
                        neg_val = neg_val.unsqueeze(0)
                    if len(fea_val) > bsz:
                        fea_val = fea_val.unsqueeze(0)
                    nce_val_loss = loss_function["infoNCE"](fea_val, pos_val, neg_val)
                    va_ce = va_ce + ce_val_loss
                    va_cl = va_cl + ((1 - lambda_value) * (nce_val_loss+contr_val))
                    val_loss = ce_val_loss + ((1 - lambda_value) * (nce_val_loss+contr_val))
                    last_batch = False
                else:
                    last_batch = True
            else:
                if len(pos_confr_val) > 0:
                    pos_val = torch.stack(pos_confr_val, dim=0).squeeze(0)
                    fea_val = torch.stack(feat_confr_val, dim=0).squeeze(0)
                    if len(pos_val) > bsz:
                        pos_val = pos_val.unsqueeze(0)
                    if len(fea_val) > bsz:
                        fea_val = fea_val.unsqueeze(0)
                    nce_val_loss = loss_function["infoNCE"](fea_val, pos_val)
                    va_ce = va_ce + ce_val_loss
                    va_cl = va_cl + ((1 - lambda_value) * (nce_val_loss+contr_val))
                    val_loss = ce_val_loss + ((1 - lambda_value) * (nce_val_loss+contr_val))
                    last_batch = False
                else:
                    last_batch = True
            total_val += labels.size(0)
            acc = (pred_1.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            running_val_loss += val_loss.item()
            if last_batch:
                div = len(val_dataloader)-1
            else:
                div = len(val_dataloader)

    # print info
    print("val")
    print(va_cl / len(val_dataloader))
    print(va_ce / len(val_dataloader))
    print("train")
    print(cl / len(train_loader))
    print(ce / len(train_loader))
    return running_loss / len(train_loader), running_val_loss / div, model, total_acc_train / total, total_acc_val / total_val

def create_embeddings(df, model):
    word_embeddings = []
    # Caricare Tokenizer
    base_dir = 'C:/Users/micof/OneDrive/Desktop/Lavoro Tesi/Disagreement/Dataset'
    tokenizer = BertTokenizer.from_pretrained(base_dir + '/TokenzierBert')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    for idx in df["text"].items():
        with torch.no_grad():
            input_dict = tokenizer(idx[1], return_tensors="pt", padding='max_length', truncation=True,
                                   max_length=128)
            input_dict.to(device)
            df = df.tail(-1)
            #hidden_states, _,_ = model.get_cls_features_ptrnsp(input_dict["input_ids"], input_dict["attention_mask"])
            _, _, hidden_states = model.get_cls_features_ptrnsp(input_dict["input_ids"], input_dict["attention_mask"])
            del input_dict
            #word_embeddings.append(hidden_states.cpu().numpy()[0]) prima

            sentence_embeddings = torch.mean(hidden_states[-1], dim=1).squeeze()
            sentence_embeddings = sentence_embeddings.cpu().numpy()
            word_embeddings.append(sentence_embeddings)
            # gc.collect()

    return np.array(word_embeddings)


from util_visualization import clean_tweet, create_labels, plot_tsne_pca


def main():
    opt = parse_option()
    warnings.filterwarnings("ignore")
    # build data loader
    train_loader, length, val, test, train_df = set_loader(opt, False)

    # build model and criterion
    model, criterion, losses = set_model(opt, train_df)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    num_training_steps = int(length * opt.epochs)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    loss_values = []
    val_loss_values = []
    acc_values_train = []
    acc_values_val = []
    #Recupero labels per fare plot PCA TSNE
    lab_test = test["hard_label"]
    col_test = create_labels(lab_test)
    lab_train = train_df["hard_label"]
    col_train = create_labels(lab_train)
    lambda_value = 0.25
    # training routine
    for epoch in range(1, opt.epochs + 1):

        running_loss = 0.0
        running_val_loss = 0.0

        # train for one epoch
        time1 = time.time()
        running_loss, running_val_loss, m, total_acc_train, total_acc_val = train(train_loader, model, criterion,
                                                                                  optimizer, epoch, opt,
                                                                                  running_loss, loss_values,
                                                                                  lr_scheduler, losses, val,
                                                                                  running_val_loss,
                                                                                  val_loss_values,
                                                                                  lambda_value=lambda_value)
        print('epoch number ', epoch, 'Loss ', round(running_loss, 3), 'Accuracy', total_acc_train, 'val loss ',
              round(running_val_loss, 3), 'Val Accuracy ', total_acc_val)
        sys.stdout.flush()
        loss_values.append(running_loss)
        val_loss_values.append(running_val_loss)
        acc_values_train.append(total_acc_train)
        acc_values_val.append(total_acc_val)

        # quando salvo faccio anche plot embeddings
        if epoch % opt.save_freq == 0:
            word_embeddings_train = create_embeddings(train_df, m)
            word_embeddings_test = create_embeddings(test, m)
            gc.collect()
            print("Plotting..............")
            title = "Epoch n^" + str(epoch)
            plot_tsne_pca(title, word_embeddings_train, word_embeddings_test, col_train, col_test)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)


    # plot graphs loss and accuracy
    plt.plot(np.array(loss_values), 'r', label='train_loss')
    plt.legend()
    plt.show()
    plt.plot(np.array(val_loss_values), 'b', label='val_loss')
    plt.legend()
    plt.show()
    plt.plot(np.array(loss_values), 'r', label='train_loss')
    plt.plot(np.array(val_loss_values), 'b', label='val_loss')
    plt.legend()
    plt.show()
    plt.plot(np.array(acc_values_train), 'r', label='train_acc')
    plt.plot(np.array(acc_values_val), 'b', label='val_acc')
    plt.legend()
    plt.show()

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
