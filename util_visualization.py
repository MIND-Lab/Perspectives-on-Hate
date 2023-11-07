import gc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt

def get_dataset_labels(df, columns=['text', 'hard_label', 'soft_label_0', 'soft_label_1', 'disagreement']):
    df['soft_label_1'] = df['soft_label'].apply(lambda x: x['1'])
    df['soft_label_0'] = df['soft_label'].apply(lambda x: x['0'])
    df['disagreement'] = df['soft_label_0'].apply(lambda x: int(x == 0 or x == 1))
    return df[columns]


def clean_tweet(df):
    for i in range(1, len(df["text"])+1):
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


def create_labels(labels):
    color = []
    for indx in labels.items():
        if indx[1] == 0:
            color.append('blue')
        else:
            color.append('red')
    return color

def plot_tsne_pca(title, word_embedding_train, word_embeddings_test, color_train, color_test):
    # Applica t-SNE ai pesi dell'ultimo layer
    perpTrain = round((len(word_embedding_train)/100)*5)
    perpTest = round((len(word_embeddings_test)/100)*5)
    pca = PCA(n_components= 10,random_state= 42)
    reduced_embeddings = pca.fit_transform(word_embedding_train)

    pca_test = PCA(n_components= 10,random_state= 42)
    reduced_embeddings_test = pca_test.fit_transform(word_embeddings_test)

    #umap_test = UMAP(n_components=2, init='random', random_state=123)
    #umap_train = UMAP(n_components=2, init='random', random_state=123)

    #hidden_states = umap_train.fit_transform(reduced_embeddings)
    #hidden_states_test = umap_test.fit_transform(reduced_embeddings_test)

    tsne = TSNE(n_components=2,perplexity=perpTrain,
                random_state=42)
    hidden_states = tsne.fit_transform(reduced_embeddings)

    tsne_test = TSNE(n_components=2, perplexity=perpTest,
                random_state=42,n_iter = 3000)
    hidden_states_test = tsne_test.fit_transform(reduced_embeddings_test)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 30))
    fig.suptitle(title)
    ax1.scatter(hidden_states[:, 0], hidden_states[:, 1], c=color_train)
    ax1.title.set_text('Train ' + title)
    ax2.scatter(hidden_states_test[:, 0], hidden_states_test[:, 1], c=color_test)
    ax2.title.set_text('Test ' + title)
    plt.show()
    gc.collect()
    return hidden_states,hidden_states_test