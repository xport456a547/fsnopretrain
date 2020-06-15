import torch
import numpy as np
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, BytePairEmbeddings, StackedEmbeddings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import random

class Dataset(object):

    def __init__(self, data, *args, embedder, use_validation=True, top_words=100):

        # Indices
        self.train, self.val, self.test = args
        self.top_words = top_words
        self.embedder = embedder

        # Data
        if use_validation:
            self.train_data = self._build_dataset(
                data, self.train)
            self.val_data = self._build_dataset(data, self.val)
        else:
            self.train += self.val
            self.train_data = self._build_dataset(
                data, self.train)
            self.val_data = None

        self.test_data = self._build_dataset(
            data, self.test)
        
    def _build_dataset(self, data, indexes):
        dataset = {}
        for index in indexes:
            labeled_data = data[str(index)]
            dataset[str(index)] = labeled_data.copy()
        return dataset

    def sample_task_dataset(self, n_way=5, k_shot=5, query_size=5, n_task=1, shuffle=True):
        
        tasks = [self.sample_task(
            n_way, k_shot, query_size, shuffle) for _ in range(n_task)]
        
        if n_task == 1:
            return tasks[0]
            
        return tasks

    def sample_task(self, n_way, k_shot, query_size, shuffle=True, seed=0):
        
        classes = sorted(random.sample(self.test, k=n_way))
        X_train, X_test, y_train, y_test = self.pick_samples(
            self.test_data, classes, k_shot, query_size, shuffle)

        return X_train, X_test, y_train, y_test

    def pick_samples(self, data, classes, k_shot, query_size, shuffle):

        support, query = [], []
        support_labels, query_labels = [], []

        for i, c in enumerate(classes):

            s, q = train_test_split(data[str(c)], train_size=k_shot, test_size=query_size, random_state=random.randint(1, 100000))
            support += s
            query += q

            support_labels += [i for j in range(k_shot)]
            query_labels += [i for j in range(query_size)]

        if shuffle:
            c = list(zip(support, support_labels))
            random.shuffle(c)
            support, support_labels = zip(*c)

            c = list(zip(query, query_labels))
            random.shuffle(c)
            query, query_labels = zip(*c)

        return support, query, support_labels, query_labels

    def embed_dataset(self):

        for data in [self.test_data]:
            if data is not None:
                for key, value in data.items():
                    print("Class: ", key)
                    value = list(zip(self.embedder.embed_dataset(value), value))

    def embed_counter(self):

        sentences = []
        indexes = []

        data = self.train_data
        for key, value in data.items():
            indexes.append(len(sentences)+np.arange(len(value)))
            sentences += value
        

        vectorizer = CountVectorizer(min_df=0.01, max_df=0.99, binary=True, ngram_range=(1,1), stop_words='english')
        #vectorizer = CountVectorizer(max_df=0.99, binary=True, ngram_range=(1,1), stop_words='english')
        tf = vectorizer.fit_transform(sentences).toarray()
        
        self.full_vocabulary = vectorizer.get_feature_names()

        self.prob_vocabulary = {voc: tf[:,i].mean() for i, voc in enumerate(self.full_vocabulary)}

        self.vocabulary = self.get_top_vocabulary(tf, self.full_vocabulary, indexes)
        self.vectorizer = CountVectorizer(vocabulary=self.vocabulary, binary=True)

        self.irrelevant_vocabulary = list(set(self.full_vocabulary) - set(self.vocabulary))

    def get_top_vocabulary(self, idf, voc, indexes):
        vocabulary = []
        for i, ind in enumerate(indexes):
            current_indexes = indexes.copy()
            del current_indexes[i]

            current_indexes = [item for sublist in current_indexes for item in sublist]
            current_indexes = np.array(current_indexes)
            ind = np.array(ind).flatten()
            
            diff = np.abs(idf[ind].mean(0) - idf[current_indexes].mean(0))
            #diff = np.abs(np.log(idf[ind].mean(0) + 10e-8) - np.log(idf[current_indexes].mean(0) + 10e-8))
            words = diff.argsort()[-self.top_words:][::-1]
            
            vocabulary += [voc[w] for w in words]
        return list(set(vocabulary))

class Embedder(object):

    def __init__(self, embedding=None, method=None, batch_size=5):

        assert method in [None, "average"], "Bad method"
        self.method = method
        self.batch_size = batch_size

        if embedding is not None:
            self.embedding = StackedEmbeddings(embedding)
        else:
            self.embedding = StackedEmbeddings([
                #WordEmbeddings('glove'),
                #WordEmbeddings('en-news'),
                #BytePairEmbeddings('en'),
                WordEmbeddings('crawl')
            ])

    def embed_data(self, sentences):
        sentences = [Sentence(s) for s in sentences]
        self.embedding.embed(sentences)

        if self.method == "average":
            sentences = [torch.stack([word.embedding.detach().cpu() for word in s]).mean(
                0) for s in sentences]
        else:
            sentences = [torch.stack(
                [word.embedding.detach().cpu() for word in s]) for s in sentences]

        return sentences

    def embed_dataset(self, sentences):
        sentences = self.embed_data(sentences)
        return sentences


