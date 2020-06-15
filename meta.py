import torch
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import math
from dataset import *
from tqdm import tqdm
from model import *
import copy

from tqdm import tqdm
from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):

    def __init__(self, cfg, sampler):

        self.cfg = cfg
        self.sampler = sampler

    def get_model(self, name, params):

        if name == "logit":
            return LogisticRegression(**params)
        elif name == "ridge":
            return RidgeClassifier(**params)
        elif name == "svm":
            return SVC(**params)
        elif name == "rf":
            return RandomForestClassifier(**params)
        elif name == "gini":
            return GiniClassifier(**params)
        elif name == "lda":
            return LinearDiscriminantAnalysis(**params)
        elif name == "knn":
            return KNeighborsClassifier(**params)
        else:
            raise "Wrong model name"

    def train(self, n_way, k_shot, query_size, n_tasks=100, shuffle=True):

        is_gini = False

        names = self.cfg["model"]
        params = self.cfg["model_parameters"]

        if (isinstance(names, list) and isinstance(params, list)) or (isinstance(names, tuple) and isinstance(params, tuple)):
            assert len(names) == len(params), "N model != N model_parameters"
            models = [self.get_model(name, param)
                      for name, param in zip(names, params)]
            if "gini" in names:
                is_gini = True
        else:
            models = self.get_model(names, params)
            if names == "gini":
                is_gini = True

        tasks = self.sampler.sample_task_dataset(
            n_way, k_shot, query_size, n_tasks, shuffle=True)

        accs = []
        t = tqdm(tasks)
        for task in t:

            X_train, X_test, y_train, y_test = task
            
            try:
                X_train, X_test = self.process_data(X_train, X_test, self.cfg["add_similarity"])
            except:
                continue
            
            y_train, y_test = np.array(y_train), np.array(y_test)

            if self.cfg["add_similarity"]:
                y_train = np.concatenate((y_train, y_train))

            if self.cfg["reduction"] or is_gini:
                pca = PCA(n_components=n_way*k_shot-1)
                pca.fit(X_train, y_train)
                X_train, X_test = pca.transform(X_train), pca.transform(X_test)

            if isinstance(models, list):
                models = VotingClassifier(
                    estimators=[(name, m) for name, m in zip(names, models)], 
                    voting="hard")

            models.fit(X_train, y_train)
            accs.append(models.score(X_test, y_test))

            t.set_description(self.cfg["dataset"] + " %g way" % n_way + " %g shot " %
                              k_shot + '(Avg. score = %g)' % round(np.array(accs).mean(), 4))

    def process_data(self, sentences_train, sentences_test, add_similarity=False):

        vocabulary_train, vocabulary_test = self.get_task_vocabulary(
            sentences_train, sentences_test)

        self.set_similarity_dict(vocabulary_train, vocabulary_test)

        X_train = self.get_themed_matrix(
            sentences_train, vocabulary_train)
        X_test = self.get_themed_matrix(sentences_test, vocabulary_test)

        if add_similarity:
            X_train_sim = self.get_themed_matrix(sentences_train, vocabulary_train, use_similarity=True)
            X_train = torch.cat((X_train, X_train_sim), dim=0)

        return X_train.cpu().numpy(), X_test.cpu().numpy()

    def get_task_vocabulary(self, sentences_train, sentences_test, use_global_vocabulary=False):

        irrelevant_vocabulary = self.sampler.irrelevant_vocabulary

        vocabulary_train = CountVectorizer(
            stop_words='english').fit(sentences_train).get_feature_names()

        vocabulary_train = [
            word for word in vocabulary_train if word not in irrelevant_vocabulary and not word.isdigit()]

        vocabulary_test = CountVectorizer(
            stop_words='english').fit(sentences_test).get_feature_names()

        vocabulary_test = [
            word for word in vocabulary_test if word not in irrelevant_vocabulary and not word.isdigit()]

        if use_global_vocabulary:
            vocabulary_train = list(
                set(vocabulary_train + self.sampler.vocabulary))
            vocabulary_test = list(
                set(vocabulary_test + self.sampler.vocabulary + vocabulary_train))

        return vocabulary_train, vocabulary_test

    def set_similarity_dict(self, vocabulary_train, vocabulary_test, use_global_vocabulary=True):

        if not use_global_vocabulary:
            vocabulary_train = list(
                set(vocabulary_train) - set(self.sampler.vocabulary))
            vocabulary_test = list(
                set(vocabulary_test) - set(self.sampler.vocabulary))

        embedded_vocabulary_train = torch.cat(
            self.sampler.embedder.embed_dataset(vocabulary_train), dim=0)
        embedded_vocabulary_test = torch.cat(
            self.sampler.embedder.embed_dataset(vocabulary_test), dim=0)

        similarity = cosine(embedded_vocabulary_test,
                            embedded_vocabulary_train)

        indices = torch.argmax(similarity, dim=-1)
        self.similarity_dict = {}
        for i, indice in enumerate(indices):

            if vocabulary_test[i] not in vocabulary_train:
                self.similarity_dict[vocabulary_test[i]
                                     ] = vocabulary_train[indice]

        self.reversed_similarity_dict = {
            v: k for k, v in self.similarity_dict.items()}

    def get_themed_matrix(self, sentences, vocabulary, use_similarity=False, reversed_dict=False, similarity_dict=None):

        factor = self.cfg["weighting_factor"]
        if similarity_dict is None:
            if reversed_dict:
                similarity_dict = copy.deepcopy(self.reversed_similarity_dict)
            else:
                similarity_dict = copy.deepcopy(self.similarity_dict)

        outputs, probs = [], []
        for sentence in sentences:

            if use_similarity:
                words = []
                for word in sentence.split():
                    if word in similarity_dict:
                        words += [similarity_dict[word]]
                    elif word in vocabulary:
                        words += [word]
            else:
                words = [word for word in sentence.split()
                         if word in vocabulary]
            
            prob = [1-self.sampler.prob_vocabulary[w]
                    if w in self.sampler.prob_vocabulary else 1. for w in words]
            prob = torch.tensor(prob).unsqueeze(0)*factor

            outputs.append(
                torch.cat(self.sampler.embedder.embed_dataset(words)))
            probs.append(torch.softmax(prob, dim=-1))

        outputs = [prob @ output for prob, output in zip(probs, outputs)]
        #outputs = [output.mean(0, keepdim=True) for output in outputs]
        return torch.cat(outputs)
