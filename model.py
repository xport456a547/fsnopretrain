import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.linalg import eig
from sklearn.base import BaseEstimator, ClassifierMixin

def cosine(a, b):
    return (a @ b.transpose(0, -1)) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True).transpose(0, -1)+10e-8)

class GiniClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=0.):

        self.alpha = alpha

    def _preprocess(self, y):

        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        return (y * 2 - 1) * 10

    def fit(self, X, y):

        y = self._preprocess(y)
        self._fit(X, y)
        return self

    def _fit(self, X, y):

        n, d = X.shape
        rank = X.argsort(axis=0).argsort(axis=0) + 1

        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        rank = np.concatenate((np.ones((n, 1)), rank), axis=1)

        self.coefficient_ = np.linalg.inv(
            rank.T @ X + self.alpha*np.identity(d + 1)) @ rank.T @ y

        self.intercept_ = self.coefficient_[0, :]
        self.coefficient_ = self.coefficient_[1:].T

    def predict(self, X):

        pred = X @ self.coefficient_.T + self.intercept_[np.newaxis, :]
        return np.argmax(pred, axis=-1)
    
    def score(self, X, y):

        return accuracy_score(y, self.predict(X))

class PLSLogit(object):

    def __init__(self, n_components=3, norm=2, alpha=0., normalize=True):

        self.n_comp = n_components
        self.norm = norm
        self.alpha = alpha
        self.normalize = normalize

    def _preprocess(self, X, y):
        
        assert isinstance(y, np.ndarray) and isinstance(X, np.ndarray), "Expect numpy.ndarray input"
        assert X.shape[0] == y.shape[0], "Samples sizes don't match"
        
        X, y = X.astype(float), y.astype(float)
        if len(y.shape)  == 1: y = y.reshape((y.shape[0],1))
        if len(X.shape) == 1: X = X.reshape((X.shape[0],1))

        if y.shape[1] == 1: 
            if not np.array_equal(np.unique(y[:,0]), np.array([0.,1.])): 
                y = OneHotEncoder(categories='auto', sparse=False).fit_transform(y)
                if y.shape[1] == 2:
                    y = y[:,0].reshape((y.shape[0],1))

        if y.shape[1] == 1:
            self._type = "binary"
        else:
            self._type = "multiclass"
        
        n_1 = y.sum(0).reshape((1,y.shape[1]))
        n_0 = y.shape[0] - n_1

        if self.normalize: 
            sd = StandardScaler()
            X = sd.fit_transform(X)
            self._mean, self._std = sd.mean_, np.sqrt(sd.var_)
            self._std[self._std == 0.] = 1
        
        return X, y, n_1, n_0

    def fit(self, X, y):

        X, y, n_1, n_0 = self._preprocess(X, y)
        comp, self.weights_ = self._fit(X, y, n_1, n_0)

        return self

    def _fit(self, X, *args):
        
        components, W = [], []
        base = np.eye(X.shape[1])
		
        for i in range(self.n_comp):
            
            comp, w = self._compute_component(X, *args)
            components.append(comp)
            W.append(base.dot(w))
            X, beta = self._deflate(X, comp)
            base = base.dot(np.eye(X.shape[1]) - w.dot(beta))

        W = np.concatenate(W, axis=1)
        components = np.concatenate(components, axis=1)
    
        return components, W
    
    def _deflate(self, X, comp):
	
        comp = comp[:,-1].reshape(comp.shape[0],1)
        beta = np.linalg.inv(comp.T.dot(comp) + self.alpha).dot(comp.T).dot(X)
        return X - comp.dot(beta), beta
    
    def _compute_component(self, X, y, n_1, n_0):

        if self._type == "binary":
            w = (y/n_1 - (1-y)/n_0).T.dot(X)
            w = w.T / np.linalg.norm(w, self.norm)
        else:
            w = (y/n_1 - (1-y)/n_0).T.dot(X)
            w = w.T.dot(w)
            w, vr = eig(w)
            w = vr[:, np.argmax(w)].reshape((X.shape[1],1)).astype(float)
            w = w / np.linalg.norm(w, self.norm)

        return X.dot(w), w

    def fit_transform(self, X, y):

        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):

        assert self.weights_.any(), "Fit before transform"
        if self.normalize:
            X = (X - self._mean)/self._std
        return X.dot(self.weights_)


class FastPLSLogit(object):

    def __init__(self, n_components=3, norm=2, normalize=True):

        self.n_comp = n_components
        self._norm = norm
        self.normalize = normalize

    def _preprocess(self, X, y):
        
        assert isinstance(y, np.ndarray) and isinstance(X, np.ndarray), "Expect numpy.ndarray input"
        assert X.shape[0] == y.shape[0], "Samples sizes don't match"
        
        X, y = X.astype(float), y.astype(float)
        if len(y.shape)  == 1: y = y.reshape((y.shape[0],1))
        if len(X.shape) == 1: X = X.reshape((X.shape[0],1))

        y = OneHotEncoder(categories='auto', sparse=False).fit_transform(y)

        n_1 = y.sum(0).reshape((1,y.shape[1]))
        n_0 = y.shape[0] - n_1

        if self.normalize: 
            sd = StandardScaler()
            X = sd.fit_transform(X)
            self._mean, self._std = sd.mean_, np.sqrt(sd.var_)
            self._std[self._std == 0.] = 1

        return X, y, n_1, n_0

    def fit(self, X, y):

        assert self.n_comp < X.shape[1], "n_comp must be < X.shape[1]"
        X, y, n_1, n_0 = self._preprocess(X, y)
        comp, self.weights_ = self._compute_component(X, y, n_1, n_0)

        return None
    
    def _compute_component(self, X, y, n_1, n_0):

        w = (y/n_1 - (1-y)/n_0).T.dot(X)
        w = w.T.dot(w)
        w, vr = eig(w)
        idx = w.argsort()[::-1][:self.n_comp]  
        w = vr[:, idx].astype(float)
        w = w / np.linalg.norm(w, self._norm)

        return X.dot(w), w

    def fit_transform(self, X, y):

        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):

        assert self.weights_.any(), "Fit before transform"
        if self.normalize:
            X = (X - self._mean)/self._std
        return X.dot(self.weights_)
    