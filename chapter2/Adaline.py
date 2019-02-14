# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import seed
class AdalineGD(object):
    """ADALINE分類機を実装してみる
    パラメータ一覧
    ------------------
    eta: float
        学習率(0~1)
    n_iter: int
        トレーニング回数（エポック数？）

    属性
    ------------------
    w_ :1次元配列
        適合後の重み
    errors_: リスト
    各エポックでの誤分類数

    """
    def __init__(self,eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """学習用メソッド

        パラメータ
        ------------------
        X: {配列みたいなもの},shape =[n_samples,n_features]
        y: {配列みたいなもの},shape =[n_samples]
            目的変数，つまりはラベル？

        戻り値
        ------------------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_=[]
        self.errors_ = []

        for i in range(self.n_iter): #niter分学習を繰り返しerrorsとwを更新
            #活性化関数の出力
            output = self.net_input(X)
            #誤差
            errors = y - output
            #重みの更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #コストの計算
            cost = (errors**2).sum()/2.0
            
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """総入力の計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self,X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

class AdalineSGD(object):
    """ADALINE分類機を実装してみる
    パラメータ一覧
    ------------------
    eta: float
        学習率(0~1)
    n_iter: int
        トレーニング回数（エポック数？）
    shuffle:
     循環を回避するためにエポックごとにデータを並び替える

    random_state: int
        shahhurunitukau 
        rランダムステートの指定と重みの初期化

    属性
    ------------------
    w_ :1次元配列
        適合後の重み
    errors_: リスト
    各エポックでの誤分類数


    """
    def __init__(self,eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """学習用メソッド

        パラメータ
        ------------------
        X: {配列みたいなもの},shape =[n_samples,n_features]
        y: {配列みたいなもの},shape =[n_samples]
            目的変数，つまりはラベル？

        戻り値
        ------------------
        self : object

        """
        #重みベクトルを作る
        self._initialize_weights(X.shape[1])
        self.cost_=[]

        for i in range(self.n_iter): #niter分学習を繰り返しerrorsとwを更新
            if self.shuffle:
                X, y=self._shuffle(X,y)
            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi,target))

            avg_cost =sum(cost)/len(y)
            self.cost_.append(avg_cost)

        return self
    
    def partial_fit(self, X, y):
        """再初期化しないでfit"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
            if y.ravel().shape[0] > 1:
                for xi, target in zip(X, y):
                    self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self,X,y):
        r =np.random.permutation(len(y))
        return X[r], y[r]
    def _initialize_weights(self, m):
        self.w_ =np.zeros(1+m)
        self.w_initialized=True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """総入力の計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self,X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
