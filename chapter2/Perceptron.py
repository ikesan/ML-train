# -*- coding: utf-8 -*-
import numpy as np
class Perceptron(object):
    """パーセプトロン分類機を実装してみる
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
        self.errors_ = []

        for _ in range(self.n_iter): #niter分学習を繰り返しerrorsとwを更新
            errors = 0
            for xi, target in zip(X,y):#サンプルごとに重みの更新
                update = self.eta *(target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        """総入力の計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self,X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
