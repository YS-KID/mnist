import sys, os
sys.path.append(os.curdir)
from dataset.mnist import load_mnist
import numpy as np
import math
import matplotlib.pyplot as plt
import random

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 3000
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

num_input = 784
num_hidden_1 = 100
num_hidden_2 = 100
num_output = 10

Input = x_batch
Ans = t_batch

epocks = 10000

eta = 0.5

W1 = np.random.normal(0, 0.01, [num_input, num_hidden_1])
b1 = np.zeros([1, num_hidden_1])
W2 = np.random.normal(0, 0.01, [num_hidden_1, num_hidden_2])
b2 = np.zeros([1, num_hidden_2])
W3 = np.random.normal(0, 0.01, [num_hidden_2, num_output])
b3 = np.zeros([1, num_output])

loss = []

dummy_ratio = 0.0

for i in range(10000):
  dummy_list = random.sample(range(0, 784), int(784*dummy_ratio))
  for j in range(len(dummy_list)):
    x_test[i][dummy_list[j]] = np.random.rand()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
  x_exp = np.exp(x)
  x_sum = np.sum(np.exp(x), axis=1)
  x_sum = np.asarray(np.asmatrix(x_sum).T)
  return x_exp / x_sum

for i in range(epocks):
  A = Input
  u1 = np.dot(A, W1) + b1
  B = sigmoid(u1)
  u2 = np.dot(B, W2) + b2
  C = sigmoid(u2)
  u3 = np.dot(C, W3) + b3
  D = softmax(u3)
  
  E = -np.sum(Ans*np.log(np.clip(D, 10e-5, 1.0))+(1-Ans)*np.log(np.clip(1-D, 10e-5, 1.0)))
  loss.append(E)
  
  delta_D = (D - Ans)
  delta_C = np.dot(delta_D, W3.T) * sigmoid_derivative(u2)
  delta_B = np.dot(delta_C, W2.T) * sigmoid_derivative(u1)

  m = Input.shape[0]

  W3 -= eta * np.dot(C.T, delta_D) / m
  b3 -= eta * np.sum(delta_D, axis=0) / m
  W2 -= eta * np.dot(B.T, delta_C) / m
  b2 -= eta * np.sum(delta_C, axis=0) / m
  W1 -= eta * np.dot(A.T, delta_B) / m
  b1 -= eta * np.sum(delta_B, axis=0) / m

  print(i, end=" ")
  print(E)

print("----------------------------------------")
print("学習係数", end=": ")
print(eta)
print("ミニバッチ数", end=": ")
print(batch_size)
print("epocks", end=": ")
print(epocks)
print("loss", end=": ")
print(loss[-1])

D_index = D.argmax(axis=1)
Ans_index = Ans.argmax(axis=1)
print("----------------------------------------")
print("ノイズなし訓練データ")
print("----------------------------------------")
print("教師データに対する正解数", end=": ")
print(sum(D_index == Ans_index))
print("教師データに対する不正解数", end=": ")
print(sum(D_index != Ans_index))
print("教師データに対する正解率", end=": ")
print(sum(D_index == Ans_index)/D_index.shape[0])

A_test = x_test
u1_test = np.dot(A_test, W1) + b1
B_test = sigmoid(u1_test) 
u2_test = np.dot(B_test, W2) + b2
C_test = sigmoid(u2_test) 
u3_test = np.dot(C_test, W3) + b3
D_test = softmax(u3_test)
D_test_index = D_test.argmax(axis=1)
t_test_index = t_test.argmax(axis=1)
print("----------------------------------------")
print("ノイズありテストデータ") 
print("----------------------------------------")
print("テストデータに対する正解数", end=": ")
print(sum(D_test_index == t_test_index))
print("テストデータに対する不正解数", end=": ")
print(sum(D_test_index != t_test_index))
print("テストデータに対する正解率", end=": ")
print(sum(D_test_index == t_test_index)/D_test_index.shape[0])
print("----------------------------------------")
print("ノイズの割合", end=": ")
print(dummy_ratio)

plt.plot(loss)
plt.show()
