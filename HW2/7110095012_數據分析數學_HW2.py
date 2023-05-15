import h5py 
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 機器學習
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# image quality index
from skimage.metrics import mean_squared_error as mse
# HOSVD
import tensorly as tl
from tensorly.decomposition import tucker
# 計算執行時間
import time
# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# tf
import tensorflow as tf
from keras.utils.vis_utils import plot_model
# mnist
from keras.datasets import mnist
# store sklearn model
import joblib

import cv2
from sklearn.model_selection import train_test_split
from PIL import Image


def dataset_show_plot(X, y):
    SIZE = int(np.sqrt(X.shape[1]))
    num_classes = len(np.unique(y))
    num_samples = 5
    
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(num_classes, num_samples, sharex = True, sharey = True ,
                           figsize = (num_samples, num_classes))
    fig.subplots_adjust(hspace=0.1, wspace=0)
    for i in range(num_classes):
        for j in range(-num_samples, 0):
            image = X[y == i][j].reshape(SIZE, SIZE)
            ax[i, j].imshow(image, cmap = 'gray')
            ax[i, j].set_axis_off()
            
def integrated_plot(image, label_list, presiction_list):
    plt.rcParams.update({'font.size': 20})
    index = 0
    set_range = int(np.ceil(len(image.keys()) / 9))
    for pages in range(set_range):
        fig, ax = plt.subplots(3, 3, sharex = True, sharey = True ,
                               figsize = (3, 3))
        fig.set_size_inches(20, 20)
        fig.subplots_adjust(hspace=0.2, wspace=0)
        for i in range(3):
            for j in range(3):
                if index in image.keys():
                    ax[i, j].imshow(image[index], cmap = 'gray')
                    ax[i, j].set_title(f'Prediction = {presiction_list[index]}', 
                                       fontsize = 40)  
                    index += 1
                
def myself_HOSVD(Tensor, rank):
    U  =[]
    for i in range(len(rank)):
        U_i, Sigma_i, V_T_i = np.linalg.svd(tl.unfold(Tensor, mode = i))
        U.append(U_i)
    S = tl.tenalg.multi_mode_dot(Tensor, U, list(range(len(rank))), transpose = True)
    return S, U

def myHOSVD_(Tensor, rank):
    U = [np.linalg.svd(tl.unfold(Tensor,mode=i))[0] for i in range(len(rank))]
    S = tl.tenalg.multi_mode_dot(Tensor, U, list(range(len(rank))),transpose=True)
    return S, U

def myHOOI(Tensor, rank, max_iter = 5, max_err = 1e-7):
    G, U = myself_HOSVD(Tensor, rank)
    for epoch in range(max_iter):
        for i in range(len(rank)):
            Y = tl.tenalg.multi_mode_dot(Tensor, U, skip = i, modes = list(range(len(rank))), transpose = True)
            tmpU, _, _ = np.linalg.svd(tl.unfold(Y, mode = i))
            U[i] = tmpU[:rank[i]]
            # 計算error
            T_hat = tl.tenalg.multi_mode_dot(G, U, list(range(len(rank))))
            error = tl.norm(Tensor - T_hat)
            if error < max_err:
                break
    G = tl.tenalg.multi_mode_dot(Tensor, U, modes = list(range(len(rank))), transpose = True)
    return G, U

# load usps dataset with hdf5 format
def hdf5(path, data_key = "data", target_key = "target", flatten = True):
    """
        loads data from hdf5: 
        - hdf5 should have 'train' and 'test' groups 
        - each group should have 'data' and 'target' dataset or spcify the key
        - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        train_X = train.get(data_key)[:]
        train_y = train.get(target_key)[:]
        test = hf.get('test')
        test_X = test.get(data_key)[:]
        test_y = test.get(target_key)[:]
        if flatten:
            train_X = train_X.reshape(train_X.shape[0], reduce(lambda a, b: a * b, train_X.shape[1:]))
            test_X = test_X.reshape(test_X.shape[0], reduce(lambda a, b: a * b, test_X.shape[1:]))
    
    return train_X, train_y, test_X, test_y


def dataset_loading(dataset_type = 'usps', isdraw = False):
    if dataset_type == 'usps':
        train_X, train_y, test_X, test_y = hdf5(f'./data/{dataset_type}/usps.h5')
        new_train_X = np.empty_like(train_X)        
    elif dataset_type == 'sign_mnist':
        sign_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 
                     6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 
                     12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
                     18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
        num_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 
                    6: 6, 7: 7, 8: 8, 10: 9, 11: 10, 12: 11, 
                    13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17,
                    19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23}
        
        dataset_type = 'sign_mnist'
        sign_mnist_train = pd.read_csv(f'./data/{dataset_type}/sign_mnist_train.csv')
        sign_mnist_test = pd.read_csv(f'./data/{dataset_type}/sign_mnist_test.csv')
        

        train_X = np.array(sign_mnist_train.iloc[:, 1:].values, dtype = float)
        test_X = np.array(sign_mnist_test.iloc[:, 1:].values, dtype = float)
        train_y = np.array(sign_mnist_train['label'].map(num_dict), dtype = int)
        test_y = np.array(sign_mnist_test['label'].map(num_dict), dtype = int)
        
        # normalization
        train_X /= 255
        test_X /= 255   
    else:
        return ''
        
    # 圖片尺寸    
    IMAGE_SIZE = int(np.sqrt(train_X.shape[1]))
    classes_num = int(np.unique(train_y).shape[0])
    if isdraw == True:
        train_count = []
        test_count = [] 
        for label in np.unique(train_y):
            train_count.append(len(train_y[train_y == label]))
            test_count.append(len(test_y[test_y == label]))
        # draw bar plot
        plt.xticks(range(0, classes_num))
        plt.bar(np.unique(train_y), train_count, color = 'blue')
        for index, value in enumerate(train_count):
            percentage = round(value / len(train_y) * 100, 2)
            plt.text(index - 0.5, value + 0.1, str(percentage) + '%')
        plt.xlabel('digit')
        plt.ylabel('percentage')
        plt.title(f'The distribution of [Training] dataset ({dataset_type})')
        plt.show()
        print(f'triaing: {train_count}')
        plt.xticks(range(0, classes_num))
        plt.bar(np.unique(test_y), test_count, color = 'orange')
        for index, value in enumerate(test_count):
            percentage = round(value / len(test_y) * 100, 2)
            plt.text(index - 0.5, value + 0.1, str(percentage) + '%')
        plt.xlabel('digit')
        plt.ylabel('percentage')
        plt.title(f'The distribution of [Test] dataset ({dataset_type})')
        plt.show()
        print(f'test: {test_count}')
        print('*' * 50)
        print(f'total: { [sum(x) for x in zip(train_count, test_count)]}')
    return train_X, train_y, test_X, test_y

# Means method
class MeansMethod:
    # Constructor
    def __init__(self):
        self.train_X_sum_dict = {}
        self.train_X_mean_dict = {}
        for label in np.unique(train_y):
            self.train_X_sum_dict[label] = 0
            self.train_X_mean_dict[label] = 0
    # Method
    def fit(self, train_X, train_y):
        self.IMAGE_SIZE = int(np.sqrt(train_X.shape[1]))
        self.classes_number = int(np.unique(train_y).shape[0])
        # summation
        for y_index, label in enumerate(train_y):
            self.train_X_sum_dict[label] += train_X[y_index]
        # means
        for label in np.unique(train_y):
            self.train_X_mean_dict[label] = self.train_X_sum_dict[label] / len(train_y[train_y == label])
        # draw means
        flag = False
        if flag == True:
            plt.rcParams.update({'font.size': 20})
            fig, ax = plt.subplots(2, 5, sharex = True, sharey = True ,
                                   figsize = (20, 8))
            fig.subplots_adjust(hspace=0.4, wspace=0)
            count = 0
            for i in range(2):
                for j in range(5):
                    image = self.train_X_mean_dict[count].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                    ax[i, j].imshow(image, cmap = 'gray')
                    ax[i, j].set_axis_off()
                    ax[i, j].set_title(f'Means = {count}', fontsize = 30)  
                    count += 1    
            plt.show()
    def predict(self, test_X, norm_type = 2):
        # for plot
        self.test_residual = {}
        for test_index in range(test_X.shape[0]):
            self.test_residual[test_index] = [] 
        # for prediction
        predict_test_y = []
        for test_index in range(test_X.shape[0]): 
            predict_label = np.inf
            min_residual = np.inf
            for label in np.unique(train_y):
                category_mean = self.train_X_mean_dict[label].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                z = test_X[test_index].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                residual = np.linalg.norm(category_mean - z, norm_type) 
                self.test_residual[test_index].append(residual) 
                if min_residual > residual:
                    min_residual = residual
                    predict_label = label
            predict_test_y.append(predict_label)
        predict_test_y = np.array(predict_test_y)
        return predict_test_y
    
    # 畫 digit vs residual
    def draw_digitvs_residual(self, test_X, test_y):
        predict_test_y = self.predict(test_X)
        # Training record
        test_y_category = {}
        for label in np.unique(train_y):
            test_y_category[label] = len(test_y[test_y == label])
        # Test record
        test_TP_dict = {}
        for label in np.unique(test_y):
            test_TP_dict[label] = 0
        for test_index in range(test_y.shape[0]): 
            predict_label = predict_test_y[test_index]
            if predict_label == test_y[test_index]:
                test_TP_dict[predict_label] += 1 
            category_acc = []
            for label in np.unique(test_y):
                acc = round((test_TP_dict[label] / test_y_category[label]) * 100, 2)
                category_acc.append(acc)
        best_acc_digit = np.argmax(category_acc)
        worst_acc_digit = np.argmin(category_acc)
        plt.rcParams.update({'font.size': 20})

        fig, ax = plt.subplots(2, 5, sharex = False, sharey = True ,
                                   figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        count = 0
        for i in range(2):
            for j in range(5):
                new_test_X = test_X[np.where(test_y == count)]
                self.predict(new_test_X)
                for new_test_index in range(new_test_X.shape[0]):
                    if count == best_acc_digit:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'red')
                    elif count == worst_acc_digit:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'blue')
                    else:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'Brown')
                    ax[i, j].set_title(f'Digit {count} ({category_acc[count]} %)', fontsize = 20, weight = 'bold')  
                    ax[i, j].set_xlabel('Basis')  
                    ax[i, j].set_ylabel('Residual') 
                    ax[i, j].set_xticks(range(0, self.classes_number))
                    ax[i, j].set_yticks(range(0, 12, 2))
                count += 1   
        plt.show()

# SVD method
class SVDMethod:
    # Constructor
    def __init__(self, decide_K = [12 for i in range(10)]):
        self.decide_K = decide_K
        self.train_X_dict = {}
        for label in np.unique(train_y):
            self.train_X_dict[label] = []
        for y_index, label in enumerate(train_y):
            self.train_X_dict[label].append(train_X[y_index])
        for label in np.unique(train_y):
            self.train_X_dict[label] = np.array(self.train_X_dict[label])
    # Method
    def fit(self, train_X, train_y):
        self.IMAGE_SIZE = int(np.sqrt(train_X.shape[1]))
        self.train_category_SVD_U = {}
        self.class_number = int(np.unique(train_y).shape[0])
        # get U
        for label in np.unique(train_y):
            U, Sigma, V_T = np.linalg.svd(self.train_X_dict[label].T)
            self.train_category_SVD_U[label] = U
    
    def predict(self, test_X, norm_type = 2):
        # for plot
        self.test_residual = {}
        for test_index in range(test_X.shape[0]):
            self.test_residual[test_index] = [] 
        # for prediction
        predict_test_y = []
        for test_index in range(test_X.shape[0]): 
            predict_label = np.inf
            min_residual = np.inf
            for label in np.unique(train_y):
                z = test_X[test_index]
                U_k = self.train_category_SVD_U[label][:, :self.decide_K[label]]
                #U_k = np.identity(256)
                residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type) / np.linalg.norm(z, norm_type)
                #residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type)
                self.test_residual[test_index].append(residual) 
                if min_residual > residual:
                    min_residual = residual
                    predict_label = label     
            predict_test_y.append(predict_label)
        predict_test_y = np.array(predict_test_y)

        return predict_test_y
    
    
    # 畫 digit vs residual
    def draw_digitvs_residual(self, test_X, test_y):
        predict_test_y = self.predict(test_X)
        # Training record
        test_y_category = {}
        for label in np.unique(train_y):
            test_y_category[label] = len(test_y[test_y == label])
        # Test record
        test_TP_dict = {}
        for label in np.unique(test_y):
            test_TP_dict[label] = 0
        for test_index in range(test_y.shape[0]): 
            predict_label = predict_test_y[test_index]
            if predict_label == test_y[test_index]:
                test_TP_dict[predict_label] += 1 
            category_acc = []
            for label in np.unique(test_y):
                acc = round((test_TP_dict[label] / test_y_category[label]) * 100, 2)
                category_acc.append(acc)
        best_acc_digit = np.argmax(category_acc)
        worst_acc_digit = np.argmin(category_acc)
        # plot
        classes_num = int(np.unique(test_y).shape[0])
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(2, 5, sharex = False, sharey = True ,
                                   figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        count = 0
        for i in range(2):
            for j in range(5):
                new_test_X = test_X[np.where(test_y == count)]
                #print(count, new_test_X.shape)
                self.predict(new_test_X)
                for new_test_index in range(new_test_X.shape[0]):
                    if count == best_acc_digit:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'red')
                    elif count == worst_acc_digit:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'blue')
                    else:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'Brown')
                    #ax[i, j].set_axis_off()
                    ax[i, j].set_title(f'Digit {count} ({category_acc[count]} %)', fontsize = 20, weight = 'bold')  
                    ax[i, j].set_xlabel('Basis')  
                    ax[i, j].set_ylabel('Residual') 
                    ax[i, j].set_xticks(range(0, int(classes_num / 5)))
                    ax[i, j].set_yticks(np.arange(0, 1.2, 0.2))
                    #ax[i, j].grid()
                count += 1   
        plt.show()
        
    def svd_k_plot(self, X, y, name = 'Training', norm_type = 2): 
        # for plot
        K_range = [i for i in range(20)]
        mse_k = []
        acc_k = []
        for k in K_range:
            # for prediction
            predict_test_y = []
            for test_index in range(X.shape[0]): 
                predict_label = np.inf
                min_residual = np.inf
                for label in np.unique(y):
                    z = X[test_index]
                    U_k = self.train_category_SVD_U[label][:, :k+1]
                    residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type) / np.linalg.norm(z, norm_type)
                    if min_residual > residual:
                        min_residual = residual
                        predict_label = label     
                predict_test_y.append(predict_label)
            predict_test_y = np.array(predict_test_y)
            mse_k.append(mse(y, predict_test_y))
            acc_k.append(round((len([i for i, j in zip(y, predict_test_y) if i == j]) / len(y)) * 100, 2))
        # draw 
        print('[SVD]')
        fig, axs = plt.subplots(1, 2, figsize=(self.class_number, 5))
        # mse
        plt.subplot(1, 2, 1)
        plt.plot([i+1 for i in K_range], mse_k, '-o')
        plt.xticks([i+1 for i in K_range])
        plt.xlabel('K value')
        plt.ylabel('MSE')
        plt.title(f'[{name}] datset MSE (SVD)')
        plt.grid()
        print(f'{name} mse')
        for index, mse_ in enumerate(mse_k):
            print(f'K = {index+1}: {round(mse_, 4)}')
        print('*' * 50)
        # acc
        plt.subplot(1, 2, 2)
        plt.plot([i+1 for i in K_range], acc_k, '-o')
        plt.xticks([i+1 for i in K_range])
        plt.xlabel('K value')
        plt.ylabel('Accuracy')
        plt.title(f'[{name}] datset Accuracy (SVD)')
        plt.grid()
        plt.show()
        print(f'{name} acc')
        for index, acc_ in enumerate(acc_k):
            print(f'K = {index+1}: {acc_} %')
        print('*' * 50)
    def each_digit_svd_k_plot(self, X, y, name = 'Training', norm_type = 2): 
        # to collect each digit set
        test_digit_category_dict = {}
        for label in np.unique(y):
            test_digit_category_dict[label] = []
        for index, label in enumerate(y):
            test_digit_category_dict[label].append(X[index])
        for label in np.unique(y):
            test_digit_category_dict[label] = np.array(test_digit_category_dict[label])
            
        # for plot
        K_range = [i for i in range(20)]
        mse_digit_dict = {}
        acc_digit_dict = {}
        for label in np.unique(y):
            mse_digit_dict[label] = []
            acc_digit_dict[label] = []
        for digit in np.unique(y):  
            digit_X = test_digit_category_dict[digit]
            digit_y = np.full(test_digit_category_dict[digit].shape[0], digit)
            mse_k = []
            acc_k = []
            for k in K_range:
                # for prediction
                predict_test_y = []
                for index in range(digit_X.shape[0]): 
                    predict_label = np.inf
                    min_residual = np.inf
                    for label in np.unique(y):
                        z = digit_X[index]
                        U_k = self.train_category_SVD_U[label][:, :k+1]
                        residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type) / np.linalg.norm(z, norm_type)
                        if min_residual > residual:
                            min_residual = residual
                            predict_label = label     
                    predict_test_y.append(predict_label)
                predict_test_y = np.array(predict_test_y)
                mse_k.append(mse(digit_y, predict_test_y))
                acc_k.append(round((len([i for i, j in zip(digit_y, predict_test_y) if i == j]) / len(digit_y)) * 100, 2))
            mse_digit_dict[digit] = mse_k
            acc_digit_dict[digit] = acc_k
        #######################################################################
        # plot
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(2, 5, sharex = False, sharey = False,
                                   figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        # MSE
        count = 0
        for i in range(2):
            for j in range(5):
                ax[i, j].plot([i+1 for i in K_range], mse_digit_dict[count], '-o')
                ax[i, j].set_title(f'Digit = {count}', fontsize = 20, weight = 'bold')  
                ax[i, j].set_xlabel('K value')  
                if (j == 0):
                    ax[i, j].set_ylabel('MSE') 
                ax[i, j].set_xticks([i for i in range(5, 25, 5)] + [1])
                ax[i, j].set_yticks([i for i in np.arange(0, int(max([np.max(mse_digit_dict[label]) for label in range(self.class_number)])) + 1.5, 1)])
                ax[i, j].grid()
                count += 1  
        fig.suptitle(f'[{name}] datset MSE (SVD)')
        plt.show()  
        # plot
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(2, 5, sharex = False, sharey = False,
                                   figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        plot_x_max = int(max([np.max(acc_digit_dict[label]) for label in range(self.class_number)]))
        plot_x_min = int(min([np.min(acc_digit_dict[label]) for label in range(self.class_number)]))
        count = 0
        for i in range(2):
            for j in range(5):
                # MSE
                ax[i, j].plot([i+1 for i in K_range], acc_digit_dict[count], '-o')
                ax[i, j].set_title(f'Digit = {count}', fontsize = 20, weight = 'bold')  
                ax[i, j].set_xlabel('K value')  
                if (j == 0):
                    ax[i, j].set_ylabel('Accuracy') 
                ax[i, j].set_xticks([i for i in range(5, 25, 5)] + [1])
                ax[i, j].set_yticks([i for i in np.arange(plot_x_min,  100, 5)] + [100])
                ax[i, j].grid()
                count += 1  
        fig.suptitle(f'[{name}] datset Accuracy (SVD)')
        plt.show() 
        
        
        

###############################################################################
# HOSVD method
class HOSVDMethod:
    # Constructor
    def __init__(self, decide_K = [12 for i in range(10)]):
        self.decide_K = decide_K
    # Method
    def fit(self, train_X, train_y, decomposition_type = 'original'):
        # 訓練紀錄
        self.class_number = int(np.unique(train_y).shape[0])
        self.train_y_category = {}
        for label in np.unique(train_y):
            self.train_y_category[label] = len(train_y[train_y == label])
            
        self.IMAGE_SIZE = int(np.sqrt(train_X.shape[1]))
        self.train_X_dict = {}
        for label in np.unique(train_y):
            self.train_X_dict[label] = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, self.train_y_category[label]))
        train_y_category_count = {}
        for label in np.unique(train_y):
            train_y_category_count[label] = 0
        #train_y_category_count = train_y_category.copy() # 倒數更新
        for y_index, label in enumerate(train_y):
            if train_y_category_count[label] < self.train_y_category[label]:
                self.train_X_dict[label][:, :, train_y_category_count[label]] = train_X[y_index].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                train_y_category_count[label] += 1  
        for label in np.unique(train_y):
            self.train_X_dict[label] = tl.tensor(self.train_X_dict[label])
        # get S, G
        self.train_category_HOSVD_S = {}
        self.train_category_HOSVD_U = {}
        for label in np.unique(train_y):
            # using package or implementation
            if decomposition_type == 'original':
                core, factors = tucker(self.train_X_dict[label], rank=self.train_X_dict[label].shape) # same as from sktensor.tucker import hooi
            elif decomposition_type == 'myHOSVD':
                core, factors = myself_HOSVD(self.train_X_dict[label], rank=self.train_X_dict[label].shape) # same as from sktensor.tucker import hooi
            self.train_category_HOSVD_S[label] = core
            self.train_category_HOSVD_U[label] = factors  
    def predict(self, test_X, norm_type = 'fro', diff_ways = 'parallel_matrix', isdraw = False):
        # time: parallel_matrix < parallel_mode_dot < matrix < mode_dot
        # for plot
        self.test_residual = {}
        for test_index in range(test_X.shape[0]):
            self.test_residual[test_index] = [] 
        # for prediction
        predict_test_y = []
        for test_index in range(test_X.shape[0]): #test_X.shape[0] # range(5)
            predict_label = np.inf
            min_residual = np.inf
            for label in np.unique(train_y):
                z = test_X[test_index].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                S = self.train_category_HOSVD_S[label]
                U_k1, U_k2, U_k3 = self.train_category_HOSVD_U[label]
                if diff_ways == 'parallel_mode_dot':
                    #SU1U2_prime = np.dot(U_k1, S).dot(U_k2.T) # 錯誤作法
                    SU1U2 = tl.tenalg.multi_mode_dot(S,[U_k1, U_k2], [0, 1])
                    zAj = [np.sum(z*SU1U2[:, :, i]) for i in range(self.decide_K[label])]
                    AjAj = [np.sum(SU1U2[:, :, i]*SU1U2[:, :, i]) for i in range(self.decide_K[label])]
                    zj = [zAj[i] / AjAj[i] for i in range(self.decide_K[label])]
                    total_zjAj = [zj[i] * SU1U2[:, :, i] for i in range(self.decide_K[label])]
                    sum_zjAj = np.sum(total_zjAj, axis = 0)
                elif diff_ways == 'parallel_matrix':
                    SU1U2 = [np.dot(U_k1, S[:, :, i]).dot(U_k2.T) for i in range(self.decide_K[label])]
                    #SU1U2 = [np.linalg.multi_dot([U_k1, S[:, :, i], U_k2.T]) for i in range(self.decide_K[label])]
                    zAj = [np.sum(z*SU1U2[i]) for i in range(self.decide_K[label])]
                    AjAj = [np.sum(SU1U2[i]*SU1U2[i]) for i in range(self.decide_K[label])]
                    zj = [zAj[i] / AjAj[i] for i in range(self.decide_K[label])]
                    total_zjAj = [zj[i] * SU1U2[i] for i in range(self.decide_K[label])]
                    sum_zjAj = np.sum(total_zjAj, axis = 0)
                else:
                    sum_zjAj = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))
                    for j in range(self.decide_K[label]):
                        if diff_ways == 'mode_dot':
                            Aj = tl.tenalg.multi_mode_dot(S[:, :, j],[U_k1, U_k2], [0, 1])
                        elif diff_ways == 'matrix':
                            Aj = np.dot(U_k1, S[:, :, j]).dot(U_k2.T)
                        zj = np.tensordot(z, Aj) / np.tensordot(Aj, Aj)
                        sum_zjAj += np.dot(zj, Aj)
                    
                #residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type) / np.linalg.norm(z, norm_type)
                residual = np.linalg.norm(z - sum_zjAj, norm_type) / np.linalg.norm(z, norm_type)
                self.test_residual[test_index].append(residual)
                if min_residual > residual:
                    min_residual = residual
                    predict_label = label
            predict_test_y.append(predict_label)
        predict_test_y = np.array(predict_test_y)
        return predict_test_y
    # 畫 digit vs residual
    def draw_digitvs_residual(self, test_X, test_y):
        predict_test_y = self.predict(test_X)
        # Training record
        test_y_category = {}
        for label in np.unique(train_y):
            test_y_category[label] = len(test_y[test_y == label])
        # Test record
        test_TP_dict = {}
        for label in np.unique(test_y):
            test_TP_dict[label] = 0
        for test_index in range(test_y.shape[0]): 
            predict_label = predict_test_y[test_index]
            if predict_label == test_y[test_index]:
                test_TP_dict[predict_label] += 1 
            category_acc = []
            for label in np.unique(test_y):
                acc = round((test_TP_dict[label] / test_y_category[label]) * 100, 2)
                category_acc.append(acc)
        best_acc_digit = np.argmax(category_acc)
        worst_acc_digit = np.argmin(category_acc)
        # plot
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(2, 5, sharex = False, sharey = True ,
                                   figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        count = 0
        for i in range(2):
            for j in range(5):
                new_test_X = test_X[np.where(test_y == count)]
                #print(count, new_test_X.shape)
                self.predict(new_test_X)
                for new_test_index in range(new_test_X.shape[0]):
                    if count == best_acc_digit:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'red')
                    elif count == worst_acc_digit:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'blue')
                    else:
                        ax[i, j].plot(self.test_residual[new_test_index], color = 'Brown')
                    #ax[i, j].set_axis_off()
                    ax[i, j].set_title(f'Digit {count} ({category_acc[count]} %)', fontsize = 20, weight = 'bold')  
                    ax[i, j].set_xlabel('Basis')  
                    ax[i, j].set_ylabel('Residual') 
                    ax[i, j].set_xticks(range(0, self.class_number))
                    ax[i, j].set_yticks(np.arange(0, 1.2, 0.2))
                    #ax[i, j].grid()
                count += 1   
        plt.show()
    def hosvd_k_plot(self, X, y, name = 'Training', norm_type = 'fro'):
        # for plot
        K_range = [i for i in range(20)]
        mse_k = []
        acc_k = []
        for k in K_range:
            # for prediction
            predict_test_y = []
            for test_index in range(X.shape[0]): #test_X.shape[0] # range(5)
                predict_label = np.inf
                min_residual = np.inf
                for label in np.unique(train_y):
                    z = X[test_index].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                    S = self.train_category_HOSVD_S[label]
                    U_k1, U_k2, U_k3 = self.train_category_HOSVD_U[label]
                    SU1U2 = [np.dot(U_k1, S[:, :, i]).dot(U_k2.T) for i in range(k+1)]
                    #SU1U2 = [np.linalg.multi_dot([U_k1, S[:, :, i], U_k2.T]) for i in range(self.decide_K)]
                    zAj = [np.sum(z*SU1U2[i]) for i in range(k+1)]
                    AjAj = [np.sum(SU1U2[i]*SU1U2[i]) for i in range(k+1)]
                    zj = [zAj[i] / AjAj[i] for i in range(k+1)]
                    total_zjAj = [zj[i] * SU1U2[i] for i in range(k+1)]
                    sum_zjAj = np.sum(total_zjAj, axis = 0)
                    
                    #residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type) / np.linalg.norm(z, norm_type)
                    residual = np.linalg.norm(z - sum_zjAj, norm_type)
                    #print(label, residual)
                    if min_residual > residual:
                        min_residual = residual
                        predict_label = label
                predict_test_y.append(predict_label)
            predict_test_y = np.array(predict_test_y)
            mse_k.append(mse(y, predict_test_y))
            acc_k.append(round((len([i for i, j in zip(y, predict_test_y) if i == j]) / len(y)) * 100, 2))
        # draw 
        print('[HOSVD]')
        fig, axs = plt.subplots(1, 2, figsize=(self.class_number, 5))
        # mse
        plt.subplot(1, 2, 1)
        plt.plot([i+1 for i in K_range], mse_k, '-o')
        #for index, value in enumerate(mse_k):
        #    plt.text(index + 1.2, value - 0.2, str(round(value, 2)), color='red')
        plt.xticks([i+1 for i in K_range])
        plt.xlabel('K value')
        plt.ylabel('MSE')
        plt.title(f'[{name}] datset MSE (HOSVD)')
        plt.grid()
        #plt.show()
        print(f'{name} mse')
        for index, mse_ in enumerate(mse_k):
            print(f'K = {index+1}: {round(mse_, 4)}')
        print('*' * 50)
        # acc
        plt.subplot(1, 2, 2)
        plt.plot([i+1 for i in K_range], acc_k, '-o')
        #for index, value in enumerate(acc_k):
        #    plt.text(index + 1.2, value - 0.2, str(value) + '%', color='red')
        plt.xticks([i+1 for i in K_range])
        plt.xlabel('K value')
        plt.ylabel('Accuracy')
        plt.title(f'[{name}] datset Accuracy (HOSVD)')
        plt.grid()
        plt.show()
        print(f'{name} acc')
        for index, acc_ in enumerate(acc_k):
            print(f'K = {index+1}: {acc_} %')
        print('*' * 50)
    def each_digit_hosvd_k_plot(self, X, y, name = 'Training', norm_type = 'fro'):
        # to collect each digit set
        test_digit_category_dict = {}
        for label in np.unique(y):
            test_digit_category_dict[label] = []
        for index, label in enumerate(y):
            test_digit_category_dict[label].append(X[index])
        for label in np.unique(y):
            test_digit_category_dict[label] = np.array(test_digit_category_dict[label])
            
        # for plot
        K_range = [i for i in range(20)]
        mse_digit_dict = {}
        acc_digit_dict = {}
        for label in np.unique(y):
            mse_digit_dict[label] = []
            acc_digit_dict[label] = []
        for digit in np.unique(y):  
            digit_X = test_digit_category_dict[digit]
            digit_y = np.full(test_digit_category_dict[digit].shape[0], digit)
            mse_k = []
            acc_k = []
            for k in K_range:
                # for prediction
                predict_test_y = []
                for test_index in range(digit_X.shape[0]): #test_X.shape[0] # range(5)
                    predict_label = np.inf
                    min_residual = np.inf
                    for label in np.unique(y):
                        z = digit_X[test_index].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE)
                        S = self.train_category_HOSVD_S[label]
                        U_k1, U_k2, U_k3 = self.train_category_HOSVD_U[label]
                        SU1U2 = [np.dot(U_k1, S[:, :, i]).dot(U_k2.T) for i in range(k+1)]
                        #SU1U2 = [np.linalg.multi_dot([U_k1, S[:, :, i], U_k2.T]) for i in range(self.decide_K)]
                        zAj = [np.sum(z*SU1U2[i]) for i in range(k+1)]
                        AjAj = [np.sum(SU1U2[i]*SU1U2[i]) for i in range(k+1)]
                        zj = [zAj[i] / AjAj[i] for i in range(k+1)]
                        total_zjAj = [zj[i] * SU1U2[i] for i in range(k+1)]
                        sum_zjAj = np.sum(total_zjAj, axis = 0)
                        
                        #residual = np.linalg.norm(z - np.dot(U_k, np.dot(U_k.T, z)), norm_type) / np.linalg.norm(z, norm_type)
                        residual = np.linalg.norm(z - sum_zjAj, norm_type)
                        #print(label, residual)
                        if min_residual > residual:
                            min_residual = residual
                            predict_label = label
                    predict_test_y.append(predict_label)
                predict_test_y = np.array(predict_test_y)
                mse_k.append(mse(digit_y, predict_test_y))
                acc_k.append(round((len([i for i, j in zip(digit_y, predict_test_y) if i == j]) / len(digit_y)) * 100, 2))
            mse_digit_dict[digit] = mse_k
            acc_digit_dict[digit] = acc_k
        #######################################################################
        # plot
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(2, 5, sharex = False, sharey = False,
                                       figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        # MSE
        count = 0
        for i in range(2):
            for j in range(5):
                ax[i, j].plot([i+1 for i in K_range], mse_digit_dict[count], '-o')
                #ax[i, j].set_axis_off()
                ax[i, j].set_title(f'Digit = {count}', fontsize = 20, weight = 'bold')  
                ax[i, j].set_xlabel('K value')  
                if (j == 0):
                    ax[i, j].set_ylabel('MSE') 
                ax[i, j].set_xticks([i for i in range(5, 25, 5)] + [1])
                ax[i, j].set_yticks([i for i in np.arange(0, int(max([np.max(mse_digit_dict[label]) for label in range(self.class_number)])) + 1.5, 1)])
                ax[i, j].grid()
                count += 1  
        fig.suptitle(f'[{name}] datset MSE (HOSVD)')
        plt.show()  
        # plot
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(2, 5, sharex = False, sharey = False,
                                       figsize = (20, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        plot_x_min = int(min([np.min(acc_digit_dict[label]) for label in range(self.class_number)]))
        count = 0
        for i in range(2):
            for j in range(5):
                # MSE
                ax[i, j].plot([i+1 for i in K_range], acc_digit_dict[count], '-o')
                #ax[i, j].set_axis_off()
                ax[i, j].set_title(f'Digit = {count}', fontsize = 20, weight = 'bold')  
                ax[i, j].set_xlabel('K value')  
                if (j == 0):
                    ax[i, j].set_ylabel('Accuracy') 
                ax[i, j].set_xticks([i for i in range(5, 25, 5)] + [1])
                ax[i, j].set_yticks([i for i in np.arange(plot_x_min,  100, 5)] + [100])
                ax[i, j].grid()
                count += 1  
        fig.suptitle(f'{name} datset Accuracy (HOSVD)')
        plt.show() 
###############################################################################
# ANN method
class ANNMethod:
    # Constructor
    def __init__(self):
        pass
    # Method
    def fit(self, train_X, train_y):
        self.class_number = int(np.unique(train_y).shape[0])
        self.IMAGE_SIZE = int(np.sqrt(train_X.shape[1]))
        self.class_number = len(np.unique(train_y))
        self.train_X = train_X
        self.train_y = train_y
        
        
        # define machine learning model
        
        # 最後一層為softmax, 搭配from_logits=False
        # sigmoid activation in the last Dense layer of your NN and set from_logits = False
        self.model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(128, activation='relu', name = "Hidden_Layer1"),
              tf.keras.layers.Dense(64, activation='relu', name = "Hidden_Layer2"),
              tf.keras.layers.Dense(32, activation='relu', name = "Hidden_Layer3"),
              
              tf.keras.layers.Dropout(0.1, name = "Dropout"),
              tf.keras.layers.Dense(self.class_number, activation="softmax", name = "Output_Layer")
        ])
        #print(self.model.summary())
        
        self.model.compile(
             optimizer="adam",
             # 預設都是false
             loss="SparseCategoricalCrossentropy",
             metrics=["accuracy"],
        )
    def predict(self, test_X):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./model/{dataset}/{dataset}-{method}_ModelCheckpoint.h5", monitor='accuracy', verbose=1, save_best_only=True)
        self.train_history = self.model.fit(
                            batch_size = 64,
                            x = self.train_X,
                            y = self.train_y,
                            epochs=50,
                            validation_split=0.1,
                            callbacks=[checkpoint]
                        )

        predict_test_y_prob = self.model.predict(test_X)
        predict_test_y = np.argmax(predict_test_y_prob, axis = 1)

        return predict_test_y
    def evaluate(self):
        return self.model.evaluate(self.train_X, self.train_y)
    # 畫 digit vs residual
    def draw_plot(self):
        
        # Accuracy
        plt.plot(self.train_history.history['accuracy'])
        plt.plot(self.train_history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid()
        plt.show()
        
        # Loss
        plt.plot(self.train_history.history['loss'])
        plt.plot(self.train_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid()
        plt.show()
        #　Draw model plot
        plot_model(self.model, to_file='ANN_Model.png', show_shapes=True, show_layer_activations=True)
        plt.imshow(plt.imread('ANN_Model.png'))
        plt.show()
    # Store model
    def model_output(self):
        return self.model
        
# CNN method
class CNNMethod:
    # Constructor
    def __init__(self):
        pass
    # Method
    def fit(self, train_X, train_y):
        self.IMAGE_SIZE = int(np.sqrt(train_X.shape[1]))
        self.class_number = len(np.unique(train_y))
        self.train_X = np.expand_dims(train_X.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE), -1)
        self.train_y = train_y
        
        self.model = tf.keras.models.Sequential([
               tf.keras.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)),
               tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name = "Conv2D_Layer1"),
               tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = "MaxPooling2D_Layer1"),
               
               tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name = "Conv2D_Layer2"),
               tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = "MaxPooling2D_Layer2"),
               
               tf.keras.layers.Flatten(name = "Flatten_Layer"),
               tf.keras.layers.Dropout(0.1, name = "Dropout1"),
               
               tf.keras.layers.Dense(128, activation="relu", name = "Hidden_Layer1"),
               tf.keras.layers.Dense(64, activation="relu", name = "Hidden_Layer2"),
               tf.keras.layers.Dropout(0.1, name = "Dropout2"),
               
               tf.keras.layers.Dense(self.class_number, activation="softmax", name = "Output_Layer"),
         ])
        
        
        self.model.compile(
             optimizer="adam",
             # 預設都是false
             loss = "SparseCategoricalCrossentropy",
             metrics = ["accuracy"],
         )
    def predict(self, test_X):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./model/{dataset}/{dataset}-{method}_ModelCheckpoint.h5", monitor='accuracy', verbose=1, save_best_only=True)
        
        
        train_xx, label_x, train_yy, label_y = train_test_split(self.train_X, self.train_y,  test_size=0.2, random_state = 42)
          
        self.train_history = self.model.fit(
                             x = train_xx, 
                             y = train_yy, 
                             batch_size = 128, 
                             epochs=25, 
                             validation_data=(label_x, label_y), 
                             callbacks = [checkpoint]
                         )
        predict_test_y_prob = self.model.predict(np.expand_dims(test_X.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE), -1))
        predict_test_y = np.argmax(predict_test_y_prob, axis = 1)   
        return predict_test_y
    def evaluate(self):
        return self.model.evaluate(self.train_X, self.train_y)
    # 畫 digit vs residual
    def draw_plot(self):
        
        # Accuracy
        plt.plot(self.train_history.history['accuracy'])
        plt.plot(self.train_history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid()
        plt.show()
        
        # Loss
        plt.plot(self.train_history.history['loss'])
        plt.plot(self.train_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid()
        plt.show()
        #　Draw model plot
        plot_model(self.model, to_file='CNN_Model.png', show_shapes=True, show_layer_activations=True)
        plt.imshow(plt.imread('CNN_Model.png'))
        plt.show()
    # Store model
    def model_output(self):
        return self.model
    
def output_prediction(predict_test_y = np.array([]), test_y = np.array([]), test_X = np.array([]), is_show_wrong = False, IMAGE_SIZE = 16):
    classes_num = int(np.unique(test_y).shape[0])
    # Training record
    test_y_category = {}
    for label in np.unique(train_y):
        test_y_category[label] = len(test_y[test_y == label])
    # Test record
    test_predict_dict = {}
    test_TP_dict = {}
    for label in np.unique(test_y):
        test_predict_dict[label] = 0
        test_TP_dict[label] = 0 
   
    # debug
    if predict_test_y.size != 0:
        for test_index in range(test_y.shape[0]): 
            predict_label = predict_test_y[test_index]
            test_predict_dict[predict_label] += 1
            if predict_label == test_y[test_index]:
                test_TP_dict[predict_label] += 1 
        print(f'[{method}]')
        total_TP = 0
        category_acc = []
        for label in np.unique(test_y):
            total_TP += test_TP_dict[label]
            acc = round((test_TP_dict[label] / test_y_category[label]) * 100, 2)
            category_acc.append(acc)
            print(f'{label} accuracy: {acc} % ({test_TP_dict[label]}/{test_y_category[label]})')
        print(f"Total accuracy: {round((total_TP / len(test_y)) * 100, 2)} % ({total_TP}/{len(test_y)})")
        category_acc.append(round((total_TP / len(test_y)) * 100, 2))
        print(category_acc)
        # draw bar plot
        plt.yticks(range(0, classes_num + 1), list(range(0, classes_num)) + ['Total'])
        barhlist = plt.barh(list(range(0, classes_num + 1)), category_acc, color = 'green')
        barhlist[-1].set_color('red')
        for index, value in enumerate(category_acc):
            plt.text(value, index, str(value) + '%')
        plt.ylabel('digit')
        plt.xlabel('accuracy')
        plt.title(f'[{method}] method')
        plt.show()
        
        # draw confusion matrix
        cm = confusion_matrix(test_y, predict_test_y, labels=np.unique(test_y))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_y))
        disp.plot()
        plt.title(f'[{method}] method')
        plt.show()
        
        
        if is_show_wrong == True:
            test_digit = input("Enter test_digit: ")
            if str.isnumeric(test_digit) and (0 <= int(test_digit) <= classes_num - 1):
                number = int(test_digit)
                wrong_index = []
                for num in range(number, number+1):
                    tmp_index = 0
                    label_list = []
                    predict_list = []
                    image = dict()
                    for index in range(test_y.shape[0]):
                        if (test_y[index] == num) and (test_y[index] != predict_test_y[index]):
                            image[tmp_index] = test_X[index].reshape(IMAGE_SIZE, IMAGE_SIZE)  
                            label_list.append(test_y[index])
                            predict_list.append(predict_test_y[index])
                            wrong_index.append(index)
                            tmp_index += 1
                            
                    integrated_plot(image, label_list, predict_list)   
            else:
                print('Error!')
        return test_predict_dict, test_TP_dict
    else:
        print('No predition')
        return None, None
    
def model_choise(dataset = 'usps', method = 'Means', istrain = False, isdraw = False, issavemodel = False):
    # loading dataset
    train_X, train_y, test_X, test_y = dataset_loading(dataset) # usps/ mnist/ mnist_usps
    class_number = int(np.unique(train_y).shape[0])
    if method == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=3)
        # 開始計算
        start = time.time()
        clf.fit(train_X, train_y)
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        isdraw = False
        
    elif method == 'Kmeans':
        clf = KMeans(n_clusters=class_number, random_state=42, n_init="auto")
        clf.fit(train_X)
        # 開始計算
        start = time.time()
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        isdraw = False
    elif method == 'SVM':
        #clf = SVC(gamma='auto')
        clf = SVC(kernel='linear')
        clf.fit(train_X, train_y)
        # 開始計算
        start = time.time()
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        isdraw = False
    elif method == 'RF':
        clf = RandomForestClassifier(random_state=42)
        # 開始計算
        start = time.time()
        clf.fit(train_X, train_y)
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        isdraw = False
    elif method == 'Means':  
        clf = MeansMethod()
        # 開始計算
        start = time.time()
        clf.fit(train_X, train_y)
        if istrain == True:
            predict_test_y = clf.predict(train_X, norm_type = 2) #'fro', 2
        else:
            predict_test_y = clf.predict(test_X, norm_type = 2) # 'fro', 2
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        if isdraw == True:
            clf.draw_digitvs_residual(test_X, test_y)
        issavemodel = False
    elif method == 'SVD':
        decide_K = [12 for i in range(class_number)]
        clf = SVDMethod(decide_K)
        # 開始計算
        start = time.time()
        clf.fit(train_X, train_y)
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        if isdraw == True:
            clf.draw_digitvs_residual(test_X, test_y)
        issavemodel = False
    elif method == 'each_SVD_k':
        clf = SVDMethod()
        clf.fit(train_X, train_y)
        clf.each_digit_svd_k_plot(train_X, train_y, 'Training')
        clf.each_digit_svd_k_plot(test_X, test_y, 'Test')  
        predict_test_y = np.array([])
        issavemodel = False
    elif method == 'HOSVD':
        decide_K = [12 for i in range(class_number)]
        clf = HOSVDMethod(decide_K)
        # 開始計算
        start = time.time()
        clf.fit(train_X, train_y, decomposition_type = 'original') #time: original < myHOSVD
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print("執行時間：%f 秒" % (end - start))
        if isdraw == True:
            clf.draw_digitvs_residual(test_X, test_y)
        issavemodel = False
    elif method == 'each_HOSVD_k':
        clf = HOSVDMethod()
        clf.fit(train_X, train_y, decomposition_type = 'original') #time: original < myHOSVD
        clf.each_digit_hosvd_k_plot(train_X, train_y, 'Training')
        clf.each_digit_hosvd_k_plot(test_X, test_y, 'Test')  
        predict_test_y = np.array([])
        issavemodel = False
    elif method == 'ANN':
        is_using_model = False
        if is_using_model == True:
            clf = tf.keras.models.load_model("./model/ANN_model")
            if istrain == True:
                predict_test_y = clf.predict(train_X)
            else:
                predict_test_y = clf.predict(test_X)
        else:
            clf = ANNMethod()
            # 開始計算
            start = time.time()
            clf.fit(train_X, train_y) 
            if istrain == True:
                predict_test_y = clf.predict(train_X)
            else:
                predict_test_y = clf.predict(test_X)
            # 結束計算
            end = time.time()
            print(f'training: {clf.evaluate()}')
            print("執行時間：%f 秒" % (end - start))
            if isdraw == True:
                clf.draw_plot()
    elif method == 'CNN':
        clf = CNNMethod()
        # 開始計算
        start = time.time()
        clf.fit(train_X, train_y) 
        if istrain == True:
            predict_test_y = clf.predict(train_X)
        else:
            predict_test_y = clf.predict(test_X)
        # 結束計算
        end = time.time()
        print(f'training: {clf.evaluate()}')
        
        print("執行時間：%f 秒" % (end - start))
        if isdraw == True:
            clf.draw_plot()
    else:
        predict_test_y = np.array([])
    
    # store model
    if issavemodel == True:
        if method in ['KNN', 'Kmeans', 'SVM', 'RF', 'LinearSVC']:
            joblib.dump(clf, f'./model/{dataset}/{dataset}-{method}')
        elif method in ['ANN', 'CNN']:
            clf.model_output().save(f'./model/{dataset}/{dataset}-{method}.h5')
    return predict_test_y

# 確認模型儲存正確
def test_store_model(test_data, dataset = 'usps', method = '', isModelCheckpoin = False, istrain = False):    
    if method in ['KNN', 'Kmeans', 'SVM', 'RF', 'LinearSVC']:
        test_model = joblib.load(f'./model/{dataset}/{dataset}-{method}')
        test_predict_test_y = test_model.predict(test_data)
    elif method in ['ANN']:
        if isModelCheckpoin == True:
            test_model = tf.keras.models.load_model(f"./model/{dataset}/{dataset}-{method}_ModelCheckpoint.h5")
        else:
            test_model = tf.keras.models.load_model(f"./model/{dataset}/{dataset}-{method}.h5")
        test_predict_test_y_prob = test_model.predict(test_data)
        test_predict_test_y = np.argmax(test_predict_test_y_prob, axis = 1)
        print(test_model.summary())
    elif method in ['CNN']:
        if isModelCheckpoin == True:
            test_model = tf.keras.models.load_model(f"./model/{dataset}/{dataset}-{method}_ModelCheckpoint.h5")
        else:
            test_model = tf.keras.models.load_model(f"./model/{dataset}/{dataset}-{method}.h5")
        print(test_model.summary())
        IMAGE_SIZE = int(np.sqrt(test_data.shape[1]))
        test_predict_test_y_prob = test_model.predict(np.expand_dims(test_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE), -1))
        test_predict_test_y = np.argmax(test_predict_test_y_prob, axis = 1) 
    
    if method in ['KNN', 'Kmeans', 'SVM', 'RF', 'LinearSVC', 'ANN', 'CNN']:
        if istrain == True:
            test_predict_dict, test_TP_dict = output_prediction(test_predict_test_y, train_y, train_X, is_show_wrong = True, IMAGE_SIZE = int(np.sqrt(train_X.shape[1])))
        else:
            test_predict_dict, test_TP_dict = output_prediction(test_predict_test_y, test_y, test_X, is_show_wrong = True, IMAGE_SIZE = int(np.sqrt(train_X.shape[1])))
            

if __name__ == '__main__':
    test_model = True
    istrain = False
    dataset = 'usps' # usps/ sign_mnist
    method = 'SVD' # Means/ SVD/ HOSVD/ KNN/ Kmeans/ SVM/ RF/ ANN/ CNN/ each_SVD_k/ each_HOSVD_k
    train_X, train_y, test_X, test_y = dataset_loading(dataset, isdraw = False)

    
    predict_test_y = model_choise(dataset, method, istrain = istrain, isdraw = True, issavemodel = False)
    if istrain == True:
        test_predict_dict, test_TP_dict = output_prediction(predict_test_y, train_y, train_X, is_show_wrong = True, IMAGE_SIZE = int(np.sqrt(train_X.shape[1])))
    else:
        test_predict_dict, test_TP_dict = output_prediction(predict_test_y, test_y, test_X, is_show_wrong = True, IMAGE_SIZE = int(np.sqrt(train_X.shape[1])))
    
    
    istrain = True
    if test_model == True:
        print('^' * 50)
        if istrain == True:
            test_store_model(train_X, method = method, dataset = dataset, isModelCheckpoin = False, istrain = istrain)
        else:
            test_store_model(test_X, method = method, dataset = dataset, isModelCheckpoin = False, istrain = istrain)