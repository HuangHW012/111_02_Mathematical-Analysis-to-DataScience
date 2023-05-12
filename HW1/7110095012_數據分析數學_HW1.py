# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

# image quality index
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import hausdorff_pair
# skimage image
import skimage
# matrix to image
from PIL import Image
import os

def svd_compression(img, K):
    # store outcome (儲存圖像重構結果)
    outcome = np.zeros_like(img)
    # SVD (利用numpy.linalg.svd對原始圖像做SVD)
    U, Sigma, VT = np.linalg.svd(img)
    # reconstruct an image with top K vectors
    outcome = U[:,:K].dot(np.diag(Sigma[:K])).dot(VT[:K,:])
    # ポカヨケ(Fool-proofing): range of img is [0, 255] & non-negative integer
    outcome[outcome < 0] = 0
    outcome[outcome > 255] = 255
    outcome = outcome.astype('uint8')
    
    return outcome


def cr(original, K):
    original_size = original.shape[0] * original.shape[1]
    sampling_size = (original.shape[0] + original.shape[1] + 1) * (K + 1)
    
    #print(f'K: {K+1}', original_size / sampling_size)
    return original_size / sampling_size

def mse(imageA, imageB):
	# 計算兩張圖片的MSE相似度
	# 注意：兩張圖片必須具有相同的維度，因爲是基於圖像中的對應像素操作的
    # 對應像素相減並將結果累加起來
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	# 進行誤差歸一化
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# 返回結果，該值越小越好，越小說明兩張圖像越相似
	return err

def integrated_plot(original, reconstruction):
    if original.shape[-1] == 1:
        plt.rcParams.update({'font.size': 15})
        k_i = 0
        for pages in range(int(len(reconstruction.keys()) / 9)):
            fig, ax = plt.subplots(3, 3)
            fig.set_size_inches(20, 18)
            fig.subplots_adjust(hspace=0.2, wspace=0)
            # 差異
            fig_diff, ax_diff = plt.subplots(3, 3)
            fig_diff.set_size_inches(20, 18)
            fig_diff.subplots_adjust(hspace=0.2, wspace=0)
            temp = np.empty_like(original)
            
            for i in range(3):
                for j in range(3):
                    temp[:, :, 0] = reconstruction[list(reconstruction.keys())[k_i]][:, :, 0]
                    ax[i, j].imshow(temp, cmap = 'gray')
                    ax[i, j].set_title(f'K: {list(reconstruction.keys())[k_i] + 1}', fontsize = 35)
                    # 差異
                    ax_diff[i, j].imshow(np.abs(original - temp), cmap = 'gray')
                    ax_diff[i, j].set_title(f'K: {list(reconstruction.keys())[k_i] + 1} (abs)', fontsize = 35)
                        
                    k_i = k_i + 1  
    else:
        plt.rcParams.update({'font.size': 15})
        
        for channel in range(original.shape[-1]):
            k_i = 0
            for pages in range(int(len(reconstruction.keys()) / 9)):
                # for one channel
                fig, ax = plt.subplots(3, 3)
                fig.set_size_inches(30, 18)
                fig.subplots_adjust(hspace=0.2, wspace=0)
                temp = np.empty_like(original)
                for i in range(3):
                    for j in range(3):
                        temp[:, :, channel] = reconstruction[list(reconstruction.keys())[k_i]][:, :, channel]
                        ax[i, j].imshow(temp)
                        ax[i, j].set_title(f'K: {list(reconstruction.keys())[k_i] + 1}', fontsize = 35)
                        k_i = k_i + 1  
        #
        k_i = 0
        for pages in range(int(len(reconstruction.keys()) / 9)):
            # for RGB
            fig_all, ax_all = plt.subplots(3, 3)
            fig_all.set_size_inches(30, 18)
            fig_all.subplots_adjust(hspace=0.2, wspace=0)
            # 
            for i in range(3):
                for j in range(3):
                    ax_all[i, j].imshow(reconstruction[list(reconstruction.keys())[k_i]])
                    ax_all[i, j].set_title(f'K: {list(reconstruction.keys())[k_i] + 1}', fontsize = 35)

                    k_i = k_i + 1  
                        

if __name__ == '__main__':
    output_flag = False
    original = plt.imread('./RGB.jpg')
    
    #original = skimage.data.camera()
    #original = plt.imread('./camera.jpg')
    if len(original.shape) != 3:
        original = original[:, :, np.newaxis]
        K_range = [1, 5, 10, 20, 30, 40, 50, 60, 70, 
                   80, 90, 100, 150, 200, 300, 400, 500, 512]
    else:
        K_range = [1, 5, 10, 20, 25, 50, 75, 100, 200, 
                   300, 400, 500, 600, 700, 800, 900, 1000, 1080]
    K_range = [x - 1 for x in K_range]
    
    
    comparsion = np.empty_like(original)
    reconstruction = dict()
    #K_range = range(200 - 10*9*2, 200, 10)
    #K_range = range(np.linalg.matrix_rank(original[:, :, 0]) - 1*9*2, np.linalg.matrix_rank(original[:, :, 0]), 1)
    for i in K_range:
        reconstruction[i] = np.empty_like(original)
    for channel in range(original.shape[-1]): #original.shape[-1]
        image = original[:, :, channel]
        comparsion[:, :, channel] = image
        
        
        U, Sigma, V_T = np.linalg.svd(image)
        sigma_k_list = []
        
        
        k_list = []
        rank_imagek_list = []
        norm2_list = []
        Sigma_k_plus1_list = []
        normF_list = []
        RSS_Sigma_list = []
        
        mse_list = []
        psnr_list = []
        ssim_list = []
        cr_list = []
        
        for k in K_range:
            temp = np.empty_like(original)
            # 避免K值超出範圍
            if k > min(original.shape[0], original.shape[1]):
                break
            else:
                k_list.append(k)
            image_k = svd_compression(image, k) 
            sigma_k_list.append(Sigma[k])
            
            original_img_array = np.asarray(image, dtype = float)
            compression_img_array = np.asarray(image_k, dtype = float)
            
            #print(f'k = {k+1}')
            #print('max(image_k): ', round(np.max(compression_img_array), 3))
            #print('min(image_k): ', round(np.min(compression_img_array), 3))
            #print('=' * 50)
            
            mse_list.append(mse(original_img_array, compression_img_array))
            psnr_list.append(psnr(original_img_array, compression_img_array, data_range = 255))
            ssim_list.append(ssim(original_img_array, compression_img_array))
            cr_list.append(cr(original_img_array, k))
            
            norm2_list.append(np.linalg.norm(original_img_array - compression_img_array, ord = 2))
            Sigma_k_plus1_list.append(Sigma[k])
            normF_list.append(np.linalg.norm(original_img_array - compression_img_array, ord = 'fro'))
            RSS_Sigma_list.append(np.sqrt(np.sum(Sigma[k:]**2)))
            
            
            reconstruction[k][:, :, channel] = image_k
            #temp[:, :, channel] = image_k
            #if original.shape[-1] == 1:
            #    plt.imshow(temp, cmap = 'gray')
            #else:
            #    plt.imshow(temp)
            #plt.title(f'K: {k}')
            #plt.show()
            
        # mse    
        plt.plot(k_list, mse_list)
        plt.xlabel('K value')
        plt.ylabel('MSE')
        plt.title('Original and compression image (MSE)')
        plt.grid()
        plt.show()
        # psnr
        plt.plot(k_list, psnr_list)
        plt.xlabel('K value')
        plt.ylabel('PSNR')
        plt.title('Original and compression image (PSNR)')
        plt.grid()
        plt.show()
        # ssim
        plt.plot(k_list, ssim_list)
        plt.xlabel('K value')
        plt.ylabel('SSIM')
        plt.title('Original and compression image (SSIM)')
        plt.grid()
        plt.show()
        # cr
        plt.plot(k_list, cr_list)
        plt.xlabel('K value')
        plt.ylabel('CR')
        plt.title('Original and compression image (CR)')
        plt.grid()
        plt.show()

        
        # 觀察sigma 
        
        # https://stackoverflow.com/questions/14762181/adding-a-y-axis-label-to-secondary-y-axis-in-matplotlib
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(list(range(Sigma.shape[0])), Sigma, 'g-')
        ax2.plot(list(range(Sigma.shape[0])), np.log(Sigma), 'b-')
        
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('$\sigma_{k}$', color='g')
        ax2.set_ylabel('$log(\sigma_{k})$', color='b')

        ax1.set_title('Singular value and Rank')
        ax1.grid()
        plt.show()
    


    # draw
    integrated_plot(original, reconstruction)        
    

    
    
    # Remark: 
    if output_flag == True:
        if original.shape[-1] == 1:
            im_ori = Image.fromarray(comparsion[:, :, 0])
            im_ori.save("./Grayscale/comparsion.jpg", compression = 'JPEG', optimize=True)
            print('  原始大小: ', os.path.getsize('./Grayscale/comparsion.jpg'))
            for i in k_list:
                im_k = Image.fromarray(reconstruction[i][:, :, 0])
                im_k.save(f"./Grayscale/SVD_{i}.jpg", compression = 'JPEG', optimize=True)
                print(f'K = {i}大小: ', os.path.getsize(f'./Grayscale/SVD_{i}.jpg'))
        else:
            im_ori = Image.fromarray(comparsion)
            im_ori.save("./RGB/comparsion.jpg", compression = 'JPEG', optimize=True)
            print('  原始大小: ', os.path.getsize('./RGB/comparsion.jpg'))
            for i in k_list:
                im_k = Image.fromarray(reconstruction[i])
                im_k.save(f"./RGB/SVD_{i}.jpg", compression = 'JPEG', optimize=True)
                print(f'K = {i}大小: ', os.path.getsize(f'./RGB/SVD_{i}.jpg'))