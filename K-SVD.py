import numpy as np
import math
from scipy.linalg import svd
import matplotlib.pyplot as plt
from skimage import io, util, restoration
from skimage.restoration import estimate_sigma
from skimage.color import rgb2gray
import os
import cv2
import pandas as pd

######################################################################################
"""### $Functions$"""
######################################################################################

def power_iteration_svd(E):

    A = np.dot(E.T, E)
    n, m = A.shape
    V0 = np.ones(n)
    V0 = V0 / np.linalg.norm(V0)
    epsi = 10**(-4)
    diff=float("inf")
    iters=0
    max = 100
    while((diff)>epsi) and iters<max:

        v_temp = np.dot(A, V0)
        # Compute A^T * A * V
        V1 = v_temp / np.linalg.norm(v_temp)
        # Normalize the vectors
        c=np.linalg.norm(V1)
        if(c>0):
          V1 = V1 / c
        diff= np.linalg.norm(V1-V0)
        V0=V1
        iters+=1

    # Calculate the singular value estimate
    sin = (np.dot(V0,np.dot(A, V0))) / (np.linalg.norm(V0))
    singular_value = (np.sqrt(sin))
    U0 = np.dot(E,V0)/float(singular_value)

    return singular_value, U0, V0

def add_gaussian_noise(image, mean, std_dev):

    noisy_image = util.random_noise(image, mode='gaussian', mean=mean/255, var=(std_dev/255)**2)
    noisy_image = np.uint8(255 * noisy_image)

    return noisy_image

def orthogonal_matching_pursuit(D, alpha, c, n_nonzero_coefs):
  #alpha=Input signal
    n_samples, n_features = D.shape
    # n_patches = alpha.shape[1]

    # Residual and coefficient vectors
    residual = alpha
    coef = np.zeros(n_features)
    idx = []
    count = 0
    while(count<=n_nonzero_coefs):
      # Compute inner products of residual with each column of X
      projections = np.abs(np.dot(D.T, residual))
      # Find the index of the maximum projection
      selected_idx = np.argmax(projections)
      # Update the indices for basis matrix
      idx.append(selected_idx)
      # Solve the least squares problem to update coefficients
      D_hat = D[:, idx] #Basis matrix
      coef[idx] = np.linalg.lstsq(D_hat, alpha, rcond=None)[0]
      # Update the residual
      residual = alpha - np.dot(D_hat, coef[idx])
      count+=1
    if(count==0):
      return coef, count
    return coef, count-1

def OMP(nonzero_coefs, D, X, alpha, noise_gain):

    _ , n_samples = X.shape

    print("OMP initiated...")
    sum=0
    for i in range(n_samples):
        alpha[:, i], count_val = orthogonal_matching_pursuit(D, X[:,i], noise_gain, nonzero_coefs)
        sum+=count_val
        if(i%(int(n_samples/5)) == 0 and i != 0):
            print(((int(i/(int(n_samples/5))))/5)*100, "% completed...")
    return alpha, sum/n_samples

def learn_dictionary(D, X, no_iters, n_nonzero_coefs, noise_gain):

    p , n_samples = X.shape
    n_atoms = D.shape[1] #number of atoms in the Dictionary D
    print("log_2: Dictionary learning initiated..., no of iters=", no_iters)
    for i in range (0, no_iters):

        # Stage 1: Sparse Coding using Orthogonal Matching Pursuit
        alpha = np.zeros((n_atoms, n_samples), float)
        alpha, avg_nos_samples = OMP(n_nonzero_coefs, D, X, alpha, noise_gain)
        print("avg no of coefs=",avg_nos_samples)

        # Stage 2: Dictionary Update
        print("Dictionary Update initiated..., iteration-", i+1)
        for j in range(1, n_atoms):

            if(j%(int(n_atoms/5)) == 0 and j != 0):
                print(((int(j/(int(n_atoms/5))))/5)*100, "% completed...")
            # Find the samples where atom j has a non-zero coefficient
            non_zero_indices = np.where(alpha[j, :] != 0)[0]

            if len(non_zero_indices) > 0:
                # Update atom j of Dictionary using the samples where it is active
                # alpha is the coefficient matrix
                E_j = X[:, non_zero_indices] - np.dot(D, alpha[:, non_zero_indices])

                # code for SVD using power method
                S, U, Vt = power_iteration_svd(E_j)
                D[:, j] = U/np.linalg.norm(U)
                alpha[j, non_zero_indices] = S* Vt

    #return the learned dictionary and alpha of all patches
    return D, alpha, X

def patch_const(noisy_img, patching_size, patch_size):
  if len(noisy_img.shape)==3:
    noisy_img = rgb2gray(noisy_img)

  m,n = noisy_img.shape
  inc_m = (m-patch_size)/(patching_size-1)
  if inc_m<1:
    inc_m=1
  inc_n = (n-patch_size)/(patching_size-1)
  if inc_n<1:
    inc_n=1
  s_p = patch_size

  c = 0
  c_i=0
  i=0
  no_of_samples = patching_size**2
  X = np.zeros((p,no_of_samples))
  patch_count = np.zeros((m,n), dtype=np.uint8)
  while(i<m-patch_size+1):
      c_j=0
      j=0
      while(j<n-patch_size+1):
          # print(i,j)
          X[:,c] = noisy_img[i:i+s_p, j:j+s_p].ravel()
          patch_count[i:i+s_p, j:j+s_p] += 1
          c_j += inc_n
          j = int(c_j)
          c+=1
      c_i += inc_m
      i = int(c_i)
  X = np.delete(X, slice(c,no_of_samples-1), axis=1)
  print("log_1: noisy image to patch created,(patchsize,no. of patches)=", X.shape)
  return X, patch_count

def recontr(lamda, noisy_img, X, patch_count, patching_size, X_hat, patch_size):

  m, n = noisy_img.shape
  rest_img = np.zeros((m,n), float)
  inc_m = (m-patch_size)/(patching_size-1)
  if inc_m<1:
    inc_m=1
  inc_n = (n-patch_size)/(patching_size-1)
  if inc_n<1:
    inc_n=1
  s_p = patch_size

  c = 0
  c_i=0
  i=0
  while(i<m-patch_size+1):
      c_j=0
      j=0
      while(j<n-patch_size+1):
          # rest_img[i:i+s_p, j:j+s_p] += (X_hat[:,c]+lamda*X[:,c]).reshape(s_p, s_p)
          rest_img[i:i+s_p, j:j+s_p] += (X_hat[:,c]).reshape(s_p, s_p)
          c_j += inc_n
          j = int(c_j)
          c+=1
      c_i += inc_m
      i = int(c_i)

  # Averaging
  patch_count[np.where(patch_count == 0)]=1
  # rest_img /= lamda + patch_count
  rest_img /= patch_count
  rest_img += lamda*noisy_img
  rest_img = rest_img/(1+lamda)
  return rest_img

def psnr(noisy_image, denoised):
  mse = np.mean((noisy_image - denoised) ** 2)
  max_pixel = 255.0
  psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
  return psnr, mse

def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def restore_data(standardized_data, mean, std):
    original_data = (standardized_data * std) + mean
    return original_data

def Ksvd_algo(noisy_img, no_iters, n_nonzero_coefs, noise_gain, no_of_samples, D):

  if len(noisy_img.shape)==3:
     noisy_img = rgb2gray(noisy_img)

  p, n_atoms = D.shape

  # Estimate the noise standard deviation using Non-Local Means
  noise = estimate_sigma(noisy_img, multichannel=False)
  # Lagrangian
  lamda = 30/noise
  m,n = noisy_img.shape

  # patch construction
  patching_size= int(math.sqrt(no_of_samples))
  X, patch_count = patch_const(noisy_img, patching_size, patch_size)

  # Normalizating patches
  X, patch_means, patch_std = standardize_data(X)

  # Update dictionary
  D, alpha,X = learn_dictionary(D, X, no_iters, n_nonzero_coefs, noise_gain*noise)

  # patches cnostruction
  X_hat = np.dot(D, alpha)

  # reconstructing the unnormalized data
  X = restore_data(X, patch_means, patch_std)                 # patches
  X_hat = restore_data(X_hat, patch_means, patch_std)         # patch approximation

  # Image reconstruction
  patching_size= int(math.sqrt(no_of_samples))
  rest_img = recontr(lamda, noisy_img, X, patch_count, patching_size, X_hat , patch_size)

  write = "noise=",noise,",n_nonzero_coefs=",n_nonzero_coefs,",n_atoms=",n_atoms,",no_iters=",no_iters,"(patchsize, no. of patches)=", X.shape
  plt.figure(figsize=(15,10))
  plt.subplot(1, 2, 1)
  plt.axis("off")
  plt.title("Noisy Image")
  plt.imshow(noisy_img , cmap="gray")
  plt.subplot(1, 2, 2)
  plt.axis("off")
  plt.title(write)
  plt.imshow(rest_img, cmap="gray" )
  print("done")

  return rest_img

######################################################################################
"""###$Initialization the dictionary$"""
######################################################################################

def DCT_OCB(p):
  M = int(np.sqrt(p))
  total_size = p
  DCT = np.zeros((p, total_size**2))
  inc = (M-1)/total_size
  arrX = (np.arange(inc,M-1+inc, inc))
  alpha_i = np.sqrt(2) / np.sqrt(M)
  arr1=np.arange(0, M)
  c=0
  for i in arrX:
    for j in arrX:
      if i==0:
        X1=np.full(arr1.shape,1)
        alpha_i = 1 / np.sqrt(M)
      else:
        X1 = np.cos(((2*i+1)*np.pi * arr1)/(2*M))
        alpha_i = np.sqrt(2) / np.sqrt(M)
      if j==0:
        X2=np.full(arr1.shape,1)
        alpha_j = 1 / np.sqrt(M)
      else:
        X2 = np.cos(((2*j+1)*np.pi * arr1)/(2*M))
        alpha_j = np.sqrt(2) / np.sqrt(M)
      meshx, meshy = np.meshgrid(X1, X2.T)
      DCT[:,c] = (meshx* meshy*alpha_i*alpha_j).ravel()
      c+=1
  return DCT


# Initialize the dictionary randomly (INITIALIZE ONLY ONE)

# 1.) Uniform distributed columns
def Dic_initialize_Uniform(p, n_atoms):
  D_un = np.random.randint(0, 255, size=(p, n_atoms))
  Dic_mean = np.mean(D_un,axis=0)
  # initiating 0 DC value in dictionary
  D0 = (D_un - Dic_mean)
  # one dictionary atom for constant DC
  D0 [:,0] = 1
  return D0/np.linalg.norm(D0, axis=0)

# 2.)Normally distributed columns
def Dic_initialize_Uniform(p, n_atoms):
  mu, sigma = 0, 1  # mean and standard deviation
  D_un = np.random.normal(mu, sigma, (p, n_atoms))
  # one dictionary atom for constant DC
  D_un [:,0] = 1
  return D_un/np.linalg.norm(D_un, axis=0)

# 3.Columns of DCT overcomplete basis columns
def Dic_DCT_basis(p):
  D_un=DCT_OCB(p)
  return D_un/np.linalg.norm(D_un, axis=0)

######################################################################################
"""###$Execute$"""
######################################################################################

def run_inducing_noise(noise_lvl, image_path):
  D = Dic_initialize_Uniform(p, n_atoms)
  Sample_img = np.array(io.imread(image_path))
  noisy_img = add_gaussian_noise(Sample_img, 0, noise_lvl)
  rest_img = Ksvd_algo(noisy_img, no_iters, n_nonzero_coefs, 
                       noise_gain, no_of_samples, D)

def run_orignal(image_path):
  D = Dic_initialize_Uniform(p, n_atoms)
  noisy_img = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
  rest_img= Ksvd_algo(noisy_img, no_iters, n_nonzero_coefs, 
                      noise_gain, no_of_samples, D)

def run_iterative_orignal(image_path):
  D0 = Dic_initialize_Uniform(p, n_atoms)
  D = D0
  noisy_img = np.array(io.imread(image_path))
  for i in range (5):
    rest_img= Ksvd_algo(noisy_img, no_iters, n_nonzero_coefs, 
                        noise_gain, no_of_samples, D)
    noisy_img = rest_img
    D = D0


######################################################################################
"""###$Global Parameters$"""
######################################################################################

# PARAMETERS
noise_lvl = 35                    # Fill if required
patch_size = 8                    # size of patch (square patch)
no_iters = 10                     # itreations in dictionary update
n_atoms = 256                     # atoms in ditionary
n_nonzero_coefs = 2               # sparcity of each patch
noise_gain = 1.15
p=patch_size**2
no_of_samples = 256**2            # should be changed according to size of inage 

# example code
image_path= r"/content/barbara.tif"
run_inducing_noise(noise_lvl, image_path)