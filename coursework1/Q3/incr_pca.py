import matplotlib.pyplot as plt
import matplotlib.image as pmimage

import numpy as np
import scipy.io as io
import random

def get_covariance(matrix):
    S = matrix.transpose() @ matrix / np.shape(matrix)[1] # covariance matrix
    return S


def covariance_by_PCA_decomp(eig_val, eig_vec, m):
    # Get covariance Matrix by using eigenvalues and eigenvectors
    # S \simeq = P\LambdaP^T (where P is the matrix which columns are eigenvectors,
    # \Lambda is diagonal matrix which entries are eigenvalues)
    P = eig_vec[:, :m]
    #print(np.shape(P))
    D = np.identity(m)*eig_val[:m]
    #print(np.shape(D))
    S = P @ D @ P.transpose()
    
    return S


def low_pca(normed_face):
    S = get_covariance(normed_face)
    #normed_face.transpose() @ normed_face / np.shape(normed_face)[1]  #(1/N)A.transpose()A
    val, vec = np.linalg.eig(S)
    vec = normed_face @ vec
    vec = vec / np.sqrt(np.sum(np.power(vec, 2), axis=0))
    return val, vec


data_path = '../material/face.mat'
face_mat = io.loadmat(data_path)

X = np.array(face_mat['X'])
L = np.array(face_mat['l'])

total_num = np.shape(X)[1]
train_idx = np.array(range(total_num)) % 10 < 8
test_idx = np.array(range(total_num)) % 10 > 8 

random.shuffle(train_idx)

train_num = total_num * 8 // 10
test_num = total_num - train_num


train_X = X[:, train_idx]
train_L = L[:, train_idx]

test_X = X[:, test_idx]
test_L = L[:, test_idx]


# let's define four subset of training data set
# training data set will be equally divided into four subsets
train_X1 = train_X[:, 0::4]
train_X2 = train_X[:, 1::4]
train_X3 = train_X[:, 2::4]
train_X4 = train_X[:, 3::4]

train_L1 = train_L[:, 0::4]
train_L2 = train_L[:, 1::4]
train_L3 = train_L[:, 2::4]
train_L4 = train_L[:, 3::4]

# The cardinality of each subset is 104
#print(np.shape(train_L1),np.shape(train_L2),np.shape(train_L3),np.shape(train_L4))


# Step 1: compute with subset 1
mean_face1 = np.mean(train_X1, axis = 1) # mean face of subset1
A1 = train_X1 - mean_face1[:, None] # normalized face
eig_val_1, eig_vec_1 = low_pca(A1) # eigenvalues and eigenvectors
N1 = np.shape(train_L1)[1] # cardinality
m1 = 20 # number of eigen something will be considered

S1 = covariance_by_PCA_decomp(eig_val_1, eig_vec_1, m1) # with m1 components


# Step 2: compute with subset 2 (which will be added to 1, and finally, forms new set S_3)
mean_face2 = np.mean(train_X2, axis = 1) # mean face of subset1
A2 = train_X2 - mean_face2[:, None] # normalized face
eig_val_2, eig_vec_2 = low_pca(A2) # eigenvalues and eigenvectors
N2 = np.shape(train_L2)[1] # cardinality
m2 = 20 # number of eigen something will be considered

S2 = covariance_by_PCA_decomp(eig_val_2, eig_vec_2, m2) # with m2 components


# Step 3: combine subset 1 and 2, generate new set N_out1. let's find combined values
N_out1 = N1 + N2
mean_face_out1 = (mean_face1*N1 + mean_face2*N2)/(N_out1) # combined mean_face
S_out1 = (N1/N_out1)*S1 + (N2/N_out1)*S2 + (N1*N2/N_out1/N_out1) * (mean_face1 - mean_face2) @ (mean_face1 - mean_face2).transpose() # Combined covariance matrix
print(np.shape(S_out1))


# Step 4: compute with subset 3
mean_face3 = np.mean(train_X3, axis = 1) # mean face of subset1
A3 = train_X3 - mean_face3[:, None] # normalized face
eig_val_3, eig_vec_3 = low_pca(A3) # eigenvalues and eigenvectors
N3 = np.shape(train_L3)[1] # cardinality
m3 = 20 # number of eigen something will be considered

S3 = covariance_by_PCA_decomp(eig_val_3, eig_vec_3, m3) # with m3 components

# Step 5: combine subset 3 and (S_out1) -> S_out2
N_out2 = N3 + N_out1
mean_face_out2 = (mean_face3*N3 + mean_face_out1*N_out1)/(N_out2) # combined mean_face
S_out2 = (N3/N_out2)*S1 + (N_out1/N_out2)*S2 + (N3*N_out1/N_out2/N_out2) * (mean_face3 - mean_face_out1) @ (mean_face3 - mean_face_out1).transpose() # Combined covariance matrix
print(np.shape(S_out2))

# Step 6: compute with subset 4
mean_face4 = np.mean(train_X4, axis = 1) # mean face of subset1
A4 = train_X4 - mean_face4[:, None] # normalized face
eig_val_4, eig_vec_4 = low_pca(A4) # eigenvalues and eigenvectors
N4 = np.shape(train_L4)[1] # cardinality
m4 = 20 # number of eigen something will be considered

# Step 7: combine subset 4 and (S_out2) -> S_out3
N_out3 = N4 + N_out2
mean_face_out3 = (mean_face4*N4 + mean_face_out2*N_out2)/(N_out3) # combined mean_face
S_out3 = (N4/N_out3)*S1 + (N_out2/N_out3)*S2 + (N4*N_out2/N_out3/N_out3) * (mean_face4 - mean_face_out2) @ (mean_face4 - mean_face_out2).transpose() # Combined covariance matrix
print(np.shape(S_out3))


# Step 8: Finally, we got the covariance matrix S = S_out3.




'''
mean_face = np.mean(train_X, axis = 1)
A = train_X - mean_face[:, None]
eigen_values_1, eigen_vectors_1 = pca(A)
eigen_values_2, eigen_vectors_2 = low_pca(A)
'''


# [Plot] Mean face reconstruction
'''
reconstructed_mean_face = mean_face.reshape((-1, 56)).transpose()
plt.imshow(reconstructed_mean_face, cmap='gray')
plt.title('Mean Face')
'''