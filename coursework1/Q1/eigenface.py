import matplotlib.pyplot as plt
import matplotlib.image as pmimage

import numpy as np
import scipy.io as io
import random

def pca(normed_face):
    S = normed_face @ normed_face.transpose() / np.shape(normed_face)[1] # (1/N)AA.transpose()
    
    return np.linalg.eig(S)

def low_pca(normed_face):
    S = normed_face.transpose() @ normed_face / np.shape(normed_face)[1]  #(1/N)A.transpose()A
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



mean_face = np.mean(train_X, axis = 1)
A = train_X - mean_face[:, None]
eigen_values_1, eigen_vectors_1 = pca(A)
eigen_values_2, eigen_vectors_2 = low_pca(A)


#print(np.shape(u1), np.shape(u2))

# [Plot] Mean face reconstruction
'''
reconstructed_mean_face = mean_face.reshape((-1, 56)).transpose()
plt.imshow(reconstructed_mean_face, cmap='gray')
plt.title('Mean Face')
'''

# sort the eigenvalues(descent order)
idx = eigen_values_1.argsort()[::-1]
eigen_values_1 = eigen_values_1[idx]
eigen_vectors_1 = eigen_vectors_1[:, idx]

idx = eigen_values_2.argsort()[::-1]
eigen_values_2 = eigen_values_2[idx]
eigen_vectors_2 = eigen_vectors_2[:, idx]


# [Plot] Eigenvalues
'''
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(eigen_values_1)
ax2.plot(eigen_values_2)
print("original nonzero eig_val is {}\n reduced eig_val is {}".format(
    np.count_nonzero(eigen_values_1 > 1), np.count_nonzero(eigen_values_2 > 1)
))
plt.show()
'''

n = 30 # 20
sample_face = test_X[:, 0]
a = (sample_face - mean_face) @ eigen_vectors_2[:, :n]
recon_with_n = mean_face + np.sum(eigen_vectors_2[:, : n] * a, axis = 1)
recon_fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(sample_face.reshape((-1, 56)).transpose(), cmap='gray')
ax2.imshow(recon_with_n.reshape((-1, 56)).transpose(), cmap='gray')
ax1.set_title('original face')
ax2.set_title('reconstructed with {} bases'.format(n));

plt.show()


