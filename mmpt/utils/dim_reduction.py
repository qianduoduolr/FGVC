import torch
import numpy as np
from sklearn.decomposition import PCA



def PCA_numpy(X, embed_dim):
    bsz, c, h, w = X.shape
    X = X.permute(0, 2, 3, 1).flatten(0,2).detach().cpu().numpy()
    X_new = X - np.mean(X, axis=0)
    # SVD
    U, Sigma, Vh = np.linalg.svd(X_new, full_matrices=False, compute_uv=True)
    X_pca_svd = np.dot(X_new, (Vh.T)[:,:embed_dim])
    X_out = torch.from_numpy(X_pca_svd).cuda()
    out = X_out.reshape(bsz, h, w, embed_dim)

    return out.permute(0, 3, 1, 2)


def PCA_torch_v1(X, embed_dim):
    bsz, c, h, w = X.shape
    X = X.permute(0, 2, 3, 1).flatten(1,2)
    U, S, V = torch.pca_lowrank(X, q=embed_dim, center=True, niter=2)
    X = torch.matmul(X, V)
    
    return X.permute(0, 2, 1).reshape(bsz, embed_dim, h, w)


def PCA_torch_v2(X, embed_dim):
    bsz, c, h, w = X.shape
    X = X.permute(0, 2, 3, 1).flatten(1,2)
    X = X - X.mean(dim=1, keepdim=True)
    U, S, V = torch.svd(X)
    X = torch.matmul(X, V[:,:,:embed_dim])
    
    return X.permute(0, 2, 1).reshape(bsz, embed_dim, h, w)

def pca_feats(ff, K=1, solver='auto', whiten=True, img_normalize=True):
    ## expect ff to be   N x C x H x W

    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3*K,
        svd_solver=solver,
        whiten=whiten
    )

    ff = ff.transpose(1, 2).transpose(2, 3)
    ff = ff.reshape(N*H*W, C).numpy()
    
    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3*K)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    pca_ff = [pca_ff[:, kk:kk+3] for kk in range(0, pca_ff.shape[1], 3)]

    if img_normalize:
        pca_ff = [(x - x.min()) / (x.max() - x.min()) for x in pca_ff]

    return pca_ff[0] if K == 1 else pca_ff