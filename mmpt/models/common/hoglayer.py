import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import cv2
import numpy as np


def timeit(x, func, iter=10):
	torch.cuda.synchronize()
	start = time.time()
	for _ in range(iter):
		y = func(x)
	torch.cuda.synchronize()
	runtime = (time.time()-start)/iter
	return runtime

class HOGLayer(nn.Module):
    def __init__(self, nbins=10, kernel=8, max_angle=math.pi, stride=1, padding=1, dilation=1, pool_padding=1, pool_stride=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.pool = pool
        self.max_angle = max_angle
        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(kernel, stride=pool_stride, padding=pool_padding)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation, 1)
            #2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:,None,:,:]
            phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

            #3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:,None,:,:]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)

            b = phase_int.floor() %self.nbins
            t = phase_int.ceil() %self.nbins
            f = phase_int % self.nbins
            t_v = norm * (1-(t-f))
            b_v = norm * (1-(f-b))
            
            out.scatter_(1, phase_int.floor().long()%self.nbins, b_v) # fix 
            out.scatter_add_(1, phase_int.ceil().long()%self.nbins, t_v) # fix

            return self.pooler(out)
        
        
class HOGLayerMoreComplicated(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=math.pi, stride=1, padding=1, dilation=1,
                 mean_in=False, max_out=False):
        super(HOGLayerMoreComplicated, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        self.max_out = max_out
        self.mean_in = mean_in

        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0).double()
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        if self.mean_in:
            return self.forward_v1(x)
        else:
            return self.forward_v2(x)

    def forward_v1(self, x):
        if x.size(1) > 1:
            x = x.mean(dim=1)[:,None,:,:]

        gxy = F.conv2d(x, self.weight, None, self.stride,
                       self.padding, self.dilation, 1)
        # 2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:, None, :, :]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
        out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)

        return self.pooler(out)

    def forward_v2(self, x):
        batch_size, in_channels, height, width = x.shape
        weight = self.weight.repeat(3, 1, 1, 1)
        gxy = F.conv2d(x, weight, None, self.stride,
                        self.padding, self.dilation, groups=in_channels)

        gxy = gxy.view(batch_size, in_channels, 2, height, width)

        if self.max_out:
            gxy = gxy.max(dim=1)[0][:,None,:,:,:]

        #2. Mag/ Phase
        mags = gxy.norm(dim=2)
        norms = mags[:,:,None,:,:]
        phases = torch.atan2(gxy[:,:,0,:,:], gxy[:,:,1,:,:])

        #3. Binning Mag with linear interpolation
        phases_int = phases / self.max_angle * self.nbins
        phases_int = phases_int[:,:,None,:,:]

        out = torch.zeros((batch_size, in_channels, self.nbins, height, width),
                          dtype=torch.double, device=gxy.device)
        out.scatter_(2, phases_int.floor().long()%self.nbins, norms)
        out.scatter_add_(2, phases_int.ceil().long()%self.nbins, 1 - norms)

        out = out.view(batch_size, in_channels * self.nbins, height, width)
        out = torch.cat((self.pooler(out), self.pooler(x)), dim=1)

        return out



if __name__ == '__main__':
    # layer = HOGLayerMoreComplicated()
    layer = HOGLayer(kernel=12, padding=2)
    image = cv2.imread('/home/lr/dataset/YouTube-VOS/2018/train/JPEGImages_s256/e4ef0a3a34/00000.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256,256)) / 255
    # x = torch.from_numpy(image).permute(2,0,1)[None]
    x = torch.from_numpy(image)[None,None]
    o = layer(x)
    
    y2 = o.cpu().numpy()
    bin = 0

    im = y2[0, bin]
    im = (im-im.min())/(im.max()-im.min()) * 255
    print('bin: ', bin, ' ang: ', float(bin)/layer.nbins * 180.0)
    # cv2.imshow('bin', im)
    # cv2.waitKey()
    # bin = (bin + 1) % layer.nbins
    cv2.imwrite('o.jpg',im)
    print('haha')