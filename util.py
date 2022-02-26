import torchvision
import torch.nn.functional as F
import numpy as np
import torch
def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3} ({:.3f})'.format(self.val, self.avg)

def adapt_xy(x,y=None):
        _, pad_h = divmod(x[0].shape[0], 64) ##256/2^6  # 436/64=6余52  pad_h=52
        if pad_h != 0:
            pad_h = 64 - pad_h  #64-52=12
        _, pad_w = divmod(x[0].shape[1], 64)
        if pad_w != 0:
            pad_w = 64 - pad_w
        x_adapt_info = None
        if pad_h != 0 or pad_w != 0:
            padding = [(0, pad_h), (0, pad_w), (0, 0)]

            x0=x[0]
            x1=x[1]

            x_adapt0 = np.pad(x0, padding, mode='constant', constant_values=0.)
            x_adapt1 = np.pad(x1, padding, mode='constant', constant_values=0.)
            
        if y is not None:
            #y_adapt = np.pad(y, padding, mode='constant', constant_values=0.)
            return [x_adapt0,x_adapt1],y
        else:
            return [x_adapt0,x_adapt1]

def realEPE(output, target, sparse=False):

    return EPE(output, target, sparse, mean=True)

def flow_kitti_error(tu, tv, u, v, mask):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param mask: ground-truth mask
    :return: End point error of the estimated flow
    """
    tau = [3, 0.05]
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    smask = mask[:]

    ind_valid = (smask != 0)
    n_total = np.sum(ind_valid)

    epe = np.sqrt((stu - su)**2 + (stv - sv)**2)
    mag = np.sqrt(stu**2 + stv**2) + 1e-5

    epe = epe[ind_valid]
    mag = mag[ind_valid]

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    mean_epe = np.mean(epe)
    mean_acc = (float(n_err) / float(n_total))
    return (mean_epe, mean_acc)


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        if target is not None:
            for t in self.co_transforms:
                input,target = t(input,target)
            return input,target
        else:
            for t in self.co_transforms:
                input = t(input,target)
            return input

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()
