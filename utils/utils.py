import torch
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
import torchvision.utils as vutils
import torch.distributed as dist
from typing import Dict, List, Tuple
import cv2

from PIL import Image
import errno
import os
import re
import sys
import numpy as np
from bisect import bisect_right

import torch.nn.functional as F
eps = 1e-7


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def dict2cuda(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2cuda(v)
        elif isinstance(v, torch.Tensor):
            v = v.cuda()
        new_dic[k] = v
    return new_dic

def dict2numpy(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2numpy(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().copy()
        new_dic[k] = v
    return new_dic

def dict2float(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2float(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        new_dic[k] = v
    return new_dic

def metric_with_thresh(depth, label, mask, thresh):
    err = torch.abs(depth - label)
    valid = err <= thresh
    mean_abs = torch.mean(err[valid])
    acc = valid.sum(dtype=torch.float) / mask.sum(dtype=torch.float)
    return mean_abs, acc

def evaluate(depth, mask, label, thresh):
    batch_abs_err = []
    batch_acc = []
    for d, m, l in zip(depth, mask, label):
        abs_err, acc = metric_with_thresh(d, l, m, thresh)
        batch_abs_err.append(abs_err)
        batch_acc.append(acc)

    tot_abs = torch.stack(batch_abs_err)
    tot_acc = torch.stack(batch_acc)
    return tot_abs.mean(), tot_acc.mean()

def save_cameras(cam, path):
    cam_txt = open(path, 'w+')

    cam_txt.write('extrinsic\n')
    for i in range(4):
        for j in range(4):
            cam_txt.write(str(cam[0, i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.write('\n')

    cam_txt.write('intrinsic\n')
    for i in range(3):
        for j in range(3):
            cam_txt.write(str(cam[1, i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.close()

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_schedule_with_warmup(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1,):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            alpha = float(current_step) / warmup_iters
            current_factor = warmup_factor * (1. - alpha) + alpha
        else:
            current_factor = 1.

        return max(0.0,  current_factor * (gamma ** bisect_right(milestones, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def add_summary(data_dict: dict, dtype: str, logger, index: int, flag: str):
    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        if img.dtype == np.bool:
            img = img.astype(np.float32)
        img = torch.from_numpy(img[:1])
        if 'depth' in name or 'label' in name:
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, range=(450, 850))
        elif 'mask' in name:
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, range=(0, 1))
        elif 'error' in name:
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, range=(0, 4))
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,)

    on_main = (not is_distributed) or (dist.get_rank() == 0)
    if not on_main:
        return

    if dtype == 'image':
        for k, v in data_dict.items():
            logger.add_image('{}/{}'.format(flag, k), preprocess(k, v), index)

    elif dtype == 'scalar':
        for k, v in data_dict.items():
            logger.add_scalar('{}/{}'.format(flag, k), v, index)
    else:
        raise NotImplementedError

class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input: dict):
        self.count += 1
        for k, v in new_input.items():
            assert isinstance(v, float), type(v)
            self.data[k] = self.data.get(k, 0) + v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

def reduce_tensors(datas: dict):
    if not is_distributed:
        return datas
    world_size = dist.get_world_size()
    with torch.no_grad():
        keys = list(datas.keys())
        vals = []
        for k in keys:
            vals.append(datas[k])
        vals = torch.stack(vals, dim=0)
        dist.reduce(vals, op=dist.reduce_op.SUM, dst=0)
        if dist.get_rank() == 0:
            vals /= float(world_size)
        reduced_datas = {k: v for k, v in zip(keys, vals)}
    return reduced_datas

def read_pair_file(filename: str) -> List[Tuple[int, List[int]]]:
    """Read image pairs from text file and output a list of tuples each containing the reference image ID and a list of
    source image IDs

    Args:
        filename: pair text file path string

    Returns:
        List of tuples with reference ID and list of source IDs
    """
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data

def read_cam_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read camera intrinsics, extrinsics, and depth values (min, max) from text file

    Args:
        filename: cam text file path string

    Returns:
        Tuple with intrinsics matrix (3x3), extrinsics matrix (4x4), and depth params vector (min and max) if exists
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # depth min and max: line 11
    if len(lines) >= 12:
        depth_params = np.fromstring(lines[11], dtype=np.float32, sep=' ')
    else:
        depth_params = np.empty(0)

    #print('read_cam_file: ', intrinsics.shape,extrinsics.shape,depth_params.shape)

    return intrinsics, extrinsics, depth_params

def read_image(filename, img_wh):

    image = Image.open(filename)
    # scale 0~255 to 0~1
    np_image = np.array(image, dtype=np.float32) / 255.0
    original_height = np_image.shape[0]
    original_width = np_image.shape[1]
    #print('height,width,original_height,original_width',height,width,original_height,original_width)
    image = cv2.resize(np_image, img_wh, interpolation=cv2.INTER_LINEAR)

    return image, original_height, original_width

def read_map(path: str, max_dim: int = -1) -> np.ndarray:
    """ Read a binary depth map from either PFM or Colmap (bin) format determined by the file extension and also scale
    the map to the max dim if given

    Args:
        path: input depth map file path string
        max_dim: max dimension to scale down the map; keep original size if -1

    Returns:
        Array of depth map values
    """
    if path.endswith('.bin'):
        in_map = read_bin(path)
    elif path.endswith('.pfm'):
        in_map, _ = read_pfm(path)
        #print('in_map',in_map.shape)
    else:
        raise Exception('Invalid input format; only pfm and bin are supported')
    return in_map

def read_bin(path: str) -> np.ndarray:
    """Read a depth map from a Colmap .bin file

    Args:
        path: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
    """
    with open(path, 'rb') as fid:
        width, height, channels = np.genfromtxt(fid, delimiter='&', max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b'&':
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        data = np.fromfile(fid, np.float32)
    data = data.reshape((width, height, channels), order='F')
    data = np.transpose(data, (1, 0, 2))
    return data

def save_image(filename: str, image: np.ndarray) -> None:
    """Save images including binary mask (bool), float (0<= val <= 1), or int (as-is)

    Args:
        filename: image output file path string
        image: output image array
    """
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype == np.float32 or image.dtype == np.float64:
        image = image * 255
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    Image.fromarray(image).save(filename)


def update_depth(pred_depth,exp_var, next_num_d_tree, scale_factor=1.0, mode='bilinear'):
    with torch.no_grad():
        if scale_factor != 1.0:
            pred_depth = F.interpolate(pred_depth, scale_factor=scale_factor, mode=mode)
            exp_var = F.interpolate(exp_var, scale_factor=scale_factor, mode=mode)

        low_bound = -torch.min(pred_depth, exp_var)
        high_bound = exp_var
        ndepth =next_num_d_tree
        # if next_depth_stage<3:
        #     ndepth = 2**(6 - next_depth_stage)
        # else:
        #     ndepth = 2 ** (6 - next_depth_stage-1)
        # assert exp_var.min() >= 0, exp_var.min()
        #print('debug in depthmap2tree2 next_depth_stage ndepth', next_depth_stage, ndepth)
        assert ndepth > 1
        #print('debug in depthmap2tree2 exp_var.min()',  exp_var.cpu().detach().numpy().min(), exp_var.cpu().detach().numpy().max())
        #print('debug in depthmap2tree2 exp_var',exp_var.min(),exp_var.max())
        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []

        for i in range(int(ndepth)):
            new_samps.append(pred_depth + low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)
        depth_range=torch.cat([new_samps[0], new_samps[ndepth-1]],dim=1)
        #depth_start= new_samps[0]
        #depth_end=new_samps[ndepth-1]
        #print('debug in depthmap2tree2 depth_range_samples, depth_range ',depth_range_samples.shape, depth_range.shape)
        # torch.Size([2, 4, 64, 80]) torch.Size([2, 2, 64, 80])
        return depth_range_samples.detach(),depth_range.detach()
