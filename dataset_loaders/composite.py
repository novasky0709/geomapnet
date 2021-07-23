"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Composite data-loaders derived from class specific data loaders
"""
"""
Add annotation at 0723
"""
import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np

import sys
sys.path.insert(0, '../')
from common.pose_utils import calc_vos_simple, calc_vos_safe

class MF(data.Dataset):
  """
  Returns multiple consecutive frames, and optionally VOs
  """
  def __init__(self, dataset, include_vos=False, no_duplicates=False,
               *args, **kwargs):
    """
    :param steps: Number of frames to return on every call
    :param skip: Number of frames to skip
    :param variable_skip: If True, skip = [1, ..., skip]
    :param include_vos: True if the VOs have to be appended to poses. If real
    and include_vos are both on, it gives absolute poses from GT and VOs from
    the SLAM / DSO
    :param no_duplicates: if True, does not duplicate frames when len(self) is
    not a multiple of skip*steps
    """
    self.steps = kwargs.pop('steps', 2)
    self.skip = kwargs.pop('skip', 1)
    self.variable_skip = kwargs.pop('variable_skip', False)
    self.real = kwargs.pop('real', False)
    self.include_vos = include_vos
    self.train = kwargs['train']
    self.vo_func = kwargs.pop('vo_func', calc_vos_simple)
    self.no_duplicates = no_duplicates

    if dataset == '7Scenes':
      from seven_scenes import SevenScenes
      self.dset = SevenScenes(*args, real=self.real, **kwargs)
      """
      We dont use vos and real in MapNet 
      """
      if self.include_vos and self.real:
        self.gt_dset = SevenScenes(*args, skip_images=True, real=False,
          **kwargs)
    elif dataset == 'RobotCar':
      from robotcar import RobotCar
      self.dset = RobotCar(*args, real=self.real, **kwargs)
      if self.include_vos and self.real:
        self.gt_dset = RobotCar(*args, skip_images=True, real=False,
          **kwargs)
    else:
      raise NotImplementedError

    self.L = self.steps * self.skip

  def get_indices(self, index):
   """
   np.random.randint(low,high,size),在[low,high)中随机生成size个随机数
   skip:I1与I2之间跳的步数
   step:一共跳几次（一共取几张图片）
   skips 数组表示每次跳的步数
 
   variable_skip: bool 
   如果variable_skip = False,将生成规则的skips：
   [5,5,5,5,5,5,5,5,...]
   如果variable_skip = True，skip不再是一个常量：
   [1,5,4,2,1,3,3,4,...]
   """
    if self.variable_skip:
      skips = np.random.randint(1, high=self.skip+1, size=self.steps-1)
    else:
      skips = self.skip * np.ones(self.steps-1)
    """
    cumsum():累加函数
    >>>a = np.array([[1,2,3], [4,5,6]])
    >>>np.cumsum(a)
        array([ 1,  3,  6, 10, 15, 21])
        
        
    np.insert(skips, 0, 0)
    在skips数组中第0个位置插入0
    
    插入0后累加，得到offesets数组，如
    [0,2,4,6,8,10,12]
    """
    offsets = np.insert(skips, 0, 0).cumsum()
    """
    把offset回归到0附近
    [-6,-4,-2,0,2,4,6]
    """
    offsets -= offsets[len(offsets) / 2]
    """
    没看懂，大意：不适用连续照片了，把offset从负数补偿回去（可能考虑首帧情况？）
    """
    if self.no_duplicates:
      offsets += self.steps/2 * self.skip
    offsets = offsets.astype(np.int)
    """
    index 是输入，通过idx求得这组连续图像的id，如id = 1500
    [1494,1496,1498,1500,1502,1504,1506]
    """
    idx = index + offsets
    """
    if(id<0):id = 0
    if(id>len-1):id = len-1
    """
    idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
    """
    assert condition
     用来让程序测试这个condition，如果condition为false，那么raise一个AssertionError出来。
    if not condition:
    raise AssertionError()
   
    """
    assert np.all(idx >= 0), '{:d}'.format(index)
    assert np.all(idx < len(self.dset))
    return idx

  def __getitem__(self, index):
    """
    :param index: 
    :return: imgs: STEPS x 3 x H x W   step:一共跳几次（一共取几张图片）
             poses: STEPS x 7
             vos: (STEPS-1) x 7 (only if include_vos = True) [MapNet no this item]
    """
    """
    输入一个index 返回一组idx
    """
    idx = self.get_indices(index)
    """
    clip:
    [(Img1,pos1),(Img2,pos2),(Img3,pos3),....]
    """
    clip  = [self.dset[i] for i in idx]
    """
    torch.stack :拼接，和cat不同，因为多拼出来一个维度
    a = [[1,2,3],[4,5,6]]       dim:(2*3)
    b = [[7,8,9],[10,11,12]]    dim:(2*3)
    torch.stack(a,b) = [a,b] = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]   dim:(2*2*3)
    """
    imgs  = torch.stack([c[0] for c in clip], dim=0)
    poses = torch.stack([c[1] for c in clip], dim=0)
    """
    pass this if
    """
    if self.include_vos:
      # vos = calc_vos_simple(poses.unsqueeze(0))[0] if self.train else \
      #   calc_vos_safe(poses.unsqueeze(0))[0]
      vos = self.vo_func(poses.unsqueeze(0))[0]
      if self.real:  # absolute poses need to come from the GT dataset
        clip = [self.gt_dset[self.dset.gt_idx[i]] for i in idx]
        poses = torch.stack([c[1] for c in clip], dim=0)
      poses = torch.cat((poses, vos), dim=0)
      
    return imgs, poses

  def __len__(self):
    L = len(self.dset)
    if self.no_duplicates:
      L -= (self.steps-1)*self.skip
    return L

class MFOnline(data.Dataset):
  """
  Returns a minibatch of train images with absolute poses and test images
  with real VOs
  """
  def __init__(self, gps_mode=False, *args, **kwargs):
    self.gps_mode = gps_mode
    self.train_set = MF(train=True, *args, **kwargs)
    self.val_set = MF(train=False, include_vos=(not gps_mode), real=True,
                      vo_func=calc_vos_safe, no_duplicates=True, *args,
                      **kwargs)

  def __getitem__(self, idx):
    train_idx = idx % len(self.train_set)
    train_ims, train_poses = self.train_set[train_idx]
    val_idx = idx % len(self.val_set)
    val_ims, val_vos = self.val_set[val_idx]  # val_vos contains abs poses if gps_mode
    if not self.gps_mode:
      val_vos = val_vos[len(val_ims):]
    ims = torch.cat((train_ims, val_ims))
    poses = torch.cat((train_poses, val_vos))
    return ims, poses

  def __len__(self):
    return len(self.val_set)

class OnlyPoses(data.Dataset):
  """
  Returns real poses aligned with GT poses
  """
  def __init__(self, dataset, *args, **kwargs):
    kwargs = dict(kwargs, skip_images=True)
    if dataset == '7Scenes':
      from seven_scenes import SevenScenes
      self.real_dset = SevenScenes(*args, real=True, **kwargs)
      self.gt_dset   = SevenScenes(*args, real=False, **kwargs)
    elif dataset == 'RobotCar':
      from robotcar import RobotCar
      self.real_dset = RobotCar(*args, real=True, **kwargs)
      self.gt_dset   = RobotCar(*args, real=False, **kwargs)
    else:
      raise NotImplementedError

  def __getitem__(self, index):
    """
    :param index:
    :return: poses: 2 x 7
    """
    _, real_pose = self.real_dset[index]
    _, gt_pose   = self.gt_dset[self.real_dset.gt_idx[index]]

    return real_pose, gt_pose

  def __len__(self):
    return len(self.real_dset)
