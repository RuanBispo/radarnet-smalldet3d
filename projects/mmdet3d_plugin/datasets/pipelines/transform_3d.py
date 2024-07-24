import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from numpy import random
import cv2 as cv
# import matplotlib.pyplot as plt
# import os


@PIPELINES.register_module()
class PadMultiViewImage:
    """Pad the multi-view image.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]

        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage:
    """Normalize the image.

    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation is applied with a
    probability of 0.5.

    The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            # print(img.shape)
            # img_temp = mmcv.bgr2hsv(img[:,:,0:3])
            #
            # img = np.concatenate((img_temp,img[:,:,3][...,np.newaxis]),axis=2)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            # print(img.shape)
            # img_temp = mmcv.hsv2bgr(img[:,:,0:3])
            # img = np.concatenate((img_temp,img[:,:,3][...,np.newaxis]),axis=2)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class CustomCollect3D:
    """Collect data from the loader relevant to the specific task. This is usually the last stage
    of the data loader pipeline. Typically keys is set to some subset of "img", "proposals",
    "gt_bboxes", "gt_bboxes_ignore", "gt_labels", and/or "gt_masks". The "img_meta" item is always
    populated.  The contents of the "img_meta" dictionary depends on "meta_keys". By default this
    includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "prev_idx",
            "next_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "scene_token",
            "can_bus",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        data = {}
        img_metas = {}

        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data["img_metas"] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage:
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results["img"]]
        x_size = [int(img.shape[1] * rand_scale) for img in results["img"]]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results["img"] = [
            mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False)
            for idx, img in enumerate(results["img"])
        ]
        lidar2img = [scale_factor @ l2i for l2i in results["lidar2img"]]
        results["lidar2img"] = lidar2img
        results["img_shape"] = [img.shape for img in results["img"]]
        results["ori_shape"] = [img.shape for img in results["img"]]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.scales}, "
        return repr_str


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles:
    """Load multichannel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack([mmcv.imread(name, self.color_type) for name in filename], axis=-1)

        # Add Radar Points as anchor points into the image
        if results["radar_pts"]:
            img_shape = img.shape
            radar_pts = results["radar_pts"]
            # img_new = np.zeros((img_shape[0], img_shape[1], img_shape[2] + 1,img_shape[3]), dtype=img.dtype)
            # img_new[:,:,:img_shape[2],:] = img
            for batch_num in range(img_shape[3]):
                for x, y in zip(radar_pts[batch_num][0], radar_pts[batch_num][1]):
                    img[round(y), round(x), 0, batch_num] = min(
                        (img[round(y), round(x), 0, batch_num] + 85), 255
                    )
                    img[round(y), round(x), 1, batch_num] = min(
                        (img[round(y), round(x), 1, batch_num] + 85), 255
                    )
                    img[round(y), round(x), 2, batch_num] = min(
                        (img[round(y), round(x), 2, batch_num] + 85), 255
                    )

        if self.to_float32:
            img = img.astype(np.float32)

        # print('img_size',img.shape)

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        # print(results['img'][0].shape)

        results["img_shape"] = img.shape

        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class RadarPoints2BEVHistogram:
    def __init__(self, pc_range, bev_h, bev_w):
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.xmin, self.xmax, self.ymin, self.ymax = (
            pc_range[0],
            pc_range[3],
            pc_range[1],
            pc_range[4],
        )

    def __call__(self, results):
        # RADAR MULTI
        radar_points = results["radar_points"]
        radial_vel = np.sqrt(radar_points[:, 3]**2 + radar_points[:, 4]**2)
        radar_fetures = [radar_points[:, 0], radar_points[:, 1], radar_points[:, 2], radial_vel]
        # radar_fetures = radar_points
        # vel_max = np.max((radar_points[:, 3],radar_points[:, 4]))*1.001
        # vel_min = np.min((radar_points[:, 3],radar_points[:, 4]))*1.001
        
        # print('BEfore=',vel_min, vel_max)
        
        hist_bins = [self.bev_w, self.bev_h, 8, 100]
        # hist_range = [[self.xmin, self.xmax], [self.ymin, self.ymax], [0, 7], [vel_min, vel_max], [vel_min, vel_max]]
        hist_range = [[self.xmin, self.xmax], [self.ymin, self.ymax], [0, 7], [0, 25]]
        
        radar_raw, edges = np.histogramdd(
            radar_fetures,
            bins=hist_bins,
            range=hist_range
        )
        
        # weightsy = edges[-1][0:100]
        # weightsx = edges[-2][0:100]
        
        # print('weights0_100', edges[-1][0:100], edges[-2][0:100])
        # print('weights1_101', edges[-1][1:101], edges[-2][1:101])

        # radar_hist = np.sum(radar_raw, axis=2)
        # radar_hist_vely = np.average(np.sum(radar_hist, axis=-2), axis=-1, weights=weightsy)
        # radar_hist_velx = np.average(np.sum(radar_hist, axis=-1), axis=-1, weights=weightsx)
        # radar_hist_pos  = np.sum(np.sum(radar_hist, axis=-1), axis=-1)
        # radar_flag = np.sum(np.sum(radar_raw, axis=-1), axis=-1)
        
        # print('velocidades_max (x,y)=', np.max(radar_hist_velx), np.max(radar_hist_vely))
        # print('velocidades_min (x,y)=', np.min(radar_hist_velx), np.min(radar_hist_vely))
        
        weights = edges[-1][1:101]
    
        radar_hist = np.sum(radar_raw, axis=2)
        radar_hist_vel = np.average(radar_hist, axis=-1, weights=weights)
        radar_hist_pos = np.sum(radar_hist, axis=-1)
        radar_flag = np.sum(radar_raw, axis=-1)
        
        radar_concat =  np.dstack((radar_hist_pos, radar_hist_vel, radar_flag)).astype(radar_points.dtype)
        
        # weights = edges[-1][0:100]
    
        # radar_hist = np.sum(radar_raw, axis=2)
        # radar_hist_vel = np.average(radar_hist, axis=-1, weights=weights)
        # radar_hist_pos = np.sum(radar_hist, axis=-1)
        # radar_flag = np.sum(radar_raw, axis=-1)
            
        # radar_concat =  np.dstack((radar_hist_pos, radar_hist_vel, radar_flag)).astype(radar_points.dtype)
        # radar_concat[radar_concat > 25] = 25
        # results["radar_hist"] = radar_concat
        
        # morphological erosion
        # defining kernel
        kernel = np.ones((3,3),np.uint8)    
             
        for layer in range(radar_concat.shape[-1]):
            
            #getting eroded image
            erod_im=cv.dilate(radar_concat[...,layer],kernel,anchor=(1, 1),iterations=1)
            erod_im=cv.erode(erod_im,kernel,anchor=(1, 1),iterations=1)      
            
            if layer == 0:
                expanded_hist = erod_im
            else:
                expanded_hist =  np.dstack((expanded_hist, erod_im)).astype(radar_points.dtype)   
        expanded_hist[expanded_hist > 25] = 25
        results["radar_hist"] = expanded_hist
        
        ########################################################################
        # PLOTS: ###############################################################
        ########################################################################
        # fig , ax = plt.subplots(2, 2, figsize=(20, 20))
        # plt.subplot(2, 2, 1)
        # plt.imshow(expanded_hist[...,1], interpolation='nearest', origin='lower', extent=[self.xmin,self.xmax,self.ymin,self.ymax], aspect="auto")
        # # plt.colorbar(c) 
        
        # # ax[1].plot(radar_points[:,1], radar_points[:,0], 'o', color='red', alpha=0.2, markersize=10) # filtered
        # # Show ego vehicle.
        
        # ax[0, 0].plot(0, 0, 'x', color='white')
        # ax[0, 1].plot(0, 0, 'x', color='white')
        # ax[1, 0].plot(0, 0, 'x', color='white')
        # ax[1, 1].plot(0, 0, 'x', color='white')
        
        # plt.title(f"radar_points [{len(radar_points)}]")
        
        # plt.subplot(2, 2, 2)
        # plt.imshow(expanded_hist[...,2], interpolation='nearest', origin='lower', extent=[self.xmin,self.xmax,self.ymin,self.ymax], aspect="auto")
        
        # plt.subplot(2, 2, 3)
        # plt.imshow(expanded_hist[...,0], interpolation='nearest', origin='lower', extent=[self.xmin,self.xmax,self.ymin,self.ymax], aspect="auto")
        
        # # plt.colorbar(c) 
        # # plt.title(f"radar_concat_raw [{len(radar_concat)}]")
        # # ax[0, 1].plot(0, 0, 'x', color='white')
        # # print(results.keys())
        
        # if results["gt_bboxes_3d"] is not None:
        #     gt_bboxes_3d = results["gt_bboxes_3d"]
        #     corners0 = gt_bboxes_3d.corners.cpu().numpy()[:,0,:]
        #     corners1 = gt_bboxes_3d.corners.cpu().numpy()[:,3,:]
        #     corners2 = gt_bboxes_3d.corners.cpu().numpy()[:,4,:]
        #     corners3 = gt_bboxes_3d.corners.cpu().numpy()[:,7,:]
        #     x_values = corners0[:,0], corners1[:,0], corners3[:,0], corners2[:,0], corners0[:,0]
        #     y_values = corners0[:,1], corners1[:,1], corners3[:,1], corners2[:,1], corners0[:,1]
        #     ax[0, 0].plot(y_values, x_values, color='white', alpha=0.6)
        #     ax[0, 1].plot(y_values, x_values, color='white', alpha=0.6)
        #     ax[1, 0].plot(y_values, x_values, color='white', alpha=0.6)
        #     ax[1, 1].plot(y_values, x_values, color='white', alpha=0.6)
        
        # # Limit visible range.
        # # ax[0].set_xlim(self.xmin,self.xmax)
        # # ax[0].set_ylim(self.ymin,self.ymax)
        # # ax[1].set_xlim(self.xmin,self.xmax)
        # # ax[1].set_ylim(self.ymin,self.ymax)
        # ax[0, 0].set_xlim(self.xmin,self.xmax)
        # ax[0, 1].set_xlim(self.xmin,self.xmax)
        # ax[1, 0].set_xlim(self.xmin,self.xmax)
        # ax[1, 1].set_xlim(self.xmin,self.xmax)
        # ax[0, 0].set_ylim(self.ymin,self.ymax)
        # ax[0, 1].set_ylim(self.ymin,self.ymax)
        # ax[1, 0].set_ylim(self.ymin,self.ymax)
        # ax[1, 1].set_ylim(self.ymin,self.ymax)

        
        # # print('pos =',np.max(expanded_hist[...,0]), np.min(expanded_hist[...,0]))
        # # print('vel =',np.max(expanded_hist[...,1]), np.min(expanded_hist[...,1]))
        
        # dir_path = 'image'
        # nmr_of_files = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
        # plt.savefig(f'{dir_path}/radar_bev_{nmr_of_files}.png', dpi=300)
        
        # if nmr_of_files >= 20: print(quebre)

        return results


@PIPELINES.register_module()
class GTDepth:
    def __init__(self, keyframe_only=False):
        self.keyframe_only = keyframe_only 

    def __call__(self, data):
        sensor2ego = data['camera2ego'].data
        cam_intrinsic = data['camera_intrinsics'].data 
        img_aug_matrix = data['img_aug_matrix'].data 
        bev_aug_matrix = data['lidar_aug_matrix'].data
        lidar2ego = data['lidar2ego'].data 
        camera2lidar = data['camera2lidar'].data
        lidar2image = data['lidar2image'].data

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        points = data['points'].data 
        img = data['img'].data

        if self.keyframe_only:
            points = points[points[:, 4] == 0]

        batch_size = len(points)
        depth = torch.zeros(img.shape[0], *img.shape[-2:]) #.to(points[0].device)

        # for b in range(batch_size):
        cur_coords = points[:, :3]

        # inverse aug
        cur_coords -= bev_aug_matrix[:3, 3]
        cur_coords = torch.inverse(bev_aug_matrix[:3, :3]).matmul(
            cur_coords.transpose(1, 0)
        )
        # lidar2image
        cur_coords = lidar2image[:, :3, :3].matmul(cur_coords)
        cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = cur_coords[:, 2, :]
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = (
            (cur_coords[..., 0] < img.shape[2])
            & (cur_coords[..., 0] >= 0)
            & (cur_coords[..., 1] < img.shape[3])
            & (cur_coords[..., 1] >= 0)
        )
        for c in range(on_img.shape[0]):
            masked_coords = cur_coords[c, on_img[c]].long()
            masked_dist = dist[c, on_img[c]]
            depth[c, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        data['depths'] = depth 
        return data
