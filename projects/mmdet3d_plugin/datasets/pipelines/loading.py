import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


@PIPELINES.register_module()
class LoadMultiRadarFromFiles:
    """Load radar points.

    Args:
        load_dim (int): Dimension number of the loaded points. Defaults to 18.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 3, 4].
    """

    def __init__(self, to_float32=False, load_dim=18, use_dim=None):
        self.load_dim = load_dim
        if use_dim is None:
            use_dim = [0, 1, 2]
        self.use_dim = use_dim
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = results["radar_filename"]
        radar = []
        for name, radar2lidar_r, radar2lidar_t in zip(
            filename, results["radar2lidar_rs"], results["radar2lidar_ts"]
        ):
            radar_point_cloud = RadarPointCloud.from_file(name)
            points = radar_point_cloud.points
            points = points.transpose()
            points = np.copy(points).reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            points = points @ radar2lidar_r + radar2lidar_t
            radar.append(points)

        radar = np.concatenate(radar, axis=0)
        if self.to_float32:
            radar = radar.astype(np.float32)
        results["radar_points"] = radar
        return results

@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 to_float32=True,
                 height_expand=False,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=5, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 compensate_velocity=False, 
                 normalize_dims=[(3, 0, 50), (4, -100, 100), (5, -100, 100)], 
                 filtering='default', 
                 normalize=False, 
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.compensate_velocity = compensate_velocity
        self.normalize_dims = normalize_dims
        self.filtering = filtering 
        self.normalize = normalize
        self.to_float32 = to_float32
        self.height_expand = height_expand
        self.xmin, self.xmax, self.ymin, self.ymax = (
            pc_range[0],
            pc_range[3],
            pc_range[1],
            pc_range[4],
        ) 

        self.encoding = [
            (3, 'one-hot', 8), # dynprop
            (11, 'one-hot', 5), # ambig_state
            (14, 'one-hot', 18), # invalid_state
            (15, 'ordinal', 7), # pdh
            (0, 'nusc-filter', 1) # binary feature: 1 if nusc would have filtered it out
        ]


    def perform_encodings(self, points, encoding):
        for idx, encoding_type, encoding_dims in self.encoding:

            assert encoding_type in ['one-hot', 'ordinal', 'nusc-filter']

            feat = points[:, idx]

            if encoding_type == 'one-hot':
                encoding = np.zeros((points.shape[0], encoding_dims))
                encoding[np.arange(feat.shape[0]), np.rint(feat).astype(int)] = 1
            if encoding_type == 'ordinal':
                encoding = np.zeros((points.shape[0], encoding_dims))
                for i in range(encoding_dims):
                    encoding[:, i] = (np.rint(feat) > i).astype(int)
            if encoding_type == 'nusc-filter':
                encoding = np.zeros((points.shape[0], encoding_dims))
                mask1 = (points[:, 14] == 0)
                mask2 = (points[:, 3] < 7)
                mask3 = (points[:, 11] == 3)

                encoding[mask1 & mask2 & mask3, 0] = 1


            points = np.concatenate([points, encoding], axis=1)
        return points

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """

        invalid_states, dynprop_states, ambig_states = {
            'default': ([0], range(7), [3]), 
            'none': (range(18), range(8), range(5)), 
        }[self.filtering]

        radar_obj = RadarPointCloud.from_file(
            pts_filename, 
            invalid_states, dynprop_states, ambig_states
        )

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def normalize_feats(self, points, normalize_dims):
        for dim, min, max in normalize_dims:
            points[:, dim] -= min 
            points[:, dim] /= (max-min)
        return points

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']

        points_sweep_list = []
        # points_sweep_list_prev = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num: # prev_radar_points
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = sweeps[0]['timestamp'] * 1e-6

            # print('teste sweep number=', idxes)
            # print('key=',key)
            # print('begin_sweeps')
            # print(sweeps)
            # print('end_sweeps')
            
            # print(quebre)

            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                if self.compensate_velocity:
                    points_sweep[:, :2] += velo_comp * time_diff
                
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)

                # current format is x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms timestamp
                points_sweep_list.append(points_sweep_)
                

        points = np.concatenate(points_sweep_list, axis=0)
        points = self.perform_encodings(points, self.encoding)
        points = points[:, self.use_dim]

        if self.normalize:
            points = self.normalize_feats(points, self.normalize_dims)
        
        # points = RadarPoints(
        #     points, points_dim=points.shape[-1], attribute_dims=None
        # )
        
        # results['radar'] = points # ruan (era assim antes)
        
        if self.height_expand:
            augmented_points = np.zeros((3, 1))
            # print('augmented_points.shape', augmented_points.shape)
            # print(points.shape)
            # for b in range(len(points)):
            #     points_repeated = np.repeat(points[b, np.newaxis], 8, axis=0)
            #     points_repeated[:, 2] = torch.arange(0.25, 2.25, 0.25)
            #     augmented_points = np.append(augmented_points, points_repeated,axis=0)
            
            # augmented_points = np.delete(augmented_points, 0)
            # print('augmented_points',augmented_points.shape)
            # augmented_points.append('aa')
        mask_range = []
        for point in points:
            if point[0] <= self.xmax and \
               point[0] >= self.xmin and \
               point[1] <= self.ymax and \
               point[1] >= self.ymin:
                   mask_range.append(True)
            else: mask_range.append(False)
        points = points[mask_range]
        
        if self.to_float32:
            points = points.astype(np.float32)
            
        results["radar_points"] = points
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'