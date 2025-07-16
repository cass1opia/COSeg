import os
import numpy as np

import torch
from torch.utils.data import Dataset

import pickle
import glob
from itertools import combinations
from itertools import permutations
from util.data_util import data_prepare_v101 as data_prepare


class S3DIS_base(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
    ):
        super().__init__()
        (
            self.split,
            self.voxel_size,
            self.transform,
            self.voxel_max,
            self.shuffle_index,
            self.loop,
        ) = (split, voxel_size, transform, voxel_max, shuffle_index, loop)

        self.data_root = data_root
        # Classes: {0:'ceiling', 1:'floor', 2:'wall', 3:'beam',
        #           4:'column', 5:'window', 6:'door', 7:'table',
        #           8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
        self.class_count = 13
        self.class_names = open(
            os.path.join(
                os.path.dirname(os.path.dirname(data_root)),
                "meta",
                "s3dis_classnames.txt",
            )
        ).readlines()
        self.class2type = {
            i: name.strip() for i, name in enumerate(self.class_names)
        }
        print(self.class2type)
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.fold_0 = [
            "beam",
            "board",
            "bookcase",
            "ceiling",
            "chair",
            "column",
        ]
        self.fold_1 = ["door", "floor", "sofa", "table", "wall", "window"]

        if cvfold == 0:
            self.test_classes = [self.type2class[i] for i in self.fold_0]
        elif cvfold == 1:
            self.test_classes = [self.type2class[i] for i in self.fold_1]
        else:
            raise NotImplementedError(
                "Unknown cvfold (%s). [Options: 0,1]" % cvfold
            )
        all_classes = [i for i in range(0, self.class_count - 1)]
        self.train_classes = [
            c for c in all_classes if c not in self.test_classes
        ]

        self.class2scans = self.get_class2scans()  # class, blockname

    def get_class2scans(self):
        """
        Build the class to scans mapping.
        """
        class2scans_file = os.path.join(
            os.path.dirname(self.data_root), "class2scans.pkl"
        )
        if os.path.exists(class2scans_file):
            print(f"Loading existing class2scans from: {class2scans_file}")
            with open(class2scans_file, "rb") as f:
                class2scans = pickle.load(f)
        else:
            print("=" * 80)
            print("BUILDING CLASS-TO-SCANS MAPPING")
            print(f"Scanning data root: {self.data_root}")
            print(f"Minimum ratio: 0.05 (5%)")
            print(f"Minimum points: 100")
            print("=" * 80)
            min_ratio = (
                0.05  # to filter out scans with only rare labelled points
            )
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k: [] for k in range(self.class_count)}

            for file in glob.glob(os.path.join(self.data_root, "*.npy")):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:, 6].astype(np.int)
                classes = np.unique(labels)
                # Only show scan details for small datasets
                if len(glob.glob(os.path.join(self.data_root, "*.npy"))) <= 20:
                    print(
                        "{0} | shape: {1} | classes: {2}".format(
                            scan_name, data.shape, list(classes)
                        )
                    )
                for class_id in classes:
                    # if the number of points for the target class is too few,
                    # do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0] * min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print("==== class to scans mapping is done ====")
            for class_id in range(self.class_count):
                print(
                    "\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}".format(
                        class_id,
                        min_ratio,
                        min_pts,
                        self.class2type[class_id],
                        len(class2scans[class_id]),
                    )
                )

            with open(class2scans_file, "wb") as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
            
            print("-" * 80)
            print("CLASS-TO-SCANS MAPPING COMPLETED")
            print(f"Mapping saved to: {class2scans_file}")
            
        # Log class2scans statistics
        print("\nClass-to-Scans Statistics:")
        print(f"{'Class':<15} {'ID':<5} {'Scans':<10} {'Sample Scans (first 3)'}")
        print("-" * 80)
        for class_id in range(self.class_count):
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                scan_count = len(class2scans[class_id])
                sample_scans = class2scans[class_id][:3]
                print(f"{class_name:<15} {class_id:<5} {scan_count:<10} {sample_scans}")
        print("=" * 80)
        
        return class2scans


class S3DIS(S3DIS_base):
    """
    The S3DIS dataset class used for backbone pretraining.
    """

    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
        )

        self.class2scans = {c: self.class2scans[c] for c in self.train_classes}

        train_block_names = []
        all_block_names = []
        for _, v in sorted(self.class2scans.items()):
            all_block_names.extend(v)
            n_blocks = len(v)
            n_test_blocks = int(n_blocks * 0.1)
            n_train_blocks = n_blocks - n_test_blocks
            train_block_names.extend(v[:n_train_blocks])

        if split == "train":
            self.block_names = list(set(train_block_names))
        elif split == "val":
            self.block_names = list(
                set(all_block_names) - set(train_block_names)
            )
        else:
            raise NotImplementedError("Mode is unknown!")

        print(
            "[Pretrain Dataset] Mode: {0} | Num_blocks: {1}".format(
                split, len(self.block_names)
            )
        )

    def __getitem__(self, idx):
        item = self.block_names[idx % len(self.block_names)]

        data = np.load(os.path.join(self.data_root, item + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )

        class_dict = {c: i + 1 for i, c in enumerate(self.train_classes)}
        for i, lb in enumerate(label):
            if lb.item() in class_dict.keys():
                label[i] = class_dict[lb.item()]
            else:
                label[i] = 0

        return coord, feat, label

    def __len__(self):
        return len(self.block_names) * self.loop


class S3DIS_FS(S3DIS_base):
    """
    The S3DIS dataset class used for few-shot learning.
    """

    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=50000,
        n_way=1,
        k_shot=2,
        n_queries=1,
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
        )

        self.n_way, self.k_shot, self.n_queries, self.num_episode = (
            n_way,
            k_shot,
            n_queries,
            num_episode,
        )

        if split == "train":
            self.classes = np.array(self.train_classes)
        elif split == "test":
            self.classes = np.array(self.test_classes)
        else:
            raise NotImplementedError(
                "Unkown mode %s! [Options: train/test]" % split
            )

        self.train_mapping = {
            c: i + 1 for i, c in enumerate(self.train_classes)
        }
        print("Classes: {0} in {1} set".format(self.classes, split))

    def get_test_episode(self, n_way_classes=None, verbose=False):
        """Generate a test episode without base lables."""
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(
                self.classes, self.n_way, replace=False
            )
        
        if verbose:
            print(f"    Generating episode for classes: {sampled_classes}")

        support_ptclouds, support_masks, query_ptclouds, query_labels = (
            [],
            [],
            [],
            [],
        )

        black_list = (
            []
        )  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(
                all_scannames, self.k_shot + self.n_queries, replace=False
            )
            black_list.extend(selected_scannames)
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            for scan_name in query_scannames:
                ptcloud, label = self.sample_test_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_labels.append(label)

            for scan_name in support_scannames:
                ptcloud, label = self.sample_test_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_masks.append(label)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            sampled_classes,
        )

    def sample_test_pointcloud(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        # construct test labels without base class labels
        feat = torch.cat((coord, feat), dim=1)
        if support:
            label = (label == sampled_class).int()
        else:
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            for i, lb in enumerate(label):
                if lb.item() in class_dict.keys():
                    label[i] = class_dict[lb.item()]
                else:
                    label[i] = 0

        return feat, label

    def __getitem__(self, idx, n_way_classes=None):
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(
                self.classes, self.n_way, replace=False
            )

        (
            support_ptclouds,
            support_base_masks,
            support_test_masks,
            query_ptclouds,
            query_base_labels,
            query_test_labels,
        ) = ([], [], [], [], [], [])

        black_list = (
            []
        )  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(
                all_scannames, self.k_shot + self.n_queries, replace=False
            )
            black_list.extend(selected_scannames)
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            for scan_name in query_scannames:
                ptcloud, base_label, test_label = self.sample_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_base_labels.append(base_label)
                query_test_labels.append(test_label)

            for scan_name in support_scannames:
                ptcloud, base_label, test_label = self.sample_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_base_masks.append(base_label)
                support_test_masks.append(test_label)

        return (
            support_ptclouds,
            support_base_masks,
            support_test_masks,
            query_ptclouds,
            query_base_labels,
            query_test_labels,
            sampled_classes,
        )

    def sample_pointcloud(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        feat = torch.cat((coord, feat), dim=1)
        if support:
            # Create a new label tensor for base calsses

            train_label = torch.zeros_like(label)
            for i, value in enumerate(label):
                if value.item() in self.train_classes:
                    train_label[i] = self.train_mapping[value.item()]

            test_label = (label == sampled_class).int()
            return feat, train_label, test_label
        else:

            # Create a new label tensor for base calsses
            train_label = torch.zeros_like(label)
            for i, value in enumerate(label):
                if value.item() in self.train_classes:
                    train_label[i] = self.train_mapping[value.item()]

            # Create a new label tensor for test
            test_mapping = {c: i + 1 for i, c in enumerate(sampled_classes)}
            test_label = torch.zeros_like(label)
            for i, value in enumerate(label):
                if value.item() in test_mapping.keys():
                    test_label[i] = test_mapping[value.item()]
                elif (
                    value.item() in self.test_classes and self.split == "train"
                ):
                    test_label[i] = 255

            return feat, train_label, test_label

    def __len__(self):
        return self.num_episode


class S3DIS_FSForVIS(S3DIS_FS):
    """
    The S3DIS dataset class used for visulizaling the few-shot results.
    """

    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=50000,
        n_way=1,
        k_shot=2,
        n_queries=1,
        target_class=None,
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
            num_episode,
            n_way,
            k_shot,
            n_queries,
        )

        self.target_class = target_class
        self.target_cls = self.type2class[self.target_class]
        self.combos = list(permutations(self.class2scans[self.target_cls], 2))

    def __getitem__(self, idx, n_way_classes=None):
        """Generate a test episode without base lables."""
        support_ptclouds, support_masks, query_ptclouds, query_labels = (
            [],
            [],
            [],
            [],
        )

        selected_scannames = self.combos[idx]
        query_scannames = selected_scannames[: self.n_queries]
        support_scannames = selected_scannames[self.n_queries :]

        for scan_name in query_scannames:
            ptcloud, label = self.sample_test_pointcloud(
                scan_name, [self.target_cls], self.target_cls, support=False
            )
            query_ptclouds.append(ptcloud)
            query_labels.append(label)

        for scan_name in support_scannames:
            ptcloud, label = self.sample_test_pointcloud(
                scan_name, [self.target_cls], self.target_cls, support=True
            )
            support_ptclouds.append(ptcloud)
            support_masks.append(label)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            np.array([self.target_cls]),
            selected_scannames,
        )

    def crop_point_cloud(self, point_cloud, feat, labels):
        # Count the number of points with label 1 in each half of x and y directions
        half_x = (point_cloud[:, 0].min() + point_cloud[:, 0].max()) / 2
        half_y = (point_cloud[:, 1].min() + point_cloud[:, 1].max()) / 2

        # Count points with label 1 in each quadrant
        q1_count = np.sum(
            (point_cloud[:, 0] <= half_x) & (labels == self.target_cls)
        )
        q2_count = np.sum(
            (point_cloud[:, 0] > half_x) & (labels == self.target_cls)
        )
        q3_count = np.sum(
            (point_cloud[:, 1] > half_y) & (labels == self.target_cls)
        )
        q4_count = np.sum(
            (point_cloud[:, 1] <= half_y) & (labels == self.target_cls)
        )

        # Choose the quadrant with the most points of label 1
        max_count = max(q1_count, q2_count, q3_count, q4_count)
        if max_count == q1_count:
            return (
                point_cloud[(point_cloud[:, 0] <= half_x)],
                feat[(point_cloud[:, 0] <= half_x)],
                labels[(point_cloud[:, 0] <= half_x)],
            )
        elif max_count == q2_count:
            return (
                point_cloud[(point_cloud[:, 0] > half_x)],
                feat[(point_cloud[:, 0] > half_x)],
                labels[(point_cloud[:, 0] > half_x)],
            )
        elif max_count == q3_count:
            return (
                point_cloud[(point_cloud[:, 1] > half_y)],
                feat[(point_cloud[:, 1] > half_y)],
                labels[(point_cloud[:, 1] > half_y)],
            )
        else:
            return (
                point_cloud[(point_cloud[:, 1] <= half_y)],
                feat[(point_cloud[:, 1] <= half_y)],
                labels[(point_cloud[:, 1] <= half_y)],
            )

    def sample_test_pointcloud(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]

        # do some crops to avoid OOM
        if support:
            while coord.shape[0] > 300000:
                print("Crop point cloud:", coord.shape[0])
                coord, feat, label = self.crop_point_cloud(coord, feat, label)
        else:
            while coord.shape[0] > 700000:
                print("Crop point cloud:", coord.shape[0])
                coord, feat, label = self.crop_point_cloud(coord, feat, label)

        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            # self.voxel_max,
            None,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        # construct test labels without base class labels
        feat = torch.cat((coord, feat), dim=1)
        if support:
            label = (label == sampled_class).int()
        else:
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            for i, lb in enumerate(label):
                if lb.item() in class_dict.keys():
                    label[i] = class_dict[lb.item()]
                else:
                    label[i] = 0

        return feat, label

    def __len__(self):
        return len(self.combos)


class S3DIS_FS_TEST(Dataset):
    def __init__(
        self,
        split="val",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=50000,
        n_way=1,
        k_shot=2,
        n_queries=1,
        num_episode_per_comb=100,
    ):
        super().__init__()

        self.dataset = S3DIS_FS(
            "test",
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
            num_episode,
            n_way,
            k_shot,
            n_queries,
        )
        self.classes = self.dataset.classes
        self.n_way = n_way
        self.num_episode_per_comb = num_episode_per_comb

        if split == "val":
            self.test_data_path = os.path.join(
                os.path.dirname(data_root),
                "S_%d_N_%d_K_%d_episodes_%d_pts_%d_vs_%.2f_rdmsp"
                % (
                    cvfold,
                    n_way,
                    k_shot,
                    num_episode_per_comb,
                    voxel_max,
                    voxel_size,
                ),
            )
        elif split == "test":
            self.test_data_path = os.path.join(
                os.path.dirname(data_root),
                "S_%d_N_%d_K_%d_test_episodes_%d_pts_%d_vs_%.2f"
                % (
                    cvfold,
                    n_way,
                    k_shot,
                    num_episode_per_comb,
                    voxel_max,
                    voxel_size,
                ),
            )
        else:
            raise NotImplementedError("Mode (%s) is unknown!" % split)

    def prepare_testt_data(self):
        import time
        start_time = time.time()
        
        if os.path.exists(self.test_data_path):
            self.file_names = glob.glob(
                os.path.join(self.test_data_path, "*.pt")
            )
            self.num_episode = len(self.file_names)
            print("=" * 80)
            print("Loading Pre-generated Test Episodes")
            print(f"Test data path: {self.test_data_path}")
            print(f"Found {self.num_episode} pre-generated episodes")
            print(f"Configuration: {self.n_way}-way {self.k_shot}-shot")
            print(f"Classes: {self.classes}")
            print("=" * 80)
        else:
            print("=" * 80)
            print("GENERATING NEW TEST EPISODES")
            print(f"Test data path: {self.test_data_path}")
            print(f"Configuration: {self.n_way}-way {self.k_shot}-shot")
            print(f"Classes: {self.classes} (total: {len(self.classes)})")
            print("=" * 80)
            
            os.mkdir(self.test_data_path)

            class_comb = list(
                combinations(self.classes, self.n_way)
            )  # [(),...]
            self.num_episode = len(class_comb) * self.num_episode_per_comb
            
            print(f"Class combinations: {len(class_comb)}")
            print(f"Episodes per combination: {self.num_episode_per_comb}")
            print(f"Total episodes to generate: {self.num_episode}")
            print("-" * 80)
            
            # Log class combinations
            for i, comb in enumerate(class_comb):
                print(f"Combination {i+1}/{len(class_comb)}: {comb}")
            print("-" * 80)

            episode_ind = 0
            self.file_names = []
            total_points = 0
            
            for comb_idx, sampled_classes in enumerate(class_comb):
                sampled_classes = list(sampled_classes)
                print(f"Processing combination {comb_idx+1}/{len(class_comb)}: {sampled_classes}")
                
                for ep_idx in range(self.num_episode_per_comb):
                    episode_start = time.time()
                    # Enable verbose logging for first few episodes
                    verbose_logging = episode_ind < 3
                    data = self.dataset.get_test_episode(sampled_classes, verbose=verbose_logging)
                    out_filename = os.path.join(
                        self.test_data_path, f"{episode_ind}.pt"
                    )
                    
                    # Calculate episode statistics
                    support_feat, support_label, query_feat, query_label, _ = data
                    support_points = sum(feat.shape[0] for feat in support_feat)
                    query_points = sum(feat.shape[0] for feat in query_feat)
                    episode_total_points = support_points + query_points
                    total_points += episode_total_points
                    
                    write_episode(out_filename, data)
                    self.file_names.append(out_filename)
                    
                    episode_time = time.time() - episode_start
                    progress = (episode_ind + 1) / self.num_episode * 100
                    
                    if (ep_idx + 1) % max(1, self.num_episode_per_comb // 4) == 0 or ep_idx == 0:
                        print(f"  Episode {ep_idx+1}/{self.num_episode_per_comb} - "
                              f"Support: {support_points} pts, Query: {query_points} pts, "
                              f"Time: {episode_time:.2f}s, Progress: {progress:.1f}%")
                    
                    episode_ind += 1
                
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / episode_ind if episode_ind > 0 else 0
                eta = (self.num_episode - episode_ind) * avg_time_per_episode
                print(f"  Combination {comb_idx+1} completed - "
                      f"Elapsed: {elapsed_time:.1f}s, ETA: {eta:.1f}s")
                print()
            
            total_time = time.time() - start_time
            avg_points_per_episode = total_points / self.num_episode if self.num_episode > 0 else 0
            
            print("=" * 80)
            print("TEST EPISODE GENERATION COMPLETED")
            print(f"Total episodes generated: {self.num_episode}")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"Average time per episode: {total_time/self.num_episode:.2f}s")
            print(f"Total points processed: {total_points:,}")
            print(f"Average points per episode: {avg_points_per_episode:.0f}")
            print(f"Episodes per second: {self.num_episode/total_time:.2f}")
            print(f"Points per second: {total_points/total_time:.0f}")
            print(f"Data saved to: {self.test_data_path}")
            print("=" * 80)
    
    # Alias for the correct function name (fixing typo)
    def prepare_test_data(self):
        """Alias for prepare_testt_data to fix the typo."""
        return self.prepare_testt_data()

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return read_episode(file_name)


def write_episode(out_filename, data):
    support_feat, support_label, query_feat, query_label, sampled_classes = (
        data
    )
    
    # Calculate episode statistics
    support_points = sum(feat.shape[0] for feat in support_feat)
    query_points = sum(feat.shape[0] for feat in query_feat)
    total_points = support_points + query_points
    
    # Calculate file size estimation
    episode_data = {
        "support_feat": support_feat,
        "support_label": support_label,
        "query_feat": query_feat,
        "query_label": query_label,
        "sampled_classes": sampled_classes,
    }
    
    torch.save(episode_data, out_filename)
    
    # Get actual file size
    file_size = os.path.getsize(out_filename)
    file_size_mb = file_size / (1024 * 1024)
    
    # Only print detailed info for every 10th episode or first few episodes
    episode_num = int(os.path.basename(out_filename).split('.')[0])
    if episode_num < 5 or episode_num % 10 == 0:
        print(f"\t Episode {episode_num} saved: {os.path.basename(out_filename)} | "
              f"Classes: {sampled_classes} | Points: {total_points:,} | "
              f"Size: {file_size_mb:.2f}MB")


def read_episode(file_name):
    data_file = torch.load(file_name)
    support_feat = data_file["support_feat"]
    support_label = data_file["support_label"]
    query_feat = data_file["query_feat"]
    query_label = data_file["query_label"]
    sampled_classes = data_file["sampled_classes"]
    return (
        support_feat,
        support_label,
        query_feat,
        query_label,
        sampled_classes,
    )
