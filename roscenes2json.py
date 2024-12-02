from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
from roscenes.data.scene import Scene
from typing import ClassVar, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from pathlib import Path
import json
from tqdm import tqdm

paths = [
    '/home/wiss/mejo/storage-deepscenario/RoScenes/validation/validation/night_split_validation_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/validation/validation/s001_split_validation_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/validation/validation/s002_split_validation_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/validation/validation/s003_split_validation_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/validation/validation/s004_split_validation_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/train/train/night_split_train_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/train/train/s001_split_train_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/train/train/s002_split_train_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/train/train/s003_split_train_difficulty_mixed_ambience_day/',
    '/home/wiss/mejo/storage-deepscenario/RoScenes/train/train/s004_split_train_difficulty_mixed_ambience_day/',
]

save_path = '/home/wiss/mejo/storage-deepscenario/RoScenes/trainval.json'

video2camera = dict()
camera_ids = 0
image_id = 0
frame_id = -1
ann_id = 0

data = {
    'images': [],
    'annotations': [],
    'categories': [{'id': 1, 'name': 'truck'},{'id': 2, 'name': 'bus'},{'id': 3, 'name': 'van'},{'id': 4, 'name': 'car'}],
    'frames': []
}

for path in tqdm(paths):
    scene = Scene.load(path)
    for i in tqdm(range(len(scene))):
        frame = scene[i]
        for j, (token, camera) in enumerate(sorted(frame.parent.cameras.items(), key=lambda x: (np.linalg.inv(x[1].extrinsic)[0, -1], x[1].intrinsic[0, 0]))):
            clip = token.split('_')[0]
            camera_name = token.split('_')[-1]
            set_name = Path(frame.images[token]).parent.parent.parent.parent.parent.name
            image_path = os.path.join(set_name, os.path.basename(path), "images", clip, camera_name, os.path.basename(frame.images[token]))
            video_name = clip + '_' + camera_name

            if video_name not in video2camera:
                video2camera[video_name] = {'extrinsics': camera.extrinsic.tolist(), 'intrinsics': camera.intrinsic.tolist(), 'id': camera_ids}
                camera_ids += 1

            if j == 0:
                frame_id += 1
                data['frames'].append({
                    'id': frame_id,
                    'set': set_name,
                    'clip': clip,
                    'timestamp': frame.timeStamp,
                    'index': frame.index
                })

                for track_id, label, box, occ, velocity in zip(frame.instancesIDs, frame.labels, frame.boxes3D, frame.instanceOcc, frame.velocities):
                    data['annotations'].append({
                        'id': ann_id,
                        'frame_id': frame_id,
                        'camera_id': camera_ids -1,
                        'location': box[:3].tolist(),
                        'dim': box[[4, 3, 5]].tolist(),
                        'rotvec': Rotation.from_quat(box[6:]).as_rotvec().tolist(),
                        'category_id': int(label),
                        'track_id': int(track_id),
                        'occlusion': float(occ),
                        'velocity': np.asarray(velocity).astype(float).tolist()
                    })
                    ann_id += 1

            data['images'].append({
                'id': image_id,
                'image_path': image_path,
                'width': 1920,
                'height': 1080,
                'camera_id': video2camera[video_name]['id'],
                'frame_id': frame_id,
                'split': 'train' if 'train' in path else 'val'
            })
            image_id += 1

data['cameras'] = list(video2camera.values())
json.dump(data, open(save_path, 'w'))
