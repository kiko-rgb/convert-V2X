import json
import numpy as np
from scipy.spatial.transform import Rotation

def values_exist(data, key, values, categories = False):
    existing_values = {item.get(key) for item in data}
    if not categories:
        return values in existing_values, values
    else:
        missing_values = [value for value in values if value not in existing_values]
        return len(missing_values) == 0, missing_values

def transform_to_matrix(transform, transposed = False):
    R = np.array(transform['rotation']).reshape(3, 3)
    t = np.array(transform['translation']).reshape(3,)
    matrix = np.eye(4)
    matrix[:3, :3] = R if not transposed else R.T
    matrix[:3, 3] = t if not transposed else - R.T @ t
    return matrix
    
def convert_v2x_to_new_format(v2x_path: str):
    # RoScenes data template
    data = {
    'images': [],
    'annotations': [],
    'categories': [{'id': 1, 'name': 'truck'},{'id': 2, 'name': 'bus'},{'id': 3, 'name': 'van'},{'id': 4, 'name': 'car'}],
    'frames': [],
    'cameras': []
    }

    v2x_vehicle_path = f'{v2x_path}/vehicle-side/'
    v2x_infrastructure_path = f'{v2x_path}/infrastructure-side/'
    v2x_cooperative_path = f'{v2x_path}/cooperative/'

    # load RoScenes for comparison, debug only
    roscenes_data = json.load(open('/home/stud/iko/storage/group/deepscenario/RoScenes/trainval_63300.json', 'r'))

    vehicle_data = json.load(open(v2x_vehicle_path + 'data_info.json', 'r'))
    infrastructure_data = json.load(open(v2x_infrastructure_path + 'data_info.json', 'r'))
    cooperative_data = json.load(open(v2x_cooperative_path + 'data_info.json', 'r'))

    ann_id = 0

    for sync_id, sync in enumerate(cooperative_data):
        # if sync_id > 220: # TODO: debug only to generate smaller data.json file
        #    break 

        veh_id = sync_id * 2
        inf_id = veh_id + 1
        # ========================= IMAGES =========================
        # add the vehicle image
        vehicle_dict = {
            'id': veh_id,
            'image_path': f'/home/stud/iko/storage/group/deepscenario/v2x-seq/v2x-seq/V2X-Seq-SPD/vehicle-side/images/{sync['vehicle_frame']}.jpg',
            'width': 1920, # this value needs to be confirmed
            'height': 1080, # this value needs to be confirmed
            'camera_id': 0, # assuming that only 2 cameras are in each scenario? # TODO: check if this entire image section needs to happen after 'cameras', then camera_id can be assigned better?
            'frame_id': int(sync['vehicle_sequence'])
        }
        data['images'].append(vehicle_dict)

        # add the infrastructure image
        infrastructure_dict = {
            'id': inf_id,
            'image_path': f'/home/stud/iko/storage/group/deepscenario/v2x-seq/v2x-seq/V2X-Seq-SPD/infrastructure-side/images/{sync['infrastructure_frame']}.jpg',
            'width': 1920, # this value needs to be confirmed
            'height': 1080, # this value needs to be confirmed
            'camera_id': 1, # assuming that only 2 cameras are in each scenario?
            'frame_id': int(sync['infrastructure_sequence'])
        }
        data['images'].append(infrastructure_dict)

        # ========================= FRAMES =========================
        # add the vehicle image
        frame_vehicle_dict = {
            'id': veh_id, 
            'clip': sync['vehicle_sequence'],
            'timestamp': int(vehicle_data[int(sync['vehicle_frame'])]['image_timestamp']),
            'index': 0, # not sure when this is used... for now: 0: vehicle, 1: infrastructure camera
            'scene': infrastructure_data[int(sync['infrastructure_frame'])]['intersection_loc'] + '/' + sync['vehicle_sequence'], 
            'split': 'train'} # hard coded this, not sure if applicable
        
        data['frames'].append(frame_vehicle_dict)

        # add the infrastructure image
        frame_infrastructure_dict = {
            'id': inf_id,
            'clip': sync['infrastructure_sequence'],
            'timestamp': int(infrastructure_data[int(sync['infrastructure_frame'])]['image_timestamp']),
            'index': 1, # not sure when this is used... for now: 0: vehicle, 1: infrastructure camera
            'scene': infrastructure_data[int(sync['infrastructure_frame'])]['intersection_loc'] + '/' + sync['infrastructure_sequence'],
            'split': 'train'} # hard coded this, not sure if applicable

        data['frames'].append(frame_infrastructure_dict)
    
        # ========================= CAMERAS =========================
        # vehicle side
        vehicle_intrinsics = json.load(open(v2x_vehicle_path + vehicle_data[sync_id]['calib_camera_intrinsic_path'], 'r'))
        lidar_to_camera = json.load(open(v2x_vehicle_path + vehicle_data[sync_id]['calib_lidar_to_camera_path'], 'r'))
        lidar_to_novatel = json.load(open(v2x_vehicle_path + vehicle_data[sync_id]['calib_lidar_to_novatel_path'], 'r'))
        novatel_to_world = json.load(open(v2x_vehicle_path + vehicle_data[sync_id]['calib_novatel_to_world_path'], 'r'))
        
        # TODO: camera distortion parameters 'cam_D' currently not considered
        R_lidar_to_camera = transform_to_matrix(lidar_to_camera, transposed = True) # needs to be .T, as it is given from lidar to camera
        R_lidar_to_novatel = transform_to_matrix(lidar_to_novatel['transform'])
        R_novatel_to_world = transform_to_matrix(novatel_to_world)

        extrinsics_vehicle = R_novatel_to_world @ R_lidar_to_novatel @ R_lidar_to_camera 
        camera_vehicle_dict = {
            'extrinsics': extrinsics_vehicle.tolist(),
            'intrinsics': [vehicle_intrinsics['cam_K'][i:i+3] for i in range(0, len(vehicle_intrinsics['cam_K']), 3)],
            'id': veh_id, # NOTE: this should be handled differently for the infrastructure, as the camera is stationary !!
            'max_x': 1000, # TODO: maybe both camera dicts need an additional parameter to flag if camera is from a vehicle or infrastructure?
            'max_y': 1000,
            'min_x': -1000,
            'min_y': -1000, # TODO: these are placeholder values!!!
        }

        data['cameras'].append(camera_vehicle_dict)

        # infrastructure side
        infrastructure_intrinsics = json.load(open(v2x_infrastructure_path + infrastructure_data[sync_id]['calib_camera_intrinsic_path'], 'r'))

        # check if infrastructure camera was already added before -> since it is static, does not need to be added again to 'cameras'
        # NOTE: up until seq_id = ~8000, camera3 is the only infrastructure camera
        values_to_check = infrastructure_intrinsics['cameraID']
        exists, missing_camera = values_exist(data = data['cameras'], key = 'id', values = values_to_check, categories = False) 

        if not exists:
            virtual_lidar_to_camera = json.load(open(v2x_infrastructure_path + infrastructure_data[sync_id]['calib_virtuallidar_to_camera_path'], 'r'))
            virtual_lidar_to_world = json.load(open(v2x_infrastructure_path + infrastructure_data[sync_id]['calib_virtuallidar_to_world_path'], 'r'))

            R_virtual_lidar_to_camera = transform_to_matrix(virtual_lidar_to_camera, transposed = True) # needs to be .T, as it is given from lidar to camera
            R_virtual_lidar_to_world = transform_to_matrix(virtual_lidar_to_world)

            extrinsics_infrastructure = R_virtual_lidar_to_world @ R_virtual_lidar_to_camera 

            camera_infrastructure_dict = {
                'extrinsics': extrinsics_infrastructure.tolist(),
                'intrinsics': [infrastructure_intrinsics['cam_K'][i:i+3] for i in range(0, len(infrastructure_intrinsics['cam_K']), 3)],
                'id': infrastructure_intrinsics['cameraID'],
                'max_x': 1000, # TODO: maybe both camera dicts need an additional parameter to flag if camera is from a vehicle or infrastructure?
                'max_y': 1000,
                'min_x': -1000,
                'min_y': -1000, # TODO: these are placeholder values!!!
            }

            data['cameras'].append(camera_infrastructure_dict)

        # ========================= ANNOTATIONS =========================
        # NOTE: - Truncated: Integer [0, 1, 2] indicating 3 types of truncated state, non-truncated, transversely truncated, longitudinally truncated respectively.
        # ----- Truncated is not clear to me.

        # load the vehicle anns
        vehicle_annotations = json.load(open(v2x_vehicle_path + vehicle_data[sync_id]['label_camera_std_path'], 'r'))
        infrastructure_annotations = json.load(open(v2x_infrastructure_path + infrastructure_data[sync_id]['label_camera_std_path'], 'r'))
        
        # vehicle side
        for ann in vehicle_annotations:
            # check if any object type was not encountered before
            values_to_check = ann['type'].lower()
            exists, missing_categorie = values_exist(data = data['categories'], key = 'name', values = values_to_check, categories = False) # TODO: Refactor function, both use categories=False
            # add it to categories if it did not exist there before
            if not exists:
                data['categories'].append({'id': len(data['categories']) + 1, 'name': missing_categorie})

            # get object (= ann) position in world frame; objects are noted in LiDAR/virtual LiDAR frame, but should be in world frame:
            location_in_world_frame = R_novatel_to_world @ R_lidar_to_novatel @ np.vstack((np.array(list(ann['3d_location'].values())).reshape(-1, 1), [[1]]))
            # get object rotation matrix/rotvec
            egoc_rot_matrix = Rotation.from_euler('xyz', [np.pi / 2, ann['rotation'], 0]).as_rotvec() # TODO: if this gives trouble, try other values instead of np.pi/2
            
            # get occlusion score: from V2X [0, 1, 2] (fully visible, partly occluded, largely occluded) to RoScenes [float]
            occlusion_score = 0 if ann['occluded_state'] == 0 else 0.5 if ann['occluded_state'] == 1 else 1

            vehicle_annotation_dict = {
                'id': ann_id,
                'frame_id': sync_id, # NOTE: !!! This is experimental, meaning all veh and inf are grouped together with another -> differs from previous approach(?)
                'location': location_in_world_frame.reshape(4, ).tolist()[:-1],  
                'dim': [ann['3d_dimensions']['l'], ann['3d_dimensions']['w'], ann['3d_dimensions']['h']], 
                'rotvec': egoc_rot_matrix.tolist(), # TODO: questionable if this is already good to go, w.r.t correct coordinate system 
                'category_id': next((d['id'] for d in data['categories'] for key, value in d.items() if value == ann['type'].lower()), None),  
                'track_id': int(ann['track_id']), 
                'occlusion': occlusion_score, 
                'velocity': [0.0, 0.0]  # NOTE: 2D velocity will be not filled out?
            }

            ann_id += 1 # NOTE: do not touch this line!!!
            data['annotations'].append(vehicle_annotation_dict)

        # infrastructure side
        for ann in infrastructure_annotations:
            # check if any object type was not encountered before
            values_to_check = ann['type'].lower()
            exists, missing_categorie = values_exist(data = data['categories'], key = 'name', values = values_to_check, categories = False) # TODO: Refactor function, both use categories=False
            # add it to categories if it did not exist there before
            if not exists:
                data['categories'].append({'id': len(data['categories']) + 1, 'name': missing_categorie})

            # get object (= ann) position in world frame; objects are noted in LiDAR/virtual LiDAR frame, but should be in world frame:
            location_in_world_frame = R_virtual_lidar_to_world @ np.vstack((np.array(list(ann['3d_location'].values())).reshape(-1, 1), [[1]]))
            # get object rotation matrix/rotvec
            egoc_rot_matrix = Rotation.from_euler('xyz', [np.pi / 2, ann['rotation'], 0]).as_rotvec() # TODO: if this gives trouble, try other values instead of np.pi/2
            
            # get occlusion score: from V2X [0, 1, 2] (fully visible, partly occluded, largely occluded) to RoScenes [float]
            occlusion_score = 0 if ann['occluded_state'] == 0 else 0.5 if ann['occluded_state'] == 1 else 1

            infrastructure_annotation_dict = {
                'id': ann_id,
                'frame_id': sync_id, # NOTE: !!! This is experimental, meaning all veh and inf are grouped together with another -> differs from previous approach(?)
                'location': location_in_world_frame.reshape(4, ).tolist()[:-1],  
                'dim': [ann['3d_dimensions']['l'], ann['3d_dimensions']['w'], ann['3d_dimensions']['h']], 
                'rotvec': egoc_rot_matrix.tolist(), # TODO: questionable if this is already good to go, w.r.t correct coordinate system 
                'category_id': next((d['id'] for d in data['categories'] for key, value in d.items() if value == ann['type'].lower()), None),  
                'track_id': int(ann['track_id']), 
                'occlusion': occlusion_score, 
                'velocity': [0.0, 0.0]  # NOTE: 2D velocity will be not filled out?
            }

            ann_id += 1 # NOTE: do not touch this line!!!
            data['annotations'].append(infrastructure_annotation_dict)

        print(sync_id)
    return data

def main():
    # load RoScenes for comparison, debug only
    roscenes_data = json.load(open('/home/stud/iko/storage/group/deepscenario/RoScenes/trainval_63300.json', 'r'))

    v2x_path = '/home/stud/iko/storage/group/deepscenario/v2x-seq/v2x-seq/V2X-Seq-SPD'
    save_path = '/home/stud/iko/test/V2X-Seq-test/v2x.json'
    new_data = convert_v2x_to_new_format(v2x_path)

    #print([type(roscenes_data[key]) for key in roscenes_data.keys()])
    #print([type(new_data[key]) for key in new_data.keys()])

    with open(save_path, 'w') as file:
        json.dump(new_data, file)


if __name__ == "__main__":
    main()