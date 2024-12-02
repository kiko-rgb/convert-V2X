import json

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

    for sync_id, sync in enumerate(cooperative_data):
        veh_id = sync_id * 2
        inf_id = veh_id + 1
        # --- IMAGES ---
        # add the vehicle image
        vehicle_dict = {
            'id': veh_id,
            'image_path': f'/home/stud/iko/storage/group/deepscenario/v2x-seq/v2x-seq/V2X-Seq-SPD/vehicle-side/images/{sync['vehicle_frame']}.jpg',
            'width': 1920, # this value needs to be confirmed
            'height': 1080, # this value needs to be confirmed
            'camera_id': 0, # assuming that only 2 cameras are in each scenario?
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

        # --- FRAMES ---
        # add the vehicle image
        frame_vehicle_dict = {
            'id': veh_id, 
            'clip': sync['vehicle_sequence'],
            'timestamp': int(vehicle_data[int(sync['vehicle_frame'])]['image_timestamp']),
            'index': int, # not sure when this is used...
            'scene': vehicle_data[int(sync['vehicle_frame'])]['sequence_id'],
            'split': 'train'} # hard coded this, not sure if applicable
        
        data['frames'].append(frame_vehicle_dict)

        # add the infrastructure image
        try:
            scene = infrastructure_data[int(sync['infrastructure_frame'])]['image_timestamp']
            ts = int(scene)
        except: 
            scene = infrastructure_data[int(sync['infrastructure_frame']) - 1]['image_timestamp']
            ts = int(scene)
            
        frame_infrastructure_dict = {
            'id': inf_id,
            'clip': sync['infrastructure_sequence'],
            'timestamp': ts,
            'index': int, # not sure when this is used...
            'scene': scene,
            'split': 'train'} # hard coded this, not sure if applicable

        data['frames'].append(frame_infrastructure_dict)


        # load the camera anns
        camera_annotations = json.load(open(v2x_vehicle_path + vehicle_data[sync_id]['label_camera_std_path'], 'r'))

        # --- CAMERAS ---

        # --- ANNOTATIONS ---

        # --- CATEGORIES ---
        key_to_check = 'name'
        # first consider all objects visible from the vehicle:
        values_to_check = set([i['type'].lower() for i in camera_annotations])
        exists, missing_categorie = values_exist(data['categories'], key_to_check, values_to_check)
        if not exists:
            for cat in missing_categorie:
                data['categories'].append({'id': len(data['categories']) + 1, 'name': cat})

    return data


def values_exist(data, key, values):
    existing_values = {item.get(key) for item in data}
    missing_values = [value for value in values if value not in existing_values]
    return len(missing_values) == 0, missing_values

def main():
    v2x_path = '/home/stud/iko/storage/group/deepscenario/v2x-seq/v2x-seq/V2X-Seq-SPD'
    save_path = '/home/stud/iko/test/V2X-Seq-test/v2x.json'
    new_data = convert_v2x_to_new_format(v2x_path)

    json.dump(new_data, open(save_path, 'w'))




if __name__ == "__main__":
    main()