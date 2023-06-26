import os
import json
data_dir = '/expanse/projects/nemar/openneuro'

for dir in os.listdir(data_dir):
    if dir.startswith('ds'):
        dataset_description = os.path.join(data_dir, dir, 'dataset_description.json')
        if os.path.exists(dataset_description):
            with open(dataset_description, 'r') as f:
                dataset_description = json.load(f)
            if 'HEDVersion' in dataset_description:
                print(dir, dataset_description['HED'])
                events_json = None
                for root, dirs, files in os.walk(os.path.join(data_dir, dir)):
                    for file in files:
                        if file.endswith('events.json'):
                            events_json = file
                            break
                    if events_json: 
                        break
                
                if events_json:
                    with open(os.path.join(root, events_json), 'r') as f:
                        events_json = json.load(f)
                        print(events_json)