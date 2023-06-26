import os
import json
import pandas as pd
data_dir = '/expanse/projects/nemar/openneuro'

def get_hed_from_dict(d, desc, hed):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == 'HED':
                if 'Levels' in d:
                    for level, value in d['Levels'].items():
                        if level in v:
                            desc.append(value)
                            hed.append(v[level])

            if isinstance(v, dict):
                get_hed_from_dict(v, desc, hed)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        get_hed_from_dict(item, desc, hed)
    return desc, hed

dataset = []
for dir in os.listdir(data_dir):
    if dir.startswith('ds'):
        dataset_description = os.path.join(data_dir, dir, 'dataset_description.json')
        if os.path.exists(dataset_description):
            with open(dataset_description, 'r') as f:
                dataset_description = json.load(f)
            if 'HEDVersion' in dataset_description:
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

                        # parse the events json file recursively and get any dictionary that has 'HED' as a key
                        desc, hed = get_hed_from_dict(events_json, [], [])
                        print(desc)
                        print(hed)

                        # create dataframe with desc and hed
                        df = pd.DataFrame({'desc': desc, 'hed': hed})
                        print(df.head())
                        dataset.append(df)
print(dataset)
dataset = pd.concat(dataset)
print(dataset)
with open('hed_dataset.csv', 'w') as out:
    dataset.to_csv(out)
