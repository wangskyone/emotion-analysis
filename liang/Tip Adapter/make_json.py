import os
import glob
import json
mapping_dict = {"Anger": 0, "Disgust": 1, "Fear": 2, "Happiness": 3, "Neutral": 4, "Sadness": 5, "Surprise": 6}

data_dir = '/home/user/work/Tip-Adapter/rafdb'
train_list = []
val_list = []
for root, dir, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg'):
            file_path = os.path.join(root, file)
            emotion = os.path.basename(os.path.dirname(file_path))
            label = mapping_dict.get(emotion)
            mode = file_path.split("/")[-3]
            if mode == 'train':
                train_list.append([file_path, label, emotion])
            else:
                val_list.append([file_path, label, emotion])

json_dict = {'train': train_list, 'val': val_list, 'test': val_list}
json_data = json.dumps(json_dict)

with open("rafdb/rafdb.json", "w") as f:
    f.write(json_data)


print("---------------------")