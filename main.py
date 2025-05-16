import json
import shutil

with open(r'D:\projects\convert\train_path\labels.json', encoding='utf-8') as file:
    data = json.load(file)

data = dict(set(data.items()))
k = list(data.keys())
new_train = {}
new_val = {}

i = 0
for key in k:
    val = data[key]
    if len(val) < 30 and "R" not in val and 'h' not in val and 'x' not in val and 'c' not in val:
        i += 1
        if i < 52500:
            new_train[key] = val
        else:
            new_val[key] = val
            src = r'D:\projects\convert\train_path\images' + "\\" + key
            dst = r'D:\projects\convert\val\images'
            shutil.move(src, dst)

with open(r'D:\projects\convert\train_path\train.json', encoding='utf-8', mode='w') as file:
    json.dump(new_train, file, indent=3, ensure_ascii=False)

with open(r'D:\projects\convert\val\labels.json', encoding='utf-8', mode='w') as file:
    json.dump(new_val, file, indent=3, ensure_ascii=False)
