import pandas as pd
import os
import numpy as np
import shutil 
from sklearn.model_selection import train_test_split
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, 'chexpert')
healthy = os.listdir(os.path.join(dir_path, 'chexpert', '0_healthy'))
pleu = os.listdir(os.path.join(dir_path, 'chexpert', '1_pleural_effusion'))

pleu_df = pd.DataFrame(pleu).sample(len(healthy), random_state=42)
healthy_df = pd.DataFrame(healthy)

for d in ['test', 'train', 'validate']:
    directory = os.path.join(data_dir, d)
    if not os.path.exists(directory):
        os.mkdir(directory)
        for d2 in ['healthy', 'pleural_effusion']:
            child_dir = os.path.join(directory, d2)
            if not os.path.exists(child_dir):
                os.mkdir(child_dir)


htrain, htest = train_test_split(healthy_df, test_size=0.1)
htrain, hval = train_test_split(htrain, test_size=0.1)
ptrain, ptest = train_test_split(pleu_df, test_size=0.1)
ptrain, pval = train_test_split(ptrain, test_size=0.1)

def img_to_npy(img_path):
    image = Image.open(img_path)
    array = np.asarray(image)[None]
    return array 

for t in [(htrain, 'train'), (htest, 'test'), (hval, 'validate')]:
    df, directory = t
    for i in df[0].values.tolist():
        input_file = os.path.join(data_dir, '0_healthy', i)
        target_file = os.path.join(data_dir, directory, 'healthy', os.path.splitext(i)[0]+'.npy')
        # shutil.copyfile(input_file, target_file)
        array = img_to_npy(input_file)
        np.save(target_file, array)

for t in [(ptrain, 'train'), (ptest, 'test'), (pval, 'validate')]:
    df, directory = t
    for i in df[0].values.tolist():
        input_file = os.path.join(data_dir, '1_pleural_effusion', i)
        target_file = os.path.join(data_dir, directory, 'pleural_effusion', os.path.splitext(i)[0]+'.npy')
        # shutil.copyfile(input_file, target_file)
        array = img_to_npy(input_file)
        np.save(target_file, array)
