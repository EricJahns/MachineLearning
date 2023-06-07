import pandas as pd
import numpy as np
import os


csv = pd.DataFrame(columns=['image', 'mask'])

weights = [.70, .15, .15]
classes = ['train', 'val', 'test']

# use weights as the probability of each row being selected

for file in os.listdir('./archive/Images'):
    if file.endswith('.jpg'):
        csv = pd.concat([csv, pd.DataFrame({'image': file, 'mask': file}, index=[0])])

classes = np.random.choice(classes, p=weights, size=len(csv))

csv['dataset'] = classes

# csv.sort_values(axis=1, inplace=True)

csv.to_csv('./archive/metadata_prepped.csv', index=False)



