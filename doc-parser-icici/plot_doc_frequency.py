import cv2 as cv
import time
import operator
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
dataset = {}
rootdir = r'icici 547 docs'
dir_names = [path.name for path in Path(rootdir).iterdir() if path.is_dir()]

for dir in dir_names:
  path = Path(rootdir, dir)
  file_count = len([f for f in path.rglob('*') if f.is_file()])
  dataset[dir] = file_count 
  print(dir, file_count)
print(sum(dataset.values()))

df = pd.DataFrame(dataset.items())
df.sort_values(by=['C'], ascending=False)

df.to_csv('number_of_docs_new2.csv')

sorted_d = dict( sorted(dataset.items(), key=operator.itemgetter(1),reverse=True)[:8])   

plt.figure(figsize=(25, 7))
plt.bar(*zip(*sorted_d.items()))
plt.show()