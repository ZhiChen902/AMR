import numpy as np
from glob import glob

from tqdm import tqdm

for tmp in tqdm(glob("/cvlabdata2/home/zhichen/code/PEAL/geo_prior_testval/7-scenes-redkitchen/*")):
    try:
        np.load(tmp)
    except:
        print(tmp)