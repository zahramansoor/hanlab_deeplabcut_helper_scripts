# determin bsoid clusters

import pandas as pd, os, numpy as np, tifffile
import multiprocessing as mp

def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''Function to load memmaped array.
    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array
    Returns
    -----------
    arr
    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr


#import video
eye = r'Y:\DLC\DLC_networks\230301_E201'
eyeimgs = [os.path.join(eye,xx) for xx in os.listdir(eye) if "tif" in xx]
img=tifffile.imread(eyeimgs[0])
y,x = img.shape
memmap = np.memmap(os.path.join(eye,os.path.basename(eye)+".npy"),
        dtype='uint8', mode='w+', shape=(len(eyeimgs),y,x))
def readtif(memmap, i, img):
    memmap[i,:,:]=tifffile.imread(img)
    if i%100==0:
        memmap.flush()
        print(i)
    return

p = mp.Pool(12)
args = [(memmap, i, eyeimg) for i,eyeimg in enumerate(eyeimgs)]
p.starmap(memmap,args)
memmap.flush()

pth = r'Y:\DLC\DLC_networks\pupil_licks_nose_paw-Zahra-2023-02-27\videos\BSOID\Mar-08-2023labels_pose_60Hz230301_E201DLC_resnet50_pupil_licks_nose_pawFeb27shuffle1_50000.csv'
df = pd.read_csv(pth, index_col=None)
labels = df["B-SOiD labels"]
#change column names
df=df.drop(columns=["Unnamed: 0", "B-SOiD labels"])
cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
df.columns = cols
df=df.drop([0,1])
df["B-SOiD labels"]=labels

#cluster 0
df[df["B-SOiD labels"]==1]