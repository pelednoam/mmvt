#!/usr/bin/env python
import sys
import numpy as np
import nibabel as nib
from scipy.signal import convolve
from scipy.ndimage.morphology import grey_closing
from scipy.ndimage.morphology import generate_binary_structure
from mne import write_surface
from mcubes import marching_cubes

def mkoutersurf(image, radius, outfile):
    #radius information is currently ignored
    #it is a little tougher to deal with the morphology in python

    fill = nib.load( image )
    filld = fill.get_data()
    filld[filld==1] = 255

    gaussian = np.ones((2,2))*.25

    image_f = np.zeros((256,256,256))

    for slice in range(256):
        temp = filld[:,:,slice]
        image_f[:,:,slice] = convolve(temp, gaussian, 'same')

    image2 = np.zeros((256,256,256))
    image2[np.where(image_f <= 25)] = 0
    image2[np.where(image_f > 25)] = 255

    strel15 = generate_binary_structure(3, 1)

    BW2 = grey_closing(image2, structure=strel15)
    thresh = np.max(BW2)/2
    BW2[np.where(BW2 <= thresh)] = 0
    BW2[np.where(BW2 > thresh)] = 255

    v, f = marching_cubes(BW2, 100)

    v2 = np.transpose(
             np.vstack( ( 128 - v[:,0],
                          v[:,2] - 128,
                          128 - v[:,1], )))
    
    write_surface(outfile, v2, f)

if __name__=='__main__':

    if not len(sys.argv) == 4:
        raise ValueError("the fail at give argument correct,\n"
            "volume filled first, diameter integral then, arguments file "
            "output")

    mkoutersurf( sys.argv[1], None, sys.argv[3] )

    

