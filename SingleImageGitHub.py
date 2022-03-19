# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:31:33 2022

@author: Y. Güray Hatipoglu
"""

### Part 1: Creating .bat file for downloading in time of interest and region of interest
# Greatly benefited from https://github.com/olivierhagolle/peps_download
# PEPS download
# Greatly benefitted from https://techcommunity.microsoft.com/t5/windows-10/bat-file-to-open-cmd-prompt-change-directory-and-execute-python/m-p/2558640
# the answer of Adrian1595


# Critical! : Run this generated "run.bat" file manually from windows
#             running with subprocess etc. won't open prompt window and risky

# peps_download.py should be retrieved from the link above
# peps.txt should include username and password from https://peps.cnes.fr/rocket/#/home
my_dir = "C:/d/" # working directory, to be used throughout the code



lomin = 1.5
lomax = 2
lamin = 43.5
lamax = 44
bbox = [[lomin,lamin],[lomax,lamin],[lomax,lamax],[lomin,lamax]]
myBat = open(my_dir + 'run.bat','w+')

#Below, ##locs## corresponds to my_dir that needs to be written manually
myBat.write('''cd ##locs## 
            python ./peps_download.py -c S2ST --lonmin '''+str(lomin)+''' --lonmax '''+str(lomax)+''' --latmin '''+str(lamin)+''' --latmax '''+str(lamax)+''' -a peps.txt -d 2021-06-28 -f 2021-07-30 --clouds 1 -w C:\d\imagery --windows 
            pause
            ''')
myBat.close()

# This part is to obtain ROI from entered lat lon above

import shapefile

path_to_shp = my_dir + "bbox"

w = shapefile.Writer(path_to_shp)
w.field('id',"1")
w.poly([bbox])
w.record()
w.close()

prj = open(path_to_shp+".prj", "w")
prj.write('''GEOGCS["WGS84",DATUM["WGS_1984",SPHEROID["WGS84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]''')
prj.close()


### Part 2: Unzipping downloaded file and reaching TCI in it.

# Greatly benefited from https://www.code-helper.com/answers/python-zip-extract-directory
# with titled "python extract zip file without directory structure"

import os
import shutil
import zipfile

# Below, ####.zip is the file that was downloaded from PEPS above.
my_zip = my_dir +"####.zip"
file1 = ""
with zipfile.ZipFile(my_zip) as zip_file:
    for member in zip_file.namelist():
        filename = os.path.basename(member)
        # for only obtaining TCI
        if "_TCI" not in filename:
            continue
        # skip directories
        if not filename:
            continue
        if "_TCI" in filename:
            file1 = filename  #automatizes part 3
        # copy file (taken from zipfile's extract)
        source = zip_file.open(member)
        target = open(os.path.join(my_dir, filename), "wb")
        if '_TCI' in member:
            with source, target:
                shutil.copyfileobj(source, target)
                

### Part 3: Cropping the TCI file to 64x64x3 format for further keras
# Benefited from https://datatofish.com/batch-file-from-python/
# Data to Fish
# Benefited from https://gis4programmers.wordpress.com/2017/01/06/using-gdal-to-get-raster-extent/
# GIS4Programmers
# Greatly benefited from https://gis.stackexchange.com/questions/264618/reprojecting-and-saving-shapefile-in-gdal
# xunilk's answer

# Preparing JPEG2000 to GDAL operations via converting it to GeoTIFF

# The following ##locs## needs to be manually changed to my_dir above

myBat2 = open(my_dir + 'conv.bat','w+')
myBat2.write('''cd ##locs## 
            gdal_translate -of GTiff '''+ file1 +''' '''+file1[:-4] + '''.tif 
            ''')
myBat2.close()

import subprocess
subprocess.call(my_dir + 'conv.bat')

file_name = file1[:-4] + ".tif"

## Now the cropping part - Currently, it centers the image and
## go right and bottom for 64 pixels

from osgeo import ogr, osr, gdal
image2 = gdal.Open(my_dir + file_name)

#shapefile with the from projection
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource =   driver.Open(path_to_shp + ".shp", 1)
layer = dataSource.GetLayer()

#set spatial reference and transformation
sourceprj = layer.GetSpatialRef()
targetprj = osr.SpatialReference(wkt = image2.GetProjection())
transform = osr.CoordinateTransformation(sourceprj, targetprj)

to_fill = ogr.GetDriverByName("Esri Shapefile")
ds = to_fill.CreateDataSource(path_to_shp+"rep.shp")
outlayer = ds.CreateLayer('', targetprj, ogr.wkbPolygon)
outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

#apply transformation
i = 0

for feature in layer:
    transformed = feature.GetGeometryRef()
    transformed.Transform(transform)
    geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
    defn = outlayer.GetLayerDefn()
    feat = ogr.Feature(defn)
    feat.SetField('id', i)
    feat.SetGeometry(geom)
    outlayer.CreateFeature(feat)
    i += 1
    feat = None
ds = None

import shapefile

sf = shapefile.Reader(path_to_shp+"rep")
toTsBox = sf.bbox
tox = toTsBox[2] - toTsBox[0]
toy = toTsBox[3] - toTsBox[1]
tox1 = toTsBox[0] + 3*(tox/4)
toy1 = toTsBox[1] + 3*(toy/4)
# Ensuring 64x64 pixels clip via "+640" meters.
# This bbox will be used through all subsequent images for TS analysis for multi version of this pipeline
tox2 = toTsBox[0] + 3*(tox/4) + 640
toy2 = toTsBox[1] + 3*(toy/4) + 640
toBox = (tox1, toy2, tox2, toy1)

gdal.Translate(my_dir + file_name[:-4] + "crop" + ".tif", my_dir + file_name, projWin = toBox)

# Now we should have 64x64x3 RGB image to try with EuroSAT models

### Part 4: Introducing and classifying our image from previous Part 3
# Very greatly benefited from 
# https://github.com/AnushkaMishra29/Eurosat-tensorflow-/blob/master/eurosat.ipynb
# AnushkaMishra29
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Here we load our pre-trained image from "dnn_part.py" file
model = keras.models.load_model(my_dir)

my_image=plt.imread(my_dir + file_name[:-4] + "crop" + ".tif")
plt.imshow(my_image)

probabilities = model.predict(np.array( [my_image,] ))
number_to_class = ['AnnualCrop','Forest',
'HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop',
'Residential','River','SeaLake']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], 
      "-- Probability:", probabilities[0,index[9]])



