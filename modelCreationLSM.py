import numpy as np
from keras.models import *
from os import listdir
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
from unetmodel import unet
from readRSF import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from pickle import dumps, loads

block = (64,64)

# Reading the migrated image ( m1 = Ltd )
file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImg = migratedImg[30:286,30:286]
migratedImgReshape = cropArraytoDataset(block, migratedImg)

# Reading the remigrated image ( m2 = LtLm )
file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImg = remigratedImg[30:286,30:286]
remigratedImgReshape = cropArraytoDataset(block, remigratedImg)

# model = unet(input_size = (*block,1))
# dataset = (remigratedImgReshape, migratedImgReshape)

# scaleModel = StandardScaler()
scaleModel = RobustScaler()
scaleModel.fit(remigratedImg)
norm_remigratedImg = scaleModel.transform(remigratedImg)
# print("mean:\n",scaleModel.mean_)
# print("var:\n",scaleModel.var_)

inputShape = (256,256)

norm_remigratedImg = norm_remigratedImg.reshape(1,*inputShape,1)
migratedImg        = migratedImg.reshape(1,*inputShape,1)

dataset = (norm_remigratedImg, migratedImg)


model = unet(input_size = (*inputShape,1))
model.fit(*dataset, epochs=1000)

# serialize weights to HDF5
model.save_weights("unet_weights.h5")
print("Saved model weights to disk.")

# Save entire model (HDF5)
model.save("unet.h5")
print("Saved model to disk.")

with open("scaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(scaleModel)
    arq.write(objectBinaryDump)


# plt.imshow(dataset[0])
# plt.colorbar()
# plt.show()
