import numpy as np
from keras.models import *
from os import listdir
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
from unetmodel import unet
from readRSF import *
from sklearn.preprocessing import normalize

block = (64,64)

# Reading the migrated image ( m1 = Ltd )
file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImgReshape = cropArraytoDataset(block, migratedImg)

# Reading the remigrated image ( m2 = LtLm )
file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImgReshape = cropArraytoDataset(block, remigratedImg)

model = unet(input_size = (*block,1))

dataset = (remigratedImgReshape, migratedImgReshape)
# dataset = (remigratedImg, migratedImg)
model.fit(*dataset, validation_split = 0.25, epochs=10, batch_size=10)

# serialize weights to HDF5
model.save_weights("unet_weights.h5")
print("Saved model weights to disk.")

# Save entire model (HDF5)
model.save("unet.h5")
print("Saved model to disk.")


plt.imshow(migratedImg[30:286,30:286])
plt.colorbar()
plt.show()
