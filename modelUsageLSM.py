import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from readRSF import *


model = load_model("unet.h5")
block = (64,64)

file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImgReshape = cropArraytoDataset(block, migratedImg)

file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImgReshape = cropArraytoDataset(block, remigratedImg)

idx = 15
prediction = model.predict(migratedImgReshape[idx,None,:,:,0,None])


fig, axes = plt.subplots(2)

axes[0].imshow(prediction[0,:,:,0])
axes[1].imshow(migratedImgReshape[idx,:,:,0])

# plt.colorbar(axes[0],ax=axes)
plt.show()


# plt.imshow(prediction[0,:,:,0])
# plt.show()
# plt.imshow(migratedImgReshape[idx,:,:,0])
# plt.show()
