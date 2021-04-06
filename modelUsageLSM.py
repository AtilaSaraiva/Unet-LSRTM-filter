import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from readRSF import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from pickle import dumps, loads

with open("scaleModel.bin","rb+") as arq:
    objectBinaryDump = arq.read()
    scaleModel = loads(objectBinaryDump)

model = load_model("unet.h5")
# block = (64,64)

file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImg = migratedImg[30:286,30:286]
# migratedImgReshape = cropArraytoDataset(block, migratedImg)

file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImg = remigratedImg[30:286,30:286]
# remigratedImgReshape = cropArraytoDataset(block, remigratedImg)


inputShape = (256,256)
scaleModel = RobustScaler()
scaleModel.fit(migratedImg)
norm_migratedImg = scaleModel.transform(migratedImg)
norm_migratedImg = norm_migratedImg.reshape(1,*inputShape,1)
prediction = model.predict(norm_migratedImg)

# idx = 15
# prediction = model.predict(migratedImgReshape[idx,None,:,:,0,None])



vmax = migratedImg.max()
vmin = migratedImg.min()
print("\nPrediction min = ",prediction.min())

fig, axes = plt.subplots(2,2)

im1 = axes[0,0].imshow(prediction[0,:,:,0], vmin=vmin, vmax=vmax)
axes[1,0].imshow(migratedImg, vmin=vmin, vmax=vmax)
axes[0,1].imshow(norm_migratedImg[0,:,:,0])
axes[1,1].imshow(remigratedImg, vmin=vmin, vmax=vmax)

axes[0,0].title.set_text("Imagem MQ predita pela Unet")
axes[0,1].title.set_text("Imagem migrada com normalização de amplitude")
axes[1,0].title.set_text("Imagem migrada RTM")
axes[1,1].title.set_text("Imagem remigrada RTM")


cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
plt.colorbar(im1, cax=cbaxes)

plt.show()


# plt.imshow(prediction[0,:,:,0])
# plt.show()
# plt.imshow(migratedImgReshape[idx,:,:,0])
# plt.show()
