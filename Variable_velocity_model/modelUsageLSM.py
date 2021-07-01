import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from keras.models import load_model
from readRSF import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from pickle import dumps, loads

# with open("inputScaleModel.bin","rb+") as arq:
    # objectBinaryDump = arq.read()
    # scaleModel = loads(objectBinaryDump)

with open("outputScaleModel.bin","rb+") as arq:
    objectBinaryDump = arq.read()
    scaleModel = loads(objectBinaryDump)

model = load_model("unet.h5")
# block = (64,64)

# Reading migrated Image
file_migratedImg = "database/Mig/rtmlap0.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImg = migratedImg[30:286,30:286]

# Reading remigrated Image (LtL)Ltd
file_remigratedImg = "database/Mig/rtmMigModlap0.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImg = remigratedImg[30:286,30:286]

# plt.imshow(curvFiltImg, cmap='gray')
# plt.show()

inputShape = (256,256)
norm_migratedImg = scaleModel.transform(migratedImg)
norm_migratedImg = norm_migratedImg.reshape(1,*inputShape,1)
prediction = model.predict(norm_migratedImg)
prediction[0,:,:,0] = scaleModel.inverse_transform(prediction[0,:,:,0])


vmax = migratedImg.max()
vmin = migratedImg.min()
print("\nPrediction min = ",prediction.min())



## Comparison image
fig, axes = plt.subplots(2,2)
im1 = axes[0,0].imshow(prediction[0,:,:,0], cmap='gray')#, vmin=vmin, vmax=vmax)
axes[1,0].imshow(migratedImg, cmap='gray')
# axes[0,1].imshow(norm_migratedImg[0,:,:,0])
axes[1,1].imshow(remigratedImg, cmap='gray')

axes[0,0].title.set_text("Imagem MQ predita pela Unet")
axes[1,0].title.set_text("Imagem migrada RTM")
axes[1,1].title.set_text("Imagem remigrada RTM")
# cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
# plt.colorbar(im1, cax=cbaxes)

plt.savefig('generalcomparison.pdf', bbox_inches='tight')
plt.show()
