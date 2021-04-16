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
file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImg = migratedImg[30:286,30:286]

# Reading remigrated Image (LtL)Ltd
file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImg = remigratedImg[30:286,30:286]

# Reading curvelet filtered Image
file_curvFiltImg = "madagascarBuild/dadoFiltrado2.rsf"
curvFiltImg = read_rsf(file_curvFiltImg)
curvFiltImg = curvFiltImg [30:286,30:286]

# Reading velocity field
file_velocityField = "madagascarBuild/marmSmooth.rsf"
velocityField = read_rsf(file_velocityField)
velocityField = velocityField [30:286,30:286]


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
axes[0,1].imshow(curvFiltImg, cmap='gray')
axes[1,1].imshow(remigratedImg, cmap='gray')

axes[0,0].title.set_text("Imagem MQ predita pela Unet")
axes[0,1].title.set_text("Imagem MQ com filtro curvelet")
axes[1,0].title.set_text("Imagem migrada RTM")
axes[1,1].title.set_text("Imagem remigrada RTM")
# cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
# plt.colorbar(im1, cax=cbaxes)

plt.savefig('generalcomparison.pdf', bbox_inches='tight')
# plt.show()



## Data plot
fig, axes = plt.subplots(1,3)
axes[0].imshow(velocityField, cmap='jet')
axes[0].title.set_text("Velocity Field")
axes[1].imshow(migratedImg, cmap='gray')
axes[1].title.set_text("$\mathbf{m}_0 = \mathbf{L}^\mathrm{T} \mathbf{d}_{obs}$")
axes[2].imshow(remigratedImg, cmap='gray')
axes[2].title.set_text("$\mathbf{m}_1 = (\mathbf{L}^\mathrm{T}\mathbf{L})\mathbf{L}^\mathrm{T} \mathbf{d}_{obs}$")
plt.savefig('inputdata.pdf', bbox_inches='tight')
plt.show()


## Result plot
fig, axes = plt.subplots(1,3)
axes[0].imshow(migratedImg, cmap='gray')
axes[0].title.set_text("$\mathbf{m}_0 = \mathbf{L}^\mathrm{T} \mathbf{d}_{obs}$")
axes[1].imshow(curvFiltImg, cmap='gray')
axes[1].title.set_text("Curvelet $\mathbf{m}_{LQ} = \mathbf{F}\, \mathbf{m}_0$")
axes[2].imshow(prediction[0,:,:,0], cmap='gray')
axes[2].title.set_text("Unet $\mathbf{m}_{LQ} = \mathrm{Unet}(\mathbf{m}_0$)")
plt.savefig('results.pdf', bbox_inches='tight')
plt.show()
