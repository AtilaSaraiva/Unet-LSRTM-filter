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

migratedImg = np.zeros((2000,256,256,3))
remigratedImg = np.zeros((2000,256,256,3))

for i in range(1000):
    # Reading the migrated image ( m1 = Ltd )
    file_migratedImg = "database/Mig/rtmlap{i}.rsf"
    migratedImgAux = read_rsf(file_migratedImg)
    migratedImg[ = migratedImgAux[30:286,30:286]

    # Reading the remigrated image ( m2 = LtLm )
    file_remigratedImg = "database/Mig/rtmMigModlap{i}.rsf"
    remigratedImg = read_rsf(file_remigratedImg)
    remigratedImg = remigratedImg[30:286,30:286]

# model = unet(input_size = (*block,1))
# dataset = (remigratedImgReshape, migratedImgReshape)

# inputScaleModel = StandardScaler()
inputScaleModel = RobustScaler()
inputScaleModel.fit(remigratedImg)
norm_remigratedImg = inputScaleModel.transform(remigratedImg)

outputScaleModel = RobustScaler()
outputScaleModel.fit(migratedImg)
norm_migratedImg = outputScaleModel.transform(migratedImg)

inputShape = (256,256)

norm_remigratedImg = norm_remigratedImg.reshape(1,*inputShape,1)
norm_migratedImg        = norm_migratedImg.reshape(1,*inputShape,1)

dataset = (norm_remigratedImg, norm_migratedImg)

model = unet(input_size = (*inputShape,1))
history = model.fit(*dataset, epochs=30)

print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
# plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig("loss.pdf")
plt.show()


# serialize weights to HDF5
model.save_weights("weights/unet_weights.h5")
print("Saved model weights to disk.")

# Save entire model (HDF5)
model.save("weights/unet.h5")
print("Saved model to disk.")

with open("inputScaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(inputScaleModel)
    arq.write(objectBinaryDump)

with open("outputScaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(outputScaleModel)
    arq.write(objectBinaryDump)

# plt.imshow(dataset[0])
# plt.colorbar()
# plt.show()
