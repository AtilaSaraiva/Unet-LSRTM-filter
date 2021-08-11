import numpy as np
from keras.models import *
from os import listdir
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
from unetmodelPaper import unet
from readRSF import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from pickle import dumps, loads

block = (64,64)

# Reading the migrated image ( m1 = Ltd )
file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImg = migratedImg[30:286,30:286]

# Reading the remigrated image ( m2 = LtLm )
file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImg = remigratedImg[30:286,30:286]

# Normaling the remigrated image
remigScaleModel = RobustScaler()
remigScaleModel.fit(remigratedImg)
norm_remigratedImg = remigScaleModel.transform(remigratedImg)

# Normaling the migrated image
migScaleModel = RobustScaler()
migScaleModel.fit(migratedImg)
norm_migratedImg = migScaleModel.transform(migratedImg)

inputShape = (256,256)

norm_remigratedImg = norm_remigratedImg.reshape(1,*inputShape,1)
norm_migratedImg   = norm_migratedImg.reshape(1,*inputShape,1)

dataset = (norm_remigratedImg, norm_migratedImg)

# Testando o melhor lr
lrs = [10**i for i in range(-7,7+1)]
logDir = "logDir"
for lr in lrs:
    model = unet(input_size = (*inputShape,1), learningRate = lr)
    history = model.fit(*dataset, epochs=200)

    tosave = history.history['loss']
    np.savetxt(f'{logDir}/teste_lr_{lr}',tosave)

    # serialize weights to HDF5
    model.save_weights(f"weights/unet_weights_lr_{lr}.h5")
    print("Saved model weights to disk.")

    # Save entire model (HDF5)
    model.save(f"weights/unet_lr_{lr}.h5")
    print("Saved model to disk.")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("results/loss.pdf")

plt.show()



with open("objects/remigScaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(remigScaleModel)
    arq.write(objectBinaryDump)

with open("objects/migScaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(migScaleModel)
    arq.write(objectBinaryDump)
