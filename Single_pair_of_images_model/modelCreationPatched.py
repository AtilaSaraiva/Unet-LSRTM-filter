import random
import numpy as np
from readRSF import *
from unetmodelPaper import unet
from sklearn.preprocessing import StandardScaler, RobustScaler
from pickle import dumps, loads


patch_num = 1000
patch_size = 64
val_split = 0.2

def extract_patches(data, mask, patch_num, patch_size):

    X = np.empty((patch_num, patch_size, patch_size,1))
    Y = np.empty((patch_num, patch_size, patch_size,1))

    (z_max, x_max) = data.shape

    for n in range(patch_num):

        # Select random point in data (not too close to edge)
        x_n = random.randint(patch_size // 2, x_max - patch_size // 2)
        z_n = random.randint(patch_size // 2, z_max - patch_size // 2)

        # Extract data and mask patch around point
        X[n,:,:,0] = data[z_n-patch_size//2:z_n+patch_size//2,x_n-patch_size//2:x_n+patch_size//2]
        Y[n,:,:,0] = mask[z_n-patch_size//2:z_n+patch_size//2,x_n-patch_size//2:x_n+patch_size//2]


    return X, Y


# Reading the migrated image ( m1 = Ltd )
file_migratedImg = "madagascarBuild/rtmlap.rsf"
migratedImg = read_rsf(file_migratedImg)
migratedImg = migratedImg[30:286,30:286]

# Reading the remigrated image ( m2 = LtLm )
file_remigratedImg = "madagascarBuild/rtmMigModlap.rsf"
remigratedImg = read_rsf(file_remigratedImg)
remigratedImg = remigratedImg[30:286,30:286]

# Normalazing the remigrated image
remigScaleModel = RobustScaler()
remigScaleModel.fit(remigratedImg)
norm_remigratedImg = remigScaleModel.transform(remigratedImg)

# Normalazing the migrated image
migScaleModel = RobustScaler()
migScaleModel.fit(migratedImg)
norm_migratedImg = migScaleModel.transform(migratedImg)


train_num = int(patch_num*(1-val_split))
val_num = int(patch_num*val_split)
X_train, Y_train = extract_patches(norm_remigratedImg, norm_migratedImg, val_num, patch_size)
X_val, Y_val = extract_patches(norm_remigratedImg, norm_migratedImg, val_num, patch_size)

fig, axs = plt.subplots(2, 10, figsize=(15,3))

k = 0
for m in range(10):
  axs[0,m].imshow(X_train[k,:,:,0], interpolation='spline16', cmap=plt.cm.gray, aspect=1)
  axs[0,m].set_xticks([])
  axs[0,m].set_yticks([])
  k += 1

k = 0
for m in range(10):
  axs[1,m].imshow(Y_train[k,:,:,0], interpolation='spline16', cmap=plt.cm.gray, aspect=1)
  axs[1,m].set_xticks([])
  axs[1,m].set_yticks([])
  k += 1

lr = 0.0001
epochs = 10
model = unet(input_size = (patch_size, patch_size,1), learningRate = lr)
history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=10,
                    validation_data=(X_val, Y_val))

# summarize history for loss
plt.plot(history.history['loss'],label=f'lr={lr}')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')

plt.show()

# serialize weights to HDF5
model.save_weights(f"weights/unet_weights_lr_{lr}_epoch_{epochs}.h5")
print("Saved model weights to disk.")

# Save entire model (HDF5)
model.save(f"weights/unet_lr_{lr}_epoch_{epochs}.h5")
print("Saved model to disk.")

with open("objects/remigScaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(remigScaleModel)
    arq.write(objectBinaryDump)

with open("objects/migScaleModel.bin","wb+") as arq:
    objectBinaryDump = dumps(migScaleModel)
    arq.write(objectBinaryDump)
