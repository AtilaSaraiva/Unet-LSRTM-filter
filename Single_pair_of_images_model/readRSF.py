import numpy as np
import matplotlib.pyplot as plt


class my_dictionary(dict):

  # __init__ function
  def __init__(self):
      self = dict()

  # Function to add key:value
  def add(self, key, value):
      self[key] = value

def read_rsf (filename):
    with open(filename) as arq:
        content = arq.readlines()
        targets = ["n1", "n2", "in"]

        param = my_dictionary()
        for line in content:
            for target in targets:
                if target in line:
                    _,value = line.split("=")
                    value = value.rstrip()
                    value = value.strip('"')
                    param.add(target, value)

        nz = int(param["n1"])
        nx = int(param["n2"])
        filename = str(param["in"])
        inputShape = (nz, nx)
        fileArray = np.fromfile(filename, dtype=np.single)
        fileArray = fileArray[0:(nz*nx)].reshape(inputShape, order="F")

        return fileArray

def indexConversion(idx, block, grid, velReshaped, vel):
    blockIdxCol = idx[1] // block[1]
    blockIdxRow = idx[0] // block[0]
    blockIdx = blockIdxRow * grid[1] + blockIdxCol
    blockElemCol = idx[1] % block[1]
    blockElemRow = idx[0] % block[0]
    try:
        velReshaped[blockIdx,
                blockElemRow,
                blockElemCol,
                0] = vel[idx[0],idx[1]]
    except:
        pass

def cropArraytoDataset(block, fileArray):
    """This function takes in a array of a given shape and divide it into blocks, reformating
    the input array into a tensorflow dataset in which the blocks represent the data. The format of the output array will be of (number of blocks, shape of the block, number of channels)"""
    grid = (fileArray.shape[0] // block[0],
            fileArray.shape[1] // block[1])
    numberOfBlocks = grid[0] * grid[1]

    fileArrayReshaped = np.empty((numberOfBlocks, *block, 1))
    for i in range(fileArray.shape[0]):
        for j in range(fileArray.shape[1]):
            indexConversion((i,j), block, grid, fileArrayReshaped, fileArray)

    return fileArrayReshaped




if __name__ == "__main__":
    filename = "madagascarBuild/rtmlap.rsf"
    fileArray = read_rsf(filename)

    block = (40,40)

    fileArrayReshaped = cropArraytoDataset(block, fileArray)

    plt.imshow(fileArrayReshaped[1,:,:,0])
    plt.show()
    plt.imshow(fileArray[0:40,40:80])
    plt.show()
    # plt.imshow(fileArray[30:,:],norm=Normalize)
    # plt.colorbar
    # plt.show()
