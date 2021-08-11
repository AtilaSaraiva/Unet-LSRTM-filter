import matplotlib.pyplot as plt
import numpy as np


lrs = [10**i for i in range(-7,7+1)]
logDir = "logDir"
for lr in lrs:
    history = np.loadtxt(f'{logDir}/teste_lr_{lr}')
    # summarize history for loss
    plt.plot(history,label=f'lr={lr}')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')

plt.savefig("results/loss.pdf")
plt.show()
