import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def filterDepth(np2darr):
    res = np.zeros((np2darr.shape[0], np2darr.shape[1]))
    binSum, binAnchor = np.histogram(np2darr.reshape(-1))
    binCount = len(binSum)
    minAnchor = binAnchor[0]
    maxAnchor = binAnchor[-1]
    zeroCount = (np2darr == 0).sum()
    for tmp in range(binCount):
        if tmp < binCount - 1:
            if (binSum[tmp] + binSum[tmp + 1]) / (binSum.sum() - zeroCount) > 0.9:
                minAnchor = binAnchor[tmp]
                maxAnchor = binAnchor[tmp + 1 + 1]

                # check one more bin
                if tmp + 1 + 1 < binCount and binSum[tmp + 1 + 1] > 0:
                    maxAnchor = binAnchor[tmp + 1 + 1 + 1]
                    pass

                pass
            pass
        pass

    # make value bigger than maxAnchor or value smaller than minAnchor = 0
    tmpTop = None
    tmpBot = None
    tmpLeft = None
    tmpRight = None
    print("filter anchor is ", minAnchor, maxAnchor)
    for j in range(res.shape[0]):
        for i in range(res.shape[1]):
            res[j][i] = np2darr[j][i]
            if j - 1 >= 0:
                tmpTop = np2darr[j - 1][i]
            else:
                tmpTop = None

            if j + 1 < res.shape[0]:
                tmpBot = np2darr[j + 1][i]
            else:
                tmpBot = None

            if i - 1 >= 0:
                tmpLeft = np2darr[j][i - 1]
            else:
                tmpLeft = None

            if i + 1 < res.shape[1]:
                tmpRight = np2darr[j][i + 1]
            else:
                tmpRight = None

            if res[j][i] >= minAnchor and res[j][i] <= maxAnchor:
                # do nothing
                pass
            else:
                tmplist = []

                if tmpLeft is not None and tmpLeft >= minAnchor and tmpLeft <= maxAnchor:
                    tmplist.append(tmpLeft)
                if tmpTop is not None and tmpTop >= minAnchor and tmpTop <= maxAnchor:
                    tmplist.append(tmpTop)
                if tmpRight is not None and tmpRight >= minAnchor and tmpRight <= maxAnchor:
                    tmplist.append(tmpRight)
                if tmpBot is not None and tmpBot >= minAnchor and tmpBot <= maxAnchor:
                    tmplist.append(tmpBot)

                #print(tmplist)

                if len(tmplist) >= 3:
                    res[j][i] = sum(tmplist) / len(tmplist)
                else:
                    res[j][i] = 0

                pass
            pass
        pass

    ajust = res.max()
    res = ajust - res
    res[res == ajust] = 0
    #tmpajust2 = res(res != ajust).max()

    return res

df = pd.read_csv("/home/hjd/depthDataFail/4000.txt", sep=" ", header = None)
nparr = df.values
print("Before Filter ", nparr.shape, nparr.dtype)

#nparr = filterDepth(nparr)

print("After Filter ", nparr.shape, nparr.dtype)

#nparr = 800 - nparr

arrsize = nparr.shape

fig = plt.figure(figsize=(10,20))
ax = plt.axes(projection="3d")

def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def z_function2(x, y):
    return np.sqrt(x ** 2 + y ** 2)

x = range(arrsize[1])
y = range(arrsize[0])

X, Y = np.meshgrid(x, y)
Z = nparr

ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.set_zlim(480, 650)

plt.show()