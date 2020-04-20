import numpy as np
import random
from multiprocessing import Manager, Pool
import time
import os
import cv2 as cv

manager = Manager()
proxyList = manager.list()

ngGlobalList = manager.list()

def drawMatrix(pmat):
    img = np.zeros((670, 670, 3), np.uint8)
    for tmprow in range(6):
        startX = 0
        endX = 0
        pointList = []
        for tmpcol in range(6):
            if pmat[tmprow][tmpcol] == 11:
                startX = tmpcol
                endX = 0
            if pmat[tmprow][tmpcol] == 12:
                endX = tmpcol
                pointList.append((startX * 110 + 10, tmprow * 110 + 10))
                pointList.append(
                    (startX * 110 + 10 + (endX - startX + 1) * 100 + (endX - startX) * 10, tmprow * 110 + 10 + 100))
            if len(pointList) == 2:
                startX = 0
                endX = 0
                cv.rectangle(img, pointList[0], pointList[1], (0, 255, 0), -1)
                pointList.clear()
                pass
            pass
    for tmpcol in range(6):
        startY = 0
        endY = 0
        pointList = []
        for tmprow in range(6):
            if pmat[tmprow][tmpcol] == 21:
                startY = tmprow
                endY = 0
            if pmat[tmprow][tmpcol] == 22:
                endY = tmprow
                pointList.append((tmpcol * 110 + 10, startY * 110 + 10))
                pointList.append(
                    (tmpcol * 110 + 10 + 100, startY * 110 + 10 + (endY - startY + 1) * 100 + (endY - startY) * 10))
            if len(pointList) == 2:
                startY = 0
                endY = 0
                cv.rectangle(img, pointList[0], pointList[1], (255, 0, 0), -1)
                pointList.clear()
                pass
            pass
    for tmpcol in range(6):
        startY = 0
        endY = 0
        pointList = []
        for tmprow in range(6):
            if pmat[tmprow][tmpcol] == 301:
                startY = tmprow
                endY = 0
            if pmat[tmprow][tmpcol] == 302:
                endY = tmprow
                pointList.append((tmpcol * 110 + 10, startY * 110 + 10))
                pointList.append(
                    (tmpcol * 110 + 10 + 100, startY * 110 + 10 + (endY - startY + 1) * 100 + (endY - startY) * 10))
            if len(pointList) == 2:
                startY = 0
                endY = 0
                cv.rectangle(img, pointList[0], pointList[1], (0, 0, 255), -1)
                pointList.clear()
                pass
            pass
    cv.imshow("foo", img)
    cv.waitKey()


# def testFunc(pVal):
#     print("Process Executed", os.getppid())
#     time.sleep(pVal)
#     proxyList.append(pVal)
#     pass
# testArr = np.random.randint(3, 4, size=302)
# pool = Pool(processes=302)
# pool.map(testFunc, testArr)
# pool.close()
# print(proxyList)
# exit()

# get all probable next state
def IterAll(objMatrix, dupList):
    resList = []
    rowRange = [0, 1, 2, 3, 4, 5]
    colRange = [0, 1, 2, 3, 4, 5]
    dRange = [0,1]
    random.shuffle(rowRange)
    random.shuffle(colRange)
    random.shuffle(dRange)
    for rndRow in rowRange:
        for rndCol in colRange:
            for direction in dRange:
                cpMat = np.copy(objMatrix)
                if objMatrix[rndRow][rndCol] == 0:
                    continue
                if objMatrix[rndRow][rndCol] == 1 or \
                        objMatrix[rndRow][rndCol] == 11 or \
                        objMatrix[rndRow][rndCol] == 12:
                    # 0 -> move left , 1 -> move right
                    if direction == 0:
                        if objMatrix[rndRow][rndCol] == 1 or objMatrix[rndRow][rndCol] == 12:
                            tmpleft = 0
                            # find left border
                            for tmpi in range(1, 6):
                                if objMatrix[rndRow][rndCol - tmpi] == 11:
                                    tmpleft = rndCol - tmpi
                                    break
                                pass
                            # make sure left move operation is acceptable
                            if tmpleft > 0 and objMatrix[rndRow][tmpleft - 1] == 0:
                                for tmpi in range(tmpleft - 1, 6):
                                    cpMat[rndRow][tmpi] = objMatrix[rndRow][tmpi + 1]
                                    # move left finish
                                    if cpMat[rndRow][tmpi] == 12:
                                        cpMat[rndRow][tmpi + 1] = 0
                                        if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                            resList.append(cpMat)
                                            pass
                                        break
                                    pass
                                pass
                            pass
                        elif objMatrix[rndRow][rndCol] == 11:
                            if rndCol < 1 or objMatrix[rndRow][rndCol - 1] != 0:
                                # reach left border or left size has value
                                # do nothing
                                pass
                            else:
                                # move left start from left side
                                for tmpi in range(rndCol - 1, 6):
                                    cpMat[rndRow][tmpi] = objMatrix[rndRow][tmpi + 1]
                                    # move left finish
                                    if cpMat[rndRow][tmpi] == 12:
                                        cpMat[rndRow][tmpi + 1] = 0
                                        if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                            resList.append(cpMat)
                                        break
                                    pass
                                pass
                            pass
                        pass
                    else:
                        # move right
                        if objMatrix[rndRow][rndCol] == 1 or objMatrix[rndRow][rndCol] == 11:
                            tmpright = 0
                            for tmpi in range(1, 6):
                                if objMatrix[rndRow][rndCol + tmpi] == 12:
                                    tmpright = rndCol + tmpi
                                    break
                                pass
                            # make sure right move operation is acceptable
                            if tmpright < 5 and objMatrix[rndRow][tmpright + 1] == 0:
                                for tmpi in range(0, 6):
                                    cpMat[rndRow][tmpright + 1 - tmpi] = objMatrix[rndRow][tmpright + 1 - tmpi - 1]
                                    # move right finish
                                    if cpMat[rndRow][tmpright + 1 - tmpi] == 11:
                                        cpMat[rndRow][tmpright + 1 - tmpi - 1] = 0
                                        if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                            resList.append(cpMat)
                                        break
                                    pass
                                pass
                            pass
                        elif objMatrix[rndRow][rndCol] == 12:
                            if rndCol >= 5 or objMatrix[rndRow][rndCol + 1] != 0:
                                # reach right border or right side has value
                                # do nothing
                                pass
                            else:
                                # move left start from left side
                                tmpright = rndCol
                                for tmpi in range(0, 6):
                                    cpMat[rndRow][tmpright + 1 - tmpi] = objMatrix[rndRow][tmpright + 1 - tmpi - 1]
                                    # move right finish
                                    if cpMat[rndRow][tmpright + 1 - tmpi] == 11:
                                        cpMat[rndRow][tmpright + 1 - tmpi - 1] = 0
                                        if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                            resList.append(cpMat)
                                        break
                                    pass
                                pass
                            pass
                        pass
                    pass
                elif objMatrix[rndRow][rndCol] == 2 or \
                        objMatrix[rndRow][rndCol] == 21 or \
                        objMatrix[rndRow][rndCol] == 22:
                    # 0 -> move top , 1 -> move bottom
                    if direction == 0:
                        # add move top code
                        tmptop = rndRow
                        if objMatrix[rndRow][rndCol] == 21:
                            pass
                        else:
                            for tmpi in range(1, 6):
                                if objMatrix[rndRow - tmpi][rndCol] == 21:
                                    tmptop = rndRow - tmpi
                                    break
                                pass
                            pass
                        # ensure move top is acceptable
                        if tmptop > 0 and objMatrix[tmptop - 1][rndCol] == 0:
                            for tmpi in range(tmptop - 1, 6):
                                cpMat[tmpi][rndCol] = objMatrix[tmpi + 1][rndCol]
                                if cpMat[tmpi][rndCol] == 22:
                                    cpMat[tmpi + 1][rndCol] = 0
                                    # move top finish
                                    if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                        resList.append(cpMat)
                                    break
                                pass
                            pass
                        pass
                    else:
                        # add move bottom code
                        tmpbotom = rndRow
                        if objMatrix[rndRow][rndCol] == 22:
                            pass
                        else:
                            for tmpi in range(1, 6):
                                if objMatrix[rndRow + tmpi][rndCol] == 22:
                                    tmpbotom = rndRow + tmpi
                                    break
                                pass
                            pass
                        # ensure move top is acceptable
                        if tmpbotom < 5 and objMatrix[tmpbotom + 1][rndCol] == 0:
                            for tmpi in range(0, 6):
                                cpMat[tmpbotom + 1 - tmpi][rndCol] = objMatrix[tmpbotom + 1 - tmpi - 1][rndCol]
                                if cpMat[tmpbotom + 1 - tmpi][rndCol] == 21:
                                    cpMat[tmpbotom + 1 - tmpi - 1][rndCol] = 0
                                    # move bottom finish
                                    if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                        resList.append(cpMat)
                                    break
                                pass
                            pass
                        pass
                    pass
                elif objMatrix[rndRow][rndCol] == 3 or \
                        objMatrix[rndRow][rndCol] == 301 or \
                        objMatrix[rndRow][rndCol] == 302:
                    # 0 -> move top , 1 -> move bottom
                    if direction == 0:
                        # add move top code
                        tmptop = rndRow
                        if objMatrix[rndRow][rndCol] == 301:
                            pass
                        else:
                            for tmpi in range(1, 6):
                                if objMatrix[rndRow - tmpi][rndCol] == 301:
                                    tmptop = rndRow - tmpi
                                    break
                                pass
                            pass
                        # ensure move top is acceptable
                        if tmptop > 0 and objMatrix[tmptop - 1][rndCol] == 0:
                            for tmpi in range(tmptop - 1, 6):
                                cpMat[tmpi][rndCol] = objMatrix[tmpi + 1][rndCol]
                                if cpMat[tmpi][rndCol] == 302:
                                    cpMat[tmpi + 1][rndCol] = 0
                                    # move top finish
                                    if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                        resList.append(cpMat)
                                    break
                                pass
                            pass
                        pass
                    else:
                        # add move bottom code
                        tmpbotom = rndRow
                        if objMatrix[rndRow][rndCol] == 302:
                            pass
                        else:
                            for tmpi in range(1, 6):
                                if objMatrix[rndRow + tmpi][rndCol] == 302:
                                    tmpbotom = rndRow + tmpi
                                    break
                                pass
                            pass
                        # ensure move top is acceptable
                        if tmpbotom < 5 and objMatrix[tmpbotom + 1][rndCol] == 0:
                            for tmpi in range(0, 6):
                                cpMat[tmpbotom + 1 - tmpi][rndCol] = objMatrix[tmpbotom + 1 - tmpi - 1][rndCol]
                                if cpMat[tmpbotom + 1 - tmpi][rndCol] == 301:
                                    cpMat[tmpbotom + 1 - tmpi - 1][rndCol] = 0
                                    # move bottom finish
                                    if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, resList))) == 0:
                                        resList.append(cpMat)
                                    break
                                pass
                            pass
                        pass
                    pass
    rtList = []
    for tmpitm in resList:
        if len(list(filter(lambda x: np.array_equal(x[0], tmpitm) == True, dupList))) == 0:
            rtList.append(tmpitm)
    return rtList


def RndNext(objMatrix):
    cpMat = None
    while True:
        rndRow = np.random.random_integers(0, 5)
        rndCol = np.random.random_integers(0, 5)
        direction = random.randrange(2)
        cpMat = np.copy(objMatrix)
        if objMatrix[rndRow][rndCol] == 0:
            continue
        if objMatrix[rndRow][rndCol] == 1 or \
                objMatrix[rndRow][rndCol] == 11 or \
                objMatrix[rndRow][rndCol] == 12:
            # 0 -> move left , 1 -> move right
            if direction == 0:
                if objMatrix[rndRow][rndCol] == 1:
                    tmpleft = 0
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow][rndCol - tmpi] == 11:
                            tmpleft = rndCol - tmpi
                            break
                        pass
                    # make sure left move operation is acceptable
                    if tmpleft > 0 and objMatrix[rndRow][tmpleft - 1] == 0:
                        for tmpi in range(tmpleft - 1, 6):
                            cpMat[rndRow][tmpi] = objMatrix[rndRow][tmpi + 1]
                            # move left finish
                            if cpMat[rndRow][tmpi] == 12:
                                cpMat[rndRow][tmpi + 1] = 0
                                break
                            pass
                        pass
                    pass
                elif objMatrix[rndRow][rndCol] == 11:
                    if rndCol < 1 or objMatrix[rndRow][rndCol - 1] != 0:
                        # reach left border or left size has value
                        # do nothing
                        pass
                    else:
                        # move left start from left side
                        for tmpi in range(rndCol - 1, 6):
                            cpMat[rndRow][tmpi] = objMatrix[rndRow][tmpi + 1]
                            # move left finish
                            if cpMat[rndRow][tmpi] == 12:
                                cpMat[rndRow][tmpi + 1] = 0
                                break
                            pass
                        pass
                    pass
                elif objMatrix[rndRow][rndCol] == 12:
                    tmpleft = 0
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow][rndCol - tmpi] == 11:
                            tmpleft = rndCol - tmpi
                            break
                        pass
                    # make sure left move operation is acceptable
                    if tmpleft > 0 and objMatrix[rndRow][tmpleft - 1] == 0:
                        for tmpi in range(tmpleft - 1, 6):
                            cpMat[rndRow][tmpi] = objMatrix[rndRow][tmpi + 1]
                            # move left finish
                            if cpMat[rndRow][tmpi] == 12:
                                cpMat[rndRow][tmpi + 1] = 0
                                break
                            pass
                        pass
                    pass
                pass
            else:
                # move right
                if objMatrix[rndRow][rndCol] == 1:
                    tmpright = 0
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow][rndCol + tmpi] == 12:
                            tmpright = rndCol + tmpi
                            break
                        pass
                    # make sure right move operation is acceptable
                    if tmpright < 5 and objMatrix[rndRow][tmpright + 1] == 0:
                        for tmpi in range(0, 6):
                            cpMat[rndRow][tmpright + 1 - tmpi] = objMatrix[rndRow][tmpright + 1 - tmpi - 1]
                            # move right finish
                            if cpMat[rndRow][tmpright + 1 - tmpi] == 11:
                                cpMat[rndRow][tmpright + 1 - tmpi - 1] = 0
                                break
                            pass
                        pass
                    pass
                elif objMatrix[rndRow][rndCol] == 12:
                    if rndCol >= 5 or objMatrix[rndRow][rndCol + 1] != 0:
                        # reach right border or right side has value
                        # do nothing
                        pass
                    else:
                        # move left start from left side
                        tmpright = rndCol
                        for tmpi in range(0, 6):
                            cpMat[rndRow][tmpright + 1 - tmpi] = objMatrix[rndRow][tmpright + 1 - tmpi - 1]
                            # move right finish
                            if cpMat[rndRow][tmpright + 1 - tmpi] == 11:
                                cpMat[rndRow][tmpright + 1 - tmpi - 1] = 0
                                break
                            pass
                        pass
                    pass
                elif objMatrix[rndRow][rndCol] == 11:
                    tmpright = 0
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow][rndCol + tmpi] == 12:
                            tmpright = rndCol + tmpi
                            break
                        pass
                    # make sure right move operation is acceptable
                    if tmpright < 5 and objMatrix[rndRow][tmpright + 1] == 0:
                        for tmpi in range(0, 6):
                            cpMat[rndRow][tmpright + 1 - tmpi] = objMatrix[rndRow][tmpright + 1 - tmpi - 1]
                            # move right finish
                            if cpMat[rndRow][tmpright + 1 - tmpi] == 11:
                                cpMat[rndRow][tmpright + 1 - tmpi - 1] = 0
                                break
                            pass
                        pass
                    pass
                pass
            pass
        elif objMatrix[rndRow][rndCol] == 2 or \
                objMatrix[rndRow][rndCol] == 21 or \
                objMatrix[rndRow][rndCol] == 22:
            # 0 -> move top , 1 -> move bottom
            if direction == 0:
                # add move top code
                tmptop = rndRow
                if objMatrix[rndRow][rndCol] == 21:
                    pass
                else:
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow - tmpi][rndCol] == 21:
                            tmptop = rndRow - tmpi
                            break
                        pass
                    pass
                # ensure move top is acceptable
                if tmptop > 0 and objMatrix[tmptop - 1][rndCol] == 0:
                    for tmpi in range(tmptop - 1, 6):
                        cpMat[tmpi][rndCol] = objMatrix[tmpi + 1][rndCol]
                        if cpMat[tmpi][rndCol] == 22:
                            cpMat[tmpi + 1][rndCol] = 0
                            # move top finish
                            break
                        pass
                    pass
                pass
            else:
                # add move bottom code
                tmpbotom = rndRow
                if objMatrix[rndRow][rndCol] == 22:
                    pass
                else:
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow + tmpi][rndCol] == 22:
                            tmpbotom = rndRow + tmpi
                            break
                        pass
                    pass
                # ensure move top is acceptable
                if tmpbotom < 5 and objMatrix[tmpbotom + 1][rndCol] == 0:
                    for tmpi in range(0, 6):
                        cpMat[tmpbotom + 1 - tmpi][rndCol] = objMatrix[tmpbotom + 1 - tmpi - 1][rndCol]
                        if cpMat[tmpbotom + 1 - tmpi][rndCol] == 21:
                            cpMat[tmpbotom + 1 - tmpi - 1][rndCol] = 0
                            # move bottom finish
                            break
                        pass
                    pass
                pass
            pass
        elif objMatrix[rndRow][rndCol] == 3 or \
                objMatrix[rndRow][rndCol] == 301 or \
                objMatrix[rndRow][rndCol] == 302:
            # 0 -> move top , 1 -> move bottom
            if direction == 0:
                # add move top code
                tmptop = rndRow
                if objMatrix[rndRow][rndCol] == 301:
                    pass
                else:
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow - tmpi][rndCol] == 301:
                            tmptop = rndRow - tmpi
                            break
                        pass
                    pass
                # ensure move top is acceptable
                if tmptop > 0 and objMatrix[tmptop - 1][rndCol] == 0:
                    for tmpi in range(tmptop - 1, 6):
                        cpMat[tmpi][rndCol] = objMatrix[tmpi + 1][rndCol]
                        if cpMat[tmpi][rndCol] == 302:
                            cpMat[tmpi + 1][rndCol] = 0
                            # move top finish
                            break
                        pass
                    pass
                pass
            else:
                # add move bottom code
                tmpbotom = rndRow
                if objMatrix[rndRow][rndCol] == 302:
                    pass
                else:
                    for tmpi in range(1, 6):
                        if objMatrix[rndRow + tmpi][rndCol] == 302:
                            tmpbotom = rndRow + tmpi
                            break
                        pass
                    pass
                # ensure move top is acceptable
                if tmpbotom < 5 and objMatrix[tmpbotom + 1][rndCol] == 0:
                    for tmpi in range(0, 6):
                        cpMat[tmpbotom + 1 - tmpi][rndCol] = objMatrix[tmpbotom + 1 - tmpi - 1][rndCol]
                        if cpMat[tmpbotom + 1 - tmpi][rndCol] == 301:
                            cpMat[tmpbotom + 1 - tmpi - 1][rndCol] = 0
                            # move bottom finish
                            break
                        pass
                    pass
                pass
            pass

        cmp1 = np.copy(cpMat)
        cmp2 = np.copy(objMatrix)
        cmp1[cmp1 == 11] = 1
        cmp1[cmp1 == 12] = 1
        cmp1[cmp1 == 21] = 2
        cmp1[cmp1 == 22] = 2
        cmp1[cmp1 == 301] = 3
        cmp1[cmp1 == 302] = 3
        cmp2[cmp2 == 11] = 1
        cmp2[cmp2 == 12] = 1
        cmp2[cmp2 == 21] = 2
        cmp2[cmp2 == 22] = 2
        cmp2[cmp2 == 301] = 3
        cmp2[cmp2 == 302] = 3
        tmpres = np.count_nonzero(cmp1 - cmp2)
        if tmpres == 2:
            return cpMat
            # if len(L1list) == 0:
            #     L1list.append(cpMat)
            # else:s
            #     if len(list(filter(lambda x: np.array_equal(x, cpMat) == True, L1list))) > 0:
            #         pass
            #     else:
            #         L1list.append(cpMat)
            #         pass
            #     pass
            # print(len(L1list))
    pass


def GenerateFulltree(pNode):
    L1list = []
    L1list.append(pNode)
    totalCount = 0
    lastSize = len(L1list)

    ngList = list(ngGlobalList)

    while True:
        tmpMat = RndNext(L1list[-1])
        if len(list(filter(lambda x: np.array_equal(x, tmpMat) == True, L1list))) > 0:
            continue
        else:
            L1list.append(tmpMat)

        if tmpMat[:, 3].sum() == 603:
            print("Path Found Start")
            print(L1list)
            print("Path Found End")
            exit()
            pass

        currentSize = len(L1list)

        if currentSize > 1:
            if len(list(filter(lambda x: np.array_equal(x, L1list[-1]) == True, ngList))) > 0:
                break
                pass
            pass

        # start count when list stop increase
        if currentSize == lastSize:
            totalCount += 1
            # tree node dead
            if totalCount > lastSize * 100:
                # print(currentSize)
                # ngList.append((L1list[-2], L1list[-1]))
                # ngList.append(L1list[-1])
                # L1list = []
                # L1list.append(pNode)
                # lastSize = len(L1list)
                proxyList.append(L1list)
                # matrix , fail probability , succeed count , fail count
                ngGlobalList.append((L1list[-1], 1.0, 1, 1))
                break
            pass
        else:
            lastSize = len(L1list)
            totalCount = 0
            pass
        pass
    pass


def SGenerateFulltree(pNode, ngList):
    L1list = []
    L1list.append(pNode)
    totalCount = 0
    lastSize = len(L1list)
    addFlag = False
    while True:
        tmpMat = RndNext(L1list[-1])
        if len(list(filter(lambda x: np.array_equal(x, tmpMat) == True, L1list))) > 0:
            pass
        else:
            L1list.append(tmpMat)
            addFlag = True

        if tmpMat[:, 3].sum() == 603:
            print("Path Found Start")
            print(L1list)
            print("Path Found End")
            exit()
            pass

        currentSize = len(L1list)

        endFlag = False
        for tmpNgItem in ngList:
            if np.array_equal(tmpNgItem[0], L1list[-1]) and tmpNgItem[1] == 1.0:
                # bottom item
                endFlag = True
                break
            pass

        # update parent node when child added
        if addFlag:
            for tmpNgItem in ngList:
                if np.array_equal(tmpNgItem[0], L1list[-2]):
                    if endFlag:
                        tmpNgItem[3] += 1
                    else:
                        tmpNgItem[2] += 1
                    if (tmpNgItem[3] + 1) / (tmpNgItem[2] + 1) > 2:
                        tmpNgItem[1] = 1.0
                        print("dead parent detected", len(ngList))
                    break
                pass

        if endFlag:
            break

        # if currentSize > 1:
        #     if len(list(filter(lambda x: np.array_equal(x, L1list[-1]) == True, ngList))) > 0:
        #         break
        #         pass
        #     pass

        # start count when list stop increase
        if currentSize == lastSize:
            totalCount += 1
            # tree node dead
            if totalCount > lastSize * 100:
                # print(currentSize)
                # ngList.append((L1list[-2], L1list[-1]))
                # ngList.append(L1list[-1])
                # L1list = []
                # L1list.append(pNode)
                # lastSize = len(L1list)
                # matrix , fail probability , succeed count , fail count
                ngList.append([L1list[-1], 1.0, 1, 1])
                ngList.append([L1list[-2], 0, 0, 1])
                break
            pass
        else:
            lastSize = len(L1list)
            totalCount = 0
            pass
        pass
    pass


GridWidth = 6
GridHeight = 6

# 1 means move along x
# 2 means move along y
# 3 means target block
# design a method to move 3 to bottom
startMatrix = np.array([
    [21, 11, 1, 12, 0, 21],
    [22, 0, 0, 301, 0, 22],
    [11, 12, 0, 302, 11, 12],
    [0, 0, 21, 11, 12, 0],
    [0, 0, 2, 11, 1, 12],
    [0, 0, 22, 0, 0, 0]
])

import datetime

totalList = []
totalList.append([startMatrix])
timeList = []
timeList.append(datetime.datetime.now())

foundFlag = False


#deep first search
while True:
    tmplist = IterAll(totalList[-1][0], totalList)
    if len(tmplist) == 0:

        #delete the first parent runs more than 10 seconds
        tmpnow = datetime.datetime.now()
        ii = len(totalList) - 1
        tooLongSearchFoundFlag = False
        while True:
            if (tmpnow - timeList[ii]).seconds >= 10:
                #delete childs behind i
                while len(totalList) > ii + 1:
                    totalList[-1].clear()
                    del totalList[-1]
                    del timeList[-1]
                    tooLongSearchFoundFlag = True
                    pass

                #update parents when parents has no child
                i = len(totalList) - 1
                childDelFlag = False
                while True:
                    if not childDelFlag:
                        del totalList[i][0]
                        childDelFlag = True
                    if i + 1 < len(totalList):
                        if len(totalList[i + 1]) == 0:
                            del totalList[i][0]
                            pass
                        pass
                    i -= 1
                    if i < 0:
                        break
                    pass
                i = len(totalList) - 1
                while True:
                    if len(totalList[i]) == 0:
                        del totalList[i]
                        if len(totalList) == 0:
                            print("Iterate Finish Inside Delete Tree Node")
                            exit()
                        pass
                    i -= 1
                    if i < 0:
                        break
                    pass
                timeList.clear()
                #reset time tree
                for resetTime in range(len(totalList)):
                    timeList.append(datetime.datetime.now())
                    pass
                #end
                break
                pass

            ii -= 1
            if ii < 0:
                break
            pass

        #if not found continue iterate tree
        if not tooLongSearchFoundFlag:
            #delete current node and set parent node to empty when its child is empty
            i = len(totalList) - 1
            childDelFlag = False
            while True:
                if not childDelFlag:
                    del totalList[i][0]
                    childDelFlag = True
                if i + 1 < len(totalList):
                    if len(totalList[i + 1]) == 0:
                        del totalList[i][0]
                        pass
                    pass
                i -= 1
                if i < 0:
                    break
                pass
            #end

            #delete blank items in totalList
            i = len(totalList) - 1
            while True:
                if len(totalList[i]) == 0:
                    del totalList[i]
                    del timeList[-1]
                    if len(totalList) == 0:
                        print("Iterate Finish")
                        exit()
                    pass
                i -= 1
                if i < 0:
                    break
                pass
            #end
        pass
    else:
        for tmpitm in tmplist:
            if tmpitm[:, 3].sum() == 603:
                print("Path Found ", len(totalList))
                foundFlag = True
                pass

        if foundFlag:
            break
        totalList.append(tmplist)
        timeList.append(datetime.datetime.now())
        if len(totalList) % 100 == 0:
            print("100 changed", len(totalList), datetime.datetime.now())
    pass

for tmpmat in totalList:
    drawMatrix(tmpmat[0])
    pass

# while True:
#     tmpMatrix = np.copy(startMatrix)
#     SGenerateFulltree(tmpMatrix, ngArr)
#     # print("NG List Length : ", len(ngArr))
#     pass

# multiple process
# while True:
#     manager = Manager()
#     proxyList = manager.list()
#     processParams = []
#     for i in range(302):
#         processParams.append(np.copy(startMatrix))
#         pass
#     pool = Pool(processes=len(processParams))
#     pool.map(GenerateFulltree, processParams)
#     pool.close()
#     tmplist = list(ngGlobalList)
#     ngGlobalList = manager.list()
#     for tmpNg in tmplist:
#         if len(list(filter(lambda x: np.array_equal(x, tmpNg) == True, ngGlobalList))) > 0:
#             pass
#         else:
#             ngGlobalList.append(tmpNg)
#         pass
#     # print("ngList",len(ngGlobalList))
