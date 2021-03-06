import numpy as np
import random
from multiprocessing import Manager, Pool
import time
import os
import cv2
import datetime

import copy

GridRowCount = 6
GridColCount = 6

VMove = 2
HMove = 1

rndSet = [1, 2, 3, 4]
random.shuffle(rndSet)

# to distinguish target block with others
MagicValue = 7


def DisplayGridState(pGridState):
    img = np.zeros((670, 670, 3), np.uint8)
    for tmpblock in pGridState:
        if tmpblock[1] == HMove:
            cv2.rectangle(img,
                          (tmpblock[0][0, 0] * 110 + 10, tmpblock[0][0, 1] * 110 + 10),
                          (tmpblock[0][0, 0] * 110 + 10 +
                           (tmpblock[0][-1, 0] - tmpblock[0][0, 0] + 1) * 100 +
                           (tmpblock[0][-1, 0] - tmpblock[0][0, 0]) * 10,
                           tmpblock[0][0, 1] * 110 + 10 + 100),
                          (0, 255, 0), -1)
            pass
        else:
            color = (0, 255, 255)
            if tmpblock[2]:
                color = (0, 0, 255)
            # (np.array([[4, 0], [4, 1]]), VMove, False),
            cv2.rectangle(img,
                          (tmpblock[0][0, 0] * 110 + 10, tmpblock[0][0, 1] * 110 + 10),
                          (tmpblock[0][0, 0] * 110 + 10 + 100,
                           tmpblock[0][0, 1] * 110 + 10 +
                           (tmpblock[0][-1, 1] - tmpblock[0][0, 1] + 1) * 100 +
                           (tmpblock[0][-1, 1] - tmpblock[0][0, 1]) * 10),
                          color, -1)
            pass
        pass
    cv2.imshow("MatrixDisplay", img)
    cv2.waitKey()
    pass


def isStateEqual(p1, p2):
    if len(p1) != len(p2) or len(p1) <= 0:
        return False
    for i in range(len(p1)):
        if not np.array_equal(p1[i], p2[i]):
            return False
        pass
    return True


def matrixView(pGridState):
    res = np.zeros((GridRowCount, GridColCount), np.int)
    for gridStateItem in pGridState:
        tmpLabel = gridStateItem[1]
        if gridStateItem[2]:
            tmpLabel += MagicValue
        if gridStateItem[1] == VMove:
            res[:, gridStateItem[0][0, 0]][gridStateItem[0][:, 1]] += tmpLabel
        else:
            res[gridStateItem[0][0, 1], :][gridStateItem[0][:, 0]] += tmpLabel
            pass
        pass
    return res


def ItrAllNextRnd(pGridState):
    matView = matrixView(pGridState)
    iterateOrder = list(range(len(pGridState)))
    random.shuffle(iterateOrder)
    rtList = []
    for idx in iterateOrder:
        target = pGridState[idx]
        if target[1] == VMove:
            # check move up or move down is acceptable
            directions = []
            if target[0][0, 1] > 0 and matView[target[0][0, 1] - 1, target[0][0, 0]] == 0:
                directions.append(-1)
            if target[0][-1, 1] < GridRowCount - 1 and matView[target[0][-1, 1] + 1, target[0][0, 0]] == 0:
                directions.append(1)
            if len(directions) == 2:
                random.shuffle(directions)
            for tmpd in directions:
                resGridState = copy.deepcopy(pGridState)
                resGridState[idx][0][:, 1] += tmpd
                rtList.append(resGridState)
            pass
        else:
            # check move left or move right is acceptable
            directions = []
            if target[0][0, 0] > 0 and matView[target[0][0, 1], target[0][0, 0] - 1] == 0:
                directions.append(-1)
            if target[0][-1, 0] < GridColCount - 1 and matView[target[0][0, 1], target[0][-1, 0] + 1] == 0:
                directions.append(1)
            if len(directions) == 2:
                random.shuffle(directions)
            for tmpd in directions:
                resGridState = copy.deepcopy(pGridState)
                resGridState[idx][0][:, 0] += tmpd
                rtList.append(resGridState)
            pass
        pass
    return rtList


def randMove1Step(pGridState):
    matView = matrixView(pGridState)
    iterateOrder = list(range(len(pGridState)))
    random.shuffle(iterateOrder)
    resGridState = copy.copy(pGridState)
    for idx in iterateOrder:
        target = resGridState[idx]
        if target[1] == VMove:
            # check move up or move down is acceptable
            canMoveUp = False
            camMoveDown = False
            if target[0][0, 1] > 0 and matView[target[0][0, 1] - 1, target[0][0, 0]] == 0:
                canMoveUp = True
                pass
            if target[0][-1, 1] < GridRowCount - 1 and matView[target[0][-1, 1] + 1, target[0][0, 0]] == 0:
                camMoveDown = True
                pass
            if canMoveUp == False and camMoveDown == False:
                # this block is not movable
                pass
            elif canMoveUp == True and camMoveDown == False:
                # move left
                target[0][:, 1] -= 1
                break
            elif canMoveUp == False and camMoveDown == True:
                # move right
                target[0][:, 1] += 1
                break
            else:
                tmpstep = 1
                if random.random() >= 0.5:
                    tmpstep = tmpstep * -1
                    pass
                target[0][:, 1] += tmpstep
                break
            pass
        else:
            # check move left or move right is acceptable
            canMoveLeft = False
            camMoveRight = False
            if target[0][0, 0] > 0 and matView[target[0][0, 1], target[0][0, 0] - 1] == 0:
                canMoveLeft = True
                pass
            if target[0][-1, 0] < GridColCount - 1 and matView[target[0][0, 1], target[0][-1, 0] + 1] == 0:
                camMoveRight = True
                pass
            if canMoveLeft == False and camMoveRight == False:
                # this block is not movable
                pass
            elif canMoveLeft == True and camMoveRight == False:
                # move left
                target[0][:, 0] -= 1
                break
            elif canMoveLeft == False and camMoveRight == True:
                # move right
                target[0][:, 0] += 1
                break
            else:
                tmpstep = 1
                if random.random() >= 0.5:
                    tmpstep = tmpstep * -1
                    pass
                target[0][:, 0] += tmpstep
                break
            pass
        pass
    return resGridState


# dfs search use
def removeTreesBottomsFirstItem(pTree):
    lastIdx = len(pTree) - 1
    childDelFlag = False
    for tmpidx in range(lastIdx, -1, -1):
        if lastIdx == tmpidx:
            pTree[tmpidx][0].clear()
            del pTree[tmpidx][0]
            pass
        if tmpidx + 1 <= lastIdx:
            if len(pTree[tmpidx + 1]) == 0:
                pTree[tmpidx][0].clear()
                del pTree[tmpidx][0]
                pass
            pass
        pass
    for tmpidx in range(lastIdx, -1, -1):
        if len(pTree[tmpidx]) == 0:
            del pTree[tmpidx]
            pass
        pass
    pass


class BFSNode:
    def __init__(self):
        self.currentState = None
        self.previousNode = None
        self.children = None
        self.level = 0
        pass

    pass


def FindSibling(rootNode, searchLevel):
    return None


def FindDuplicate(root, tmpstate):
    if isStateEqual(root.currentState, tmpstate):
        return root

    for child in root.children:
        if isStateEqual(child.currentState, tmpstate):
            return child

        if child.children is not None and len(child.children) > 0:
            tmpres = FindDuplicate(child, tmpstate)
            if tmpres is not None:
                return tmpres

    return None


def FindDuplicate2(duplist, tmpstate):
    for child in duplist:
        if isStateEqual(child.currentState, tmpstate):
            return child
    return None


def FindFromTopToBot(root, searchLevel):
    for child in root.children:
        if child.level < searchLevel:
            if child.children is not None and len(child.children) > 0:
                tmpres = FindFromTopToBot(child, searchLevel)
                if tmpres is not None:
                    return tmpres
            pass
        else:
            if child.children is not None and len(child.children) == 0:
                return child
        pass

    return None

teststart = datetime.datetime.now()

# not finish
def bfsSearch(pNode):
    dplst = []
    baseNode = BFSNode()
    baseNode.currentState = pNode
    baseNode.children = []
    baseNode.level = 0
    dplst.append(baseNode)

    reslist = ItrAllNextRnd(pNode)
    for tmpitem in reslist:
        tmpnode = BFSNode()
        tmpnode.currentState = tmpitem
        tmpnode.children = []
        tmpnode.previousNode = baseNode
        tmpnode.level = baseNode.level + 1
        baseNode.children.append(tmpnode)
        dplst.append(tmpnode)
        pass

    itrIdx = 0
    itrNode = baseNode.children[itrIdx]
    # current filling level
    currentLevel = 1
    while True:
        #runing 120 seconds
        # if (datetime.datetime.now() - teststart).seconds > 120:
        #     print("*Duplicate List Length ", len(dplst))
        #     return

        if itrNode.children is None:
            # dead node
            pass
        elif len(itrNode.children) == 0:
            # fill children
            tmplist = ItrAllNextRnd(itrNode.currentState)
            toaddlist = []
            for tmpstate in tmplist:
                # if target state found return
                # return (itrNode,tmpstate)
                if matrixView(tmpstate)[:, 3].sum() == 18:
                    print("Found ...")
                    return (itrNode, tmpstate)

                # tmpsearchres = FindDuplicate(baseNode, tmpstate)
                tmpsearchres = FindDuplicate2(dplst, tmpstate)
                if tmpsearchres is None:
                    # do not add child inside loop
                    toaddlist.append(tmpstate)
                pass
            if len(toaddlist) > 0:
                for tmpstate in toaddlist:
                    tmpnode = BFSNode()
                    tmpnode.currentState = tmpstate
                    tmpnode.children = []
                    tmpnode.previousNode = itrNode
                    tmpnode.level = itrNode.level + 1
                    itrNode.children.append(tmpnode)
                    dplst.append(tmpnode)
                    if len(dplst) % 100 == 0:
                        print("Duplicate List Reach",len(dplst),datetime.datetime.now())
                        pass
                    # if (datetime.datetime.now() - teststart).seconds > 120:
                    #     print("Duplicate List Length ", len(dplst))
                    #     return
                    pass
            else:
                # this node is dead node
                itrNode.children = None

            # find itrNode sibling node
            itrIdx += 1
            if itrIdx >= len(itrNode.previousNode.children):
                itrNode = FindFromTopToBot(baseNode, currentLevel)
                if itrNode is None:
                    currentLevel += 1
                    print("Current Level ", currentLevel, datetime.datetime.now())
                    itrIdx = 0
                    itrNode = FindFromTopToBot(baseNode, currentLevel)
                    pass
            else:
                itrNode = itrNode.previousNode.children[itrIdx]
            pass
        else:
            # this node is full
            pass
        pass
    pass


gridState = [(np.array([[1, 0], [1, 1], [1, 2]]), VMove, False),
             (np.array([[2, 0], [3, 0]]), HMove, False),
             (np.array([[4, 0], [4, 1]]), VMove, False),
             (np.array([[2, 1], [2, 2]]), VMove, False),
             (np.array([[3, 1], [3, 2]]), VMove, True),
             (np.array([[4, 2], [5, 2]]), HMove, False),

             (np.array([[3, 3], [4, 3], [5, 3]]), HMove, False),
             (np.array([[2, 4], [3, 4]]), HMove, False),
             (np.array([[4, 4], [5, 4]]), HMove, False),
             ]

# simplify
# gridState = [
#     (np.array([[0, 0], [0, 1]]), VMove, False),
#     (np.array([[3, 1], [3, 2]]), VMove, True),
#
#     (np.array([[3, 3], [4, 3], [5, 3]]), HMove, False),
#     (np.array([[2, 4], [3, 4]]), HMove, False),
#     (np.array([[4, 4], [5, 4]]), HMove, False),
# ]

# import cProfile
#
# pr = cProfile.Profile()
# pr.enable()

nd, lst = bfsSearch(gridState)
# bfsSearch(gridState)
# pr.disable()
# pr.print_stats(sort="tottime")
# exit()
dsp = []
dsp.append(lst)
while nd is not None:
    dsp.append(nd.currentState)
    nd = nd.previousNode
    pass
for tmpidx in range(len(dsp) - 1, -1, -1):
    DisplayGridState(dsp[tmpidx])
    pass
exit()


tree = []
tree.append([gridState])
foundFlag = False

print("Iteration Started ", datetime.datetime.now())

while True:
    if (datetime.datetime.now() - teststart).seconds > 120:
        pr.disable()
        pr.print_stats(sort="tottime")
        exit()
    reslist = ItrAllNextRnd(tree[-1][0])

    # remove duplicated node
    delIdx = 0
    delIdxList = []
    for tmpitem in reslist:
        for tmpparent in tree:
            if isStateEqual(tmpitem, tmpparent[0]):
                delIdxList.append(delIdx)
                # find duplicate node break loop
                break
            pass
        delIdx += 1
    if len(delIdxList) > 0:
        for tmpidx in range(len(delIdxList) - 1, -1, -1):
            del reslist[tmpidx]
        pass
    # remove duplicated node finish

    # tree branch reach end
    if len(reslist) == 0:
        removeTreesBottomsFirstItem(tree)
        if len(tree) == 0:
            print("Iteration Finished ", datetime.datetime.now())
            break
    else:
        # tree.append(reslist)

        for tmpitm in reslist:
            if matrixView(tmpitm)[:, 3].sum() == 18:
                tree[-1][0] = tmpitm
                print("Path Found ", len(tree))
                foundFlag = True
                break
        if foundFlag:
            break

        # if len(tree) % 50 == 0:
        #     print("50 Nodes Changed To:", len(tree), datetime.datetime.now())
        #     pass

        # tree node depth less than 100
        if len(tree) >= 10:
            removeTreesBottomsFirstItem(tree)
            if len(tree) == 0:
                print("*Iteration Finished ", datetime.datetime.now())
                break
        else:
            tree.append(reslist)

    pass

for tmplist in tree:
    DisplayGridState(tmplist[0])
    pass
