import math

# a point in r-tree
class Tree_Point:
    def __init__(self, pointInfo):
        self.score = pointInfo[0]
        self.x = pointInfo[1]
        self.y = pointInfo[2]
        self.id = pointInfo[3]
        self.range = [self.x, self.x, self.y, self.y]

    # get point position
    def position(self, index):
        if index == 1:
            return self.x
        elif index == 2:
            return self.y


# r-tree node (extended by leaf and branch)
class Node:
    def __init__(self, Bvalue, num_of_category, level):
        self.childList = []
        self.range = []
        self.centre = []
        self.Bvalue = Bvalue
        self.n_c = num_of_category
        self.paren = None
        self.level = level

        # Associated descriptors
        self.zeta = [1 for __ in range(num_of_category)]
        self.ps = set()

    # add a new child (may be a point or a node) to current node
    def addChild(self, child):
        self.childList.append(child)
        self.update(child)

    # update the cover range of a node when adding a new point or node
    def update(self, child):
        # update x range and y range
        if isinstance(child, Tree_Point):
            self.updateRange([child.x, child.x, child.y, child.y])

        elif isinstance(child, Node):
            self.updateRange(child.range)

        self.updateZeta(child)

        self.updatePs(child)

        # update the centre coordinates
        self.centre[0] = sum(self.range[0:2]) / 2
        self.centre[1] = sum(self.range[2:4]) / 2

    # assistant function of "update" function
    def updateRange(self, newRange):
        # compare and update range
        if newRange[0] < self.range[0]:
            self.range[0] = newRange[0]

        if newRange[1] > self.range[1]:
            self.range[1] = newRange[1]

        if newRange[2] < self.range[2]:
            self.range[2] = newRange[2]

        if newRange[3] > self.range[3]:
            self.range[3] = newRange[3]

    def updateZeta(self, newZeta):
        if isinstance(newZeta, Tree_Point):
            # Leaf node
            self.zeta = [self.zeta[i] * (1-newZeta.score[i]) for i in range(self.n_c)]

        elif isinstance(newZeta, Node):
            # Non-leaf node
            self.zeta = [self.zeta[i] * newZeta.zeta[i] for i in range(self.n_c)]

    def updatePs(self, newP):
        if isinstance(newP, Tree_Point):
            self.ps.add(newP.id)

        elif isinstance(newP, Node):
            self.ps = self.ps | newP.ps

    # return whether the current node is overflow
    def isOverFlow(self):
        if len(self.childList) > self.Bvalue:
            return True
        else:
            return False

    # the distance from a given point to the node centre
    def disToCentre(self, point):
        return ((self.centre[0] - point.x) ** 2 + (self.centre[1] - point.y) ** 2) ** 0.5

    def getIncrease(self, point):
        result = 0
        # increase on x axis
        if point.x > self.range[1]:
            result += point.x - self.range[1]
        elif point.x < self.range[0]:
            result += self.range[0] - point.x
        # increase on y axis
        if point.y > self.range[3]:
            result += point.y - self.range[3]
        elif point.y < self.range[2]:
            result += self.range[2] - point.y

        return result

    # the perimeter of current node
    def getPerimeter(self):
        return self.range[1] - self.range[0] + self.range[3] - self.range[2]

    # split a node, overridden by Leaf and Branch
    def split(self):
        return None


# a leaf node which contains only points
class Leaf(Node):
    def __init__(self, Bvalue, n_c, level, point):
        super().__init__(Bvalue, n_c, level)
        self.range = [point.x, point.x, point.y, point.y]
        self.centre = [point.x, point.y]

    def split(self):
        # sort by x coordinate
        self.childList.sort(key=lambda x: x.position(1))
        nodes = self.getBestSplit()
        periSum = nodes[0].getPerimeter() + nodes[1].getPerimeter()
        # sort by y coordinate
        self.childList.sort(key=lambda x: x.position(2))
        newNodes = self.getBestSplit()
        newSum = newNodes[0].getPerimeter() + newNodes[1].getPerimeter()
        # return the best split
        if newSum < periSum:
            return newNodes
        else:
            return nodes

    '''
    # sort the childList by x if index is 1, by y if index is 2
    def sortChildren(self, index):
        #self.childList.sort(key=lambda x: x.position(index))
        length = len(self.childList)
        for i in range(0, length):
            for j in range(i + 1, length):
                if self.childList[i].position(index) > self.childList[j].position(index):
                    self.childList[i], self.childList[j] = self.childList[j], self.childList[i]
                    #temp = self.childList[i]
                    #self.childList[i] = self.childList[j]
                    #self.childList[j] = temp
    '''

    # get best split based on a sorted children list
    def getBestSplit(self):
        # used to store the minimal sum of perimeters
        periSum = float('inf')
        # used to store the best split
        nodes = []
        b = max(1, math.floor(0.4 * self.Bvalue))

        for i in range(b, len(self.childList) - b + 1):
            # the set of the first i rectangles
            node1 = Leaf(self.Bvalue, self.n_c, 1, self.childList[0])
            node1.paren = self.paren
            # the MBR of the first set
            for j in range(0, i):
                node1.addChild(self.childList[j])
            # the set of the remained rectangles
            node2 = Leaf(self.Bvalue, self.n_c, 1, self.childList[i])
            node2.paren = self.paren
            # the MBR of the second set
            for j in range(i, len(self.childList)):
                node2.addChild(self.childList[j])

            # check whether this is a better split
            newSum = node1.getPerimeter() + node2.getPerimeter()
            if newSum < periSum:
                periSum = newSum
                nodes = [node1, node2]

        # return the best split
        return nodes


# a branch node which contains only nodes
class Branch(Node):
    def __init__(self, Bvalue, n_c, level, node):
        super().__init__(Bvalue, n_c, level)
        self.range = node.range[:]
        self.centre = node.centre[:]

    # choose a child which has a shortest distance from a given point
    def chooseChild(self, point):
        result = None
        increase = None
        for child in self.childList:
            newIncrease = child.disToCentre(point)
            # newIncrease = child.getIncrease(point)
            if increase == None:
                increase = newIncrease
                result = child
            elif increase != 0 and newIncrease / increase > 0.93 and newIncrease / increase < 1.07:
                if len(result.childList) / len(child.childList) > 2:
                    increase = newIncrease
                    result = child
            elif newIncrease < increase:
                increase = newIncrease
                result = child

        return result

    def split(self):
        # sort by xleft and get the sum of perimeter
        self.childList.sort(key=lambda x: x.range[0])
        nodes = self.getBestSplit()
        periSum = nodes[0].getPerimeter() + nodes[1].getPerimeter()
        # sort by xright, ybottom, ytop respectively
        for i in range(1, 4):
            self.childList.sort(key=lambda x: x.range[i])
            newNodes = self.getBestSplit()
            newSum = newNodes[0].getPerimeter() + newNodes[1].getPerimeter()
            # check whether this is a better split
            if newSum < periSum:
                periSum = newSum
                nodes = newNodes

        # set nodes parents and return the best split
        for node in nodes[0].childList:
            node.paren = nodes[0]
        for node in nodes[1].childList:
            node.paren = nodes[1]
        return nodes

    # sort the childList by different elements of self.range
    '''
    def sortChildren(self, index):
        length = len(self.childList)
        for i in range(0, length):
            for j in range(i + 1, length):
                if self.childList[i].range[index] > self.childList[j].range[index]:
                    temp = self.childList[i]
                    self.childList[i] = self.childList[j]
                    self.childList[j] = temp
    '''

    # get best split based on a sorted children list
    def getBestSplit(self):
        # used to store the minimal sum of perimeters
        periSum = float('inf')
        # used to store the best split
        nodes = []
        b = math.floor(0.4 * self.Bvalue)
        for i in range(b, len(self.childList) - b + 1):
            # the set of the first i rectangles
            node1 = Branch(self.Bvalue, self.n_c, self.level, self.childList[0])
            node1.paren = self.paren
            # the MBR of the first set
            for j in range(0, i):
                node1.addChild(self.childList[j])
            # the set of the remained rectangles
            node2 = Branch(self.Bvalue, self.n_c, self.level, self.childList[i])
            node2.paren = self.paren
            # the MBR of the second set
            for j in range(i, len(self.childList)):
                node2.addChild(self.childList[j])
            # check whether this is a better split
            newSum = node1.getPerimeter() + node2.getPerimeter()
            if newSum < periSum:
                periSum = newSum
                nodes = [node1, node2]
        # return the best split
        return nodes