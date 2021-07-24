import nodes
import CONSTANTS
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import math
global root


def handleOverFlow(node):
    global root

    # split node into two new nodes
    clus = node.split()

    #print([x.childList for x in clus])
    #print([x.range for x in clus])

    # if root node is overflow, new root need to build
    if node.paren == None:
        root = nodes.Branch(CONSTANTS.Bvalue, CONSTANTS.num_of_category, node.level + 1, clus[0])
        root.addChild(clus[0])
        root.addChild(clus[1])
        root.childList[0].paren = root
        root.childList[1].paren = root
    else:
        # update the parent node
        parent = node.paren
        parent.childList.remove(node)
        parent.childList += clus
        # check whether parent node is overflow
        if parent.isOverFlow():
            handleOverFlow(parent)


# insert a point to a node
def insert(node, point):
    # if the node is a leaf, add this point
    if isinstance(node, nodes.Leaf):
        node.addChild(point)
        if node.isOverFlow():
            handleOverFlow(node)

    # if the node is a branch, choose a child to add this point
    elif isinstance(node, nodes.Branch):
        node.update(point)
        childNode = node.chooseChild(point)
        insert(childNode, point)

    else:
        pass


# check all nodes and points in a r-tree
def checktree(nodes):
    checkBranch(nodes)
    print('Finished checking HR-tree')


# check the correctness of a leaf node in r-tree
# check the correctness of a leaf node in r-tree
def checkLeaf(leaf):
    # check whether a point is inside of a leaf
    def insideLeaf(x, y, parent):
        if x < parent[0] or x > parent[1] or y < parent[2] or y > parent[3]:
            return False
        else:
            return True

    # general check
    checkNode(leaf)
    # check whether each child point is inside of leaf's range
    for point in leaf.childList:
        if not insideLeaf(point.x, point.y, leaf.range):
            print('point(', point.x, point.y, 'is not in leaf range:', leaf.range)


# check the correctness of a branch node in r-tree
def checkBranch(branch):
    # check whether a branch is inside of another branch
    def insideBranch(child, parent):
        if child[0] < parent[0] or child[1] > parent[1] or child[2] < parent[2] or child[3] > parent[3]:
            return False
        else:
            return True

    # general check
    checkNode(branch)
    # check whether child's range is inside of this node's range
    for child in branch.childList:
        if not insideBranch(child.range, branch.range):
            print('child range:', child.range, 'is not in node range:', branch.range)
        # check this child
        if isinstance(child, nodes.Branch):
            # if child is still a branch node, check recursively
            checkBranch(child)
        elif isinstance(child, nodes.Leaf):
            # if child is a leaf node
            checkLeaf(child)


# general check for both branch and leaf node
def checkNode(node):

    length = len(node.childList)
    # check whether is empty
    if length == 0:
        print('empty node. node level:', node.level, 'node range:', node.range)
    # check whether overflow
    if length > CONSTANTS.Bvalue:
        print('overflow. node level:', node.level, 'node range:', node.range)

    # check whether the centre is really in the centre of the node's range
    r = node.range
    if (r[0] + r[1]) / 2 != node.centre[0] or (r[2] + r[3]) / 2 != node.centre[1]:
        print('wrong centre. node level:', node.level, 'node range:', node.range)
    if r[0] > r[1] or r[2] > r[3]:
        print('wrong range. node level:', node.level, 'node range:', node.range)


def isIntersect(mbr_range, q_p, dist):
    if (mbr_range[0] > q_p[0] and geodesic((q_p[1], q_p[0]), (q_p[1], mbr_range[0])).m > dist) or \
            (mbr_range[1] < q_p[0] and geodesic((q_p[1], mbr_range[1]), (q_p[1], q_p[0])).m > dist) or \
            (q_p[1] < mbr_range[2] and geodesic((q_p[1], q_p[0]), (mbr_range[2], q_p[0])).m > dist) or \
            (q_p[1] > mbr_range[3] and geodesic((mbr_range[3], q_p[0]), (q_p[1], q_p[0])).m > dist):
        return False
    else:
        return True


def range_query(q_p, distance, root, collected, place_feature):
    res = [1 for __ in range(CONSTANTS.num_of_category)]
    q = q_p
    dist = distance
    collected = collected
    place_feature = place_feature

    def __helper(mbr):
        nonlocal res
        # geodesic distance (lat, lng)
        '''
        One example:
                                mbr.range[1], mbr.range[3]
                |-------------------------|
                |                         |
                |                         |
                |                         |
                |            o            |
                |      (q[0], q[1])       |
                |                         |
                |                         |
                |-------------------------|
        mbr.range[0], mbr.range[2]
        '''
        if geodesic((q[1], q[0]), (q[1], mbr.range[0])).m <= dist and \
                geodesic((q[1], q[0]), (q[1], mbr.range[1])).m <= dist and \
                geodesic((q[1], q[0]), (mbr.range[2], q[0])).m <= dist and \
                geodesic((q[1], q[0]), (mbr.range[3], q[0])).m <= dist:
            res = [res[i] * mbr.zeta[i] for i in range(CONSTANTS.num_of_category)]
            if mbr.ps.intersection(collected):
                remove_ps = mbr.ps.intersection(collected)
                remove_res = [1 for _ in range(CONSTANTS.num_of_category)]
                for each_p in remove_ps:
                    p_feature = place_feature[each_p]
                    remove_res = [remove_res[i] * (1 - p_feature[i]) for i in range(CONSTANTS.num_of_category)]
                res = [res[i]/remove_res[i] for i in range(CONSTANTS.num_of_category)]
        else:
            for each_child in mbr.childList:
                # If current mbr is Leaf-node and all its children would be Points
                if isinstance(each_child, nodes.Tree_Point):
                    if geodesic((each_child.y, each_child.x), (q[1], q[0])).m <= dist and \
                            each_child.id not in collected:
                        res = [res[i] * (1 - each_child.score[i]) for i in range(CONSTANTS.num_of_category)]
                else:
                    if isIntersect(each_child.range, q, dist):
                        __helper(each_child)

    __helper(root)

    return [1-i for i in res]


def range_query_ellipse(focus1_east_north, focus2_east_north, distance, root, collected, place_feature):
    """
    :param focus1_east_north: (east, north) utm -- focus point 1
    :param focus2_east_north: (east, north) utm -- focus point 2
    :param distance: Max sum of distance to two foci
    :param root: HR-tree
    :param collected: Collected PoIs
    :param place_feature: Probability distribution of PoIs
    :return:
    """

    if focus1_east_north == focus2_east_north:
        min_x, max_x = focus1_east_north[0] - distance, focus1_east_north[0] + distance
        min_y, max_y = focus1_east_north[1] - distance, focus1_east_north[1] + distance
    else:
        (focus1_x, focus1_y) = focus1_east_north
        (focus2_x, focus2_y) = focus2_east_north

        center_x, center_y = (focus1_x + focus2_x)/2, (focus1_y + focus2_y)/2

        # Notations used AT https://en.wikipedia.org/wiki/Ellipse
        # c -- center to focus
        # a -- semi-major (sum <= 2a) / width
        # b -- semi-minor / height
        c = math.sqrt((focus1_x - center_x) ** 2 + (focus1_y - center_y) ** 2)
        a = distance/2
        b = math.sqrt(a ** 2 - c ** 2)

        # equation of ellipse (phi is the rotated angle):
        #   x(t) = center.x + a*cos(t)*cos(phi) - b*sin(t)*sin(phi)
        #   y(t) = center.y + b*sin(t)*cos(phi) + a*cos(t)*sin(phi)
        cos_phi, sin_phi = (focus2_x - focus1_x) / (2 * c), (focus2_y - focus1_y) / (2 * c)

        # dx/dt = 0 => a * -sin(t) * cos(phi) - b * cos(t) * sin(phi) = 0
        t_x = math.atan((-1 * b * sin_phi) / (a * cos_phi))

        [min_x, max_x] = [center_x + a * math.cos(t) * cos_phi - b * math.sin(t) * sin_phi for t in [t_x, t_x+math.pi]]

        if min_x > max_x:
            max_x, min_x = min_x, max_x

        # dy/dt = 0 => b * cos(t) * cos(phi) - a * sin(t) * sin(phi) = 0
        t_y = math.atan((b * cos_phi) / (a * sin_phi))

        [min_y, max_y] = [center_y + b * math.sin(t) * cos_phi + a * math.cos(t) * sin_phi for t in [t_y, t_y+math.pi]]

        if min_y > max_y:
            max_y, min_y = min_y, max_y

    ############################
    ###### VISUAL_ELLIPSE #####
    ############################
    '''
    fig, ax = plt.subplots()

    ellipse = Ellipse((center_x, center_y), 2*a, 2*b, angle=math.degrees(math.acos(cos_phi)), facecolor='None', edgecolor='b')
    ax.add_artist(ellipse)

    rect = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, facecolor='None', edgecolor='r')
    ax.add_patch(rect)

    ax.scatter(center_x, center_y)
    ax.scatter(focus1_x, focus1_y)
    ax.scatter(focus2_x, focus2_y)

    ax.set_xlim(min_x-100, max_x+100)
    ax.set_ylim(min_y-100, max_y+100)

    plt.show()
    '''

    res = [1 for __ in range(CONSTANTS.num_of_category)]
    collected = collected
    place_feature = place_feature

    def __helper(mbr):
        nonlocal res
        '''
                One example:
                                        mbr.range[1], mbr.range[3]
                        |-------------------------|
                        |                         |
                        |            max_x, max_y |
                        |     |--------------|    |
                        |     |              |    |
                        |     |--------------|    |
                        |  min_x, min_y           |
                        |-------------------------|
                mbr.range[0], mbr.range[2]
                '''

        if min_x >= mbr.range[0] and min_y >= mbr.range[2] and max_x <= mbr.range[1] and max_y <= mbr.range[3]:
            res = [res[i] * mbr.zeta[i] for i in range(CONSTANTS.num_of_category)]
            if mbr.ps.intersection(collected):
                remove_ps = mbr.ps.intersection(collected)
                remove_res = [1 for _ in range(CONSTANTS.num_of_category)]
                for each_p in remove_ps:
                    p_feature = place_feature[each_p]
                    remove_res = [remove_res[i] * (1 - p_feature[i]) for i in range(CONSTANTS.num_of_category)]
                res = [res[i]/remove_res[i] for i in range(CONSTANTS.num_of_category)]
        else:
            for each_child in mbr.childList:
                # If current mbr is Leaf-node and all its children would be Points
                if isinstance(each_child, nodes.Tree_Point):
                    if min_x <= each_child.x <= max_x and min_y <= each_child.y <= max_y and \
                            each_child.id not in collected:
                        res = [res[i] * (1 - each_child.score[i]) for i in range(CONSTANTS.num_of_category)]
                else:
                    if max(each_child.range[0], min_x) <= min(each_child.range[1], max_x) and \
                            max(each_child.range[2], min_y) <= min(each_child.range[3], max_y):
                        __helper(each_child)

    __helper(root)

    return [1-i for i in res]

def main():

    global root

    t_point = nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 1, 2, 'o'])

    root = nodes.Leaf(CONSTANTS.Bvalue, CONSTANTS.num_of_category, 1, t_point)

    root.addChild(t_point)

    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 2, 4, 'a']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 5, 6, 'b']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 9, 5, 'c']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 10, 20, 'd']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 1, 3, 'e']))

    checktree(root)

    print([x.ps for x in root.childList])

    print(range_query((0, 0), 8, root, set('b')))


if __name__ == "__main__":
    main()
