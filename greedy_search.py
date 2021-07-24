import heapq
import itertools
import math

import CONSTANTS
import hrtree
import time
import random

from geopy.distance import geodesic


class Search_Node:
    def __init__(self, node_id, priority, d_to_start, res=None, explored=None, parent=None):
        '''
        :param node_id: ID of node (same in road network)
        :param priority: Priority (Heuristic value) of node
        :param d_to_start: The distance from start to current node
        :param res: The result SET gotten until now
        :param parent: Previous node of current node
        :param explored: Explored nodes/edges along path
        '''
        if not res:
            res = set()
        if not explored:
            explored = set()

        self.id = node_id
        self.priority = priority
        self.d_to_start = d_to_start
        self.parent = parent
        self.res = res
        self.explored = explored

    def get_node(self):
        return self.id

    def print_path(self):
        print("The solution (Start -> End) has been found: ")

        path, path_n = [], self

        while path_n:
            path.append(path_n.get_node())
            path_n = path_n.parent

        path.reverse()

        print("----------------------------")
        print(" -> ".join([str(x) for x in path]))
        print("----------------------------")

        return path

    def __lt__(self, other):
        return (self.priority, -self.d_to_start) < (other.priority, -other.d_to_start)


class Priority_Queue:
    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        # Default order of heapq is ascending
        if order == 'min':
            self.f = f
        elif order == 'max':
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Queue Order is either 'min' or 'max'!")

    def append(self, item):
        heapq.heappush(self.heap, (self.f(item.priority), item))

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Queue has already been empty!")

    def __delitem__(self, key):
        try:
            del self.heap[[item.get_node() == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError("Delete Fail! Cannot find key")

        heapq.heapify(self.heap)

    def __contains__(self, key):
        return any([item.get_node() == key for _, item in self.heap])

    def __getitem__(self, key):
        for val, item in self.heap:
            if item.get_node() == key:
                return val
        raise KeyError(str(key) + " cannot be found in Priority Queue")

    def __len__(self):
        return len(self.heap)


def div_score(res_set, category_num=CONSTANTS.Bvalue):
    '''
    Example:
    :param res_set: [[0.1, 0.2, ..., 0.2], [0.8, 0.7, ..., 0.1], ...]
    :param category_num: Default 6. Only useful if res_set is empty
    :return: Diversity score [0.34, 0.21, ..., 0.11]
    '''

    if not res_set:
        return [0]*category_num
    else:
        category_num = len(res_set[0])

        res = [0] * category_num

        for i in range(category_num):
            tmp_product = 1

            for each_res in res_set:
                tmp_product *= (1 - each_res[i])

            res[i] = 1 - tmp_product

        return res


def swap_res(cur_res, pois, place_feature, k):
    '''
    Example:
    :param cur_res: set(['Park, NY, USA', 'Zoo, NY, USA', ...])
    :param pois: New Sites. set(['Square, NY, USA', ...])
    :param place_feature: Hashtable {'Park, NY, USA': [0.1, 0.2, ...], 'Zoo, NY, USA': [0, 0.8, ...], ...}
    :param k: Limit of result set
    :return: set(['Zoo, NY, USA', 'Square, NY, USA', ...])
    '''
    total_set = cur_res | pois

    res = set()

    # Get all possible combination (k out of k+1)
    subset_idxs = list(itertools.combinations({x for x in range(k+1)}, k))

    for each_poi in total_set:
        if len(res) < k:
            res.add(each_poi)
        else:
            res.add(each_poi)

            res_list = list(res)

            max_score = float('-inf')

            for each_comb in subset_idxs:
                score_list = [place_feature[res_list[idx]] for idx in each_comb]

                if max_score < sum(div_score(score_list)):
                    max_score = sum(div_score(score_list))
                    res = set([ res_list[idx] for idx in each_comb ])

    return res


def adj_priority_calculate(res_list, potential_feature, k):
    '''
    Example
    :param res_list: [[0.1, 0.2, ..., 0.2], [0.8, 0.7, ..., 0.1], ...]
    :param potential_feature: [0.75, 0.5, ...]
    :param k: Limit of result set
    :return: Max possible Div score
    '''
    total_list = [x for x in res_list]
    total_list.append(potential_feature)

    if len(total_list) <= k:
        return sum(div_score(total_list))
    else:
        subset_idxs = list(itertools.combinations({x for x in range(k + 1)}, k))

        max_score = float('-inf')

        for each_comb in subset_idxs:
            score_list = [total_list[idx] for idx in each_comb]

            max_score = max(max_score, sum(div_score(score_list)))

        return max_score


def max_res(new_node, place_feature, cur_res, cur_res_score):
    if cur_res_score < sum(div_score([place_feature[x] for x in new_node.res])):
        cur_res_score = sum(div_score([place_feature[x] for x in new_node.res]))
        cur_res = new_node

    return cur_res, cur_res_score


def greedy_graph_search(root, init_node, dest_node, d_range, ns, es, place_feature, k, verbal_log=True, complexity=True):
    '''
    :param root: HR-tree
    :param init_node: Starting node -- id
    :param dest_node: Destination -- id
    :param d_range: Distance limit
    :param ns: Nodes in road graph {id1: {'lng': -74, 'lat': 41, 'sites':set(['Park, NY, USA', ...])}, ...}
    :param es: Edges in road graph {(id1, id2): 20.1, ...}
    :param place_feature: Hashtable {'Park, NY, USA': [0.1, 0.2, ...], 'Zoo, NY, USA': [0, 0.8, ...], ...}
    :param k: Limit of result set
    :param verbal_log: Print detailed information
    :param complexity: Return time & edge count
    :return:
    '''

    if complexity:
        t_start = time.time()
        edge_count = 0
        time_complexity, edge_complexity = [], []

    # Priority queue waiting for exploration
    frontier = Priority_Queue(order='max')

    # returned results
    res_res, max_res_score = None, float('-inf')

    if ns[init_node]['sites']:
        # node_id, priority, d_to_start, res=None, explored=None, parent=None
        start_node = Search_Node(init_node, 0, 0, res=swap_res(set(), ns[init_node]['sites'], place_feature, k))
        if verbal_log:
            print("Found Site(s)!!!")
        #res_res, max_res_score = max_res(start_node, place_feature, res_res, max_res_score)
    else:
        start_node = Search_Node(init_node, 0, 0)

    frontier.append(start_node)

    # All the visited node
    explored = set()

    while frontier:
        if verbal_log:
            print("Awaiting unexplored nodes", len(frontier))
            print("Current Diversity", max_res_score)

        cur_node = frontier.pop()

        if complexity:
            edge_count += 1

        if cur_node.priority <= max_res_score:
            if verbal_log:
                print("///////////////////////////////////////////")
                print("Stop Earlier! Nothing in queue has greater diversity")
                print(res_res.res)
                res_res.print_path()

            if complexity:
                time_complexity.append((time.time() - t_start, max_res_score))
                edge_complexity.append((edge_count, max_res_score))
                return res_res, max_res_score, time_complexity, edge_complexity
            else:
                return res_res, max_res_score

        explored.add(cur_node.get_node())

        # Get all adjacent nodes of current node
        adj_nodes = set([k[1] for k in es if k[0] == cur_node.get_node()])

        for each_adj_n in adj_nodes:
            # If adjacent node had been explored among this path
            if each_adj_n in explored:
                continue

            used_d = cur_node.d_to_start + es[(cur_node.get_node(), each_adj_n)]

            # If adjacent node exceeds the distance range
            # each_adj_n -> dest_node
            # We use a ellipse to bound the distance limit
            if used_d + math.sqrt((ns[each_adj_n]['east'] - ns[dest_node]['east'])**2 +
                                  (ns[each_adj_n]['north'] - ns[dest_node]['north'])**2) > d_range:
                continue

            # Got the destination
            if each_adj_n == dest_node:
                if ns[each_adj_n]['sites']:
                    final_res = swap_res(cur_node.res, ns[each_adj_n]['sites'], place_feature, k)
                    if verbal_log:
                        print("Found Site(s)!!! and Got destination")
                else:
                    final_res = cur_node.res

                final_div = sum(div_score([place_feature[x] for x in final_res]))

                if final_div >= min(k, CONSTANTS.num_of_category):
                    if verbal_log:
                        print("///////////////////////////////////////////")
                        print("Stop Earlier! Already found the MAX diversity")
                        print(cur_node.res)
                        cur_node.print_path()

                    if complexity:
                        time_complexity.append((time.time() - t_start, final_div))
                        edge_complexity.append((edge_count, final_div))
                        return cur_node, final_div, time_complexity, edge_complexity
                    else:
                        return cur_node, final_div
                elif final_div > max_res_score:
                    res_res, max_res_score = cur_node, final_div
                    if complexity:
                        time_complexity.append((time.time() - t_start, max_res_score))
                        edge_complexity.append((edge_count, max_res_score))

                #continue

            potential_div = hrtree.range_query_ellipse((ns[each_adj_n]['east'], ns[each_adj_n]['north']),
                                                       (ns[dest_node]['east'], ns[dest_node]['north']),
                                                       d_range-used_d,
                                                       root,
                                                       cur_node.res,
                                                       place_feature)

            adj_priority = adj_priority_calculate([place_feature[x] for x in cur_node.res],
                                                  potential_div,
                                                  k)

            if ns[each_adj_n]['sites']:
                adj_res = swap_res(cur_node.res, ns[each_adj_n]['sites'], place_feature, k)
                if verbal_log:
                    print("Found Site(s)!!!")
            else:
                adj_res = cur_node.res

            if verbal_log:
                print("---------------------------------------")
                print("Next direction: ", cur_node.get_node(), '->', each_adj_n,
                      'with potential', potential_div)

            # node_id, priority, d_to_start, res=None, parent=None
            next_node = Search_Node(each_adj_n, adj_priority, used_d, res=adj_res, parent=cur_node)

            if not next_node.get_node() in frontier:
                frontier.append(next_node)
                # res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
            elif next_node.get_node() in frontier:
                if adj_priority > frontier[next_node.get_node()]:
                    del frontier[next_node.get_node()]
                    frontier.append(next_node)
                    # res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
    if verbal_log:
        print("///////////////////////////////////////////")
        print("Explored graph and found the path with MAX diversity", max_res_score)
        print(res_res.res)
        res_res.print_path()
    if complexity:
        time_complexity.append((time.time() - t_start, max_res_score))
        edge_complexity.append((edge_count, max_res_score))
        return res_res, max_res_score, time_complexity, edge_complexity
    else:
        return res_res, max_res_score


def greedy_path_search(root, init_node, dest_node, d_range, ns, es, place_feature, k, verbal_log=True, complexity=True):
    '''
    :param root: HR-tree
    :param init_node: Starting node -- id
    :param dest_node: Destination -- id
    :param d_range: Distance limit
    :param ns: Nodes in road graph {id1: {'lng': -74, 'lat': 41, 'sites':set(['Park, NY, USA', ...])}, ...}
    :param es: Edges in road graph {(id1, id2): 20.1, ...}
    :param place_feature: Hashtable {'Park, NY, USA': [0.1, 0.2, ...], 'Zoo, NY, USA': [0, 0.8, ...], ...}
    :param k: Limit of result set
    :param verbal_log: Print detailed information
    :param complexity: Return time & edge count
    :return:
    '''

    if complexity:
        t_start = time.time()
        edge_count = 0
        time_complexity, edge_complexity = [], []

    # Priority queue waiting for exploration
    frontier = Priority_Queue(order='max')

    # returned results
    res_res, max_res_score = None, float('-inf')

    if ns[init_node]['sites']:
        # node_id, priority, d_to_start, res=None, explored=None, parent=None
        start_node = Search_Node(init_node,
                                 0,
                                 0,
                                 res=swap_res(set(), ns[init_node]['sites'], place_feature, k))
        if verbal_log:
            print("Found Site(s)!!!")
        # res_res, max_res_score = max_res(start_node, place_feature, res_res, max_res_score)
    else:
        start_node = Search_Node(init_node, 0, 0)

    frontier.append(start_node)

    while frontier:
        if verbal_log:
            print("Awaiting unexplored nodes", len(frontier))
            print("Current Diversity", max_res_score)

        cur_node = frontier.pop()

        if complexity:
            edge_count += 1

        if cur_node.priority <= max_res_score:
            if verbal_log:
                print("///////////////////////////////////////////")
                print("Stop Earlier! Nothing in queue has greater diversity")
                print(res_res.res)
                res_res.print_path()

            if complexity:
                time_complexity.append((time.time() - t_start, max_res_score))
                edge_complexity.append((edge_count, max_res_score))
                return res_res, max_res_score, time_complexity, edge_complexity
            else:
                return res_res, max_res_score

        # Get all adjacent nodes of current node
        adj_nodes = set([k[1] for k in es if k[0] == cur_node.get_node()])

        for each_adj_n in adj_nodes:
            if (cur_node.get_node(), each_adj_n) in cur_node.explored:
                continue

            used_d = cur_node.d_to_start + es[(cur_node.get_node(), each_adj_n)]

            # If adjacent node exceeds the distance range
            # We use a ellipse to bound the distance limit
            if used_d + math.sqrt((ns[each_adj_n]['east'] - ns[dest_node]['east']) ** 2 +
                                  (ns[each_adj_n]['north'] - ns[dest_node]['north']) ** 2) > d_range:
                continue

            # Got the destination
            if each_adj_n == dest_node:
                if ns[each_adj_n]['sites']:
                    final_res = swap_res(cur_node.res, ns[each_adj_n]['sites'], place_feature, k)
                    if verbal_log:
                        print("Found Site(s)!!! and Got destination")
                else:
                    final_res = cur_node.res

                final_div = sum(div_score([place_feature[x] for x in final_res]))

                if final_div >= min(k, CONSTANTS.num_of_category):
                    if verbal_log:
                        print("///////////////////////////////////////////")
                        print("Stop Earlier! Already found the MAX diversity")
                        print(cur_node.res)
                        cur_node.print_path()

                    if complexity:
                        time_complexity.append((time.time() - t_start, final_div))
                        edge_complexity.append((edge_count, final_div))
                        return cur_node, final_div, time_complexity, edge_complexity
                    else:
                        return cur_node, final_div
                elif final_div > max_res_score:
                    res_res, max_res_score = cur_node, final_div
                    if complexity:
                        time_complexity.append((time.time() - t_start, max_res_score))
                        edge_complexity.append((edge_count, max_res_score))

                #continue

            potential_div = hrtree.range_query_ellipse((ns[each_adj_n]['east'], ns[each_adj_n]['north']),
                                                       (ns[dest_node]['east'], ns[dest_node]['north']),
                                                       d_range - used_d,
                                                       root,
                                                       cur_node.res,
                                                       place_feature)

            adj_priority = adj_priority_calculate([place_feature[x] for x in cur_node.res],
                                                  potential_div,
                                                  k)

            if ns[each_adj_n]['sites']:
                adj_res = swap_res(cur_node.res, ns[each_adj_n]['sites'], place_feature, k)
                if verbal_log:
                    print("Found Site(s)!!!")
            else:
                adj_res = cur_node.res

            if verbal_log:
                print("---------------------------------------")
                print("Next direction: ", cur_node.get_node(), '->', each_adj_n,
                      'with potential', potential_div)

            adj_explored = set([x for x in cur_node.explored])
            adj_explored.add((cur_node.get_node(), each_adj_n))

            # node_id, priority, d_to_start, res=None, explored=None, parent=None
            next_node = Search_Node(each_adj_n, adj_priority, used_d,
                                    res=adj_res, explored=adj_explored, parent=cur_node)

            if not next_node.get_node() in frontier:
                frontier.append(next_node)
                # res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
            elif next_node.get_node() in frontier:
                if adj_priority > frontier[next_node.get_node()]:
                    del frontier[next_node.get_node()]
                    frontier.append(next_node)
                    # res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
    if verbal_log:
        print("///////////////////////////////////////////")
        print("Explored graph and found the path with MAX diversity", max_res_score)
        print(res_res.res)
        res_res.print_path()
    if complexity:
        time_complexity.append((time.time() - t_start, max_res_score))
        edge_complexity.append((edge_count, max_res_score))
        return res_res, max_res_score, time_complexity, edge_complexity
    else:
        return res_res, max_res_score

#######################################################################################################################
#######################################################################################################################
class baseline_node:
    def __init__(self, node_id, res):
        self.id = node_id
        self.res = res

    def __lt__(self, other):
        return random.random() < 0.5


def dijkstra_alg(init_node, dest_node, d_range, k, ns, es, place_feature):

    t_start = time.time()
    edge_count = 0

    dist = [float('inf')] * len(ns)
    prev = [None] * len(ns)

    dist[init_node] = 0

    frontier = []

    visited = set()

    # node.res store a list of collected PoIs [(time1, edge_count1, set(PoI1, ...), 0.2),
    #                                          (time2, edge_count2, set(PoI1, PoI3, ...)), 0.23]
    # which means I met PoI1 at time1 and PoI3 at time2
    if ns[init_node]['sites']:
        tmp_res = swap_res(set(), ns[init_node]['sites'], place_feature, k)
        tmp_div = sum(div_score([place_feature[x] for x in tmp_res]))
        heapq.heappush(frontier, (dist[init_node], baseline_node(init_node,
                                                                 [(time.time()-t_start, 0, tmp_res, tmp_div)]
                                                                 )))
    else:
        heapq.heappush(frontier, (dist[init_node], baseline_node(init_node,
                                                                 []
                                                                 )))

    while frontier:
        cur_distance, cur_node = heapq.heappop(frontier)

        if cur_distance > d_range:
            return -1, [], [(time.time() - t_start, edge_count, set(), 0)]
        elif cur_node.id == dest_node:
            path = []
            parent_node = cur_node.id
            while prev[parent_node] is not None:
                path.append(parent_node)
                parent_node = prev[parent_node]
            path.append(parent_node)

            path.reverse()

            if cur_node.res:
                final_res = cur_node.res[-1][2]
                final_div = cur_node.res[-1][3]
            else:
                final_res = set()
                final_div = 0

            return cur_distance, path, cur_node.res + [(time.time() - t_start, edge_count, final_res, final_div)]

        visited.add(cur_node.id)

        adj_ns = [each_e[1] for each_e in es if each_e[0] == cur_node.id]

        for each_adj in adj_ns:
            if each_adj in visited: continue

            edge_count += 1

            alt = dist[cur_node.id] + es[(cur_node.id, each_adj)]

            if alt < dist[each_adj]:
                dist[each_adj], prev[each_adj] = alt, cur_node.id
                # Modify priority in queue
                for idx, element in enumerate(frontier):
                    if element[1].id == each_adj:
                        del frontier[idx]
                        heapq.heapify(frontier)
                        break

                cur_record = [x for x in cur_node.res]

                if ns[each_adj]['sites']:
                    if cur_record:
                        cur_poi_set = [x for x in cur_record[-1][2]]
                    else:
                        cur_poi_set = []

                    tmp_res = swap_res(set(cur_poi_set), ns[each_adj]['sites'], place_feature, k)
                    tmp_div = sum(div_score([place_feature[x] for x in tmp_res]))
                    cur_record += [(time.time()-t_start, edge_count, tmp_res, tmp_div)]

                heapq.heappush(frontier, (alt, baseline_node(each_adj, cur_record)))

    return -1, [], [(time.time() - t_start, edge_count, set(), 0)]


def random_walk_restart(init_node, dest_node, d_range, ns, es, place_feature, k, timer, verbal_log=True, complexity=True):
    t_start = time.time()

    if complexity:
        edge_count = 0
        time_complexity, edge_complexity = [], []

    # returned results
    res_path, res_subset, max_res_score = None, set(), float('-inf')

    while time.time() - t_start <= timer:
        # Start a new path
        if verbal_log:
            print("Another new round")

        new_path, k_div_res, d_from_q, cur_node = [init_node], set(), 0, init_node

        if ns[cur_node]['sites'] is not None:
            k_div_res = swap_res(set(), ns[cur_node]['sites'], place_feature, k)

            if verbal_log:
                print("Found sites!!!")

        while d_from_q < d_range:
            adj_ns = [x[1] for x in es if x[0] == cur_node]

            if dest_node in adj_ns:
                if d_from_q + es[(cur_node, dest_node)] > d_range:
                    break

                if verbal_log:
                    print("Got destination!!!")

                if ns[dest_node]['sites']:
                    final_res = swap_res(k_div_res, ns[dest_node]['sites'], place_feature, k)
                    if verbal_log:
                        print("Found Site(s)!!! and Got destination")
                else:
                    final_res = k_div_res

                final_div = sum(div_score([place_feature[x] for x in final_res]))

                if final_div >= min(k, CONSTANTS.num_of_category):
                    if verbal_log:
                        print("///////////////////////////////////////////")
                        print("Stop Earlier! Already found the MAX diversity")
                        print(new_path)

                    if complexity:
                        time_complexity.append((time.time() - t_start, final_div))
                        edge_complexity.append((edge_count, final_div))
                        return new_path, k_div_res, final_div, time_complexity, edge_complexity
                    else:
                        return new_path, k_div_res, final_div,
                elif final_div > max_res_score:
                    max_res_score = final_div
                    res_path = new_path
                    res_subset = k_div_res
                    if complexity:
                        time_complexity.append((time.time() - t_start, max_res_score))
                        edge_complexity.append((edge_count, max_res_score))

                break

            if not adj_ns:
                # Go into dead end
                break

            adj_idx = random.randint(0, len(adj_ns)-1)

            next_node = adj_ns[adj_idx]

            new_path.append(next_node)

            d_from_q += es[(cur_node, next_node)]

            if ns[next_node]['sites'] is not None:
                k_div_res = swap_res(k_div_res, ns[next_node]['sites'], place_feature, k)

                if verbal_log:
                    print("Found sites!!!")

            cur_node = next_node

    if complexity:
        time_complexity.append((time.time() - t_start, max_res_score))
        edge_complexity.append((edge_count, max_res_score))
        return res_path, res_subset, max_res_score, time_complexity, edge_complexity
    else:
        return res_path, res_subset, max_res_score
















