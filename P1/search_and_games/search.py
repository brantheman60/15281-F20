# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"

    # remember, no heuristics or path costs!
    explored = [] # list of states
    frontier = util.Queue() # queue of Nodes
    start = problem.getStartState()
    S = Node(start, None, None, 0)
    frontier.push(S)
    
    while True:
        if frontier.isEmpty(): return []
        elem = frontier.pop()

        if problem.goalTest(elem.state):
            return backtrack(problem, elem)

        explored.append(elem.state)
        actions = problem.getActions(elem.state)
        for action in actions:
            child_state = problem.getResult(elem.state, action)
            child = Node(child_state, elem, action, 0)
            # look for node in queue w/ same child state
            sameNode = getFromQueue(frontier, child_state)
            if not sameNode and child_state not in explored:
                frontier.push(child)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    """
    "*** YOUR CODE HERE ***"
    
    depth = 1
    while True:
        goal, limitedByDepth = dls(problem, depth)
        if goal and problem.goalTest(goal.state):
            return backtrack(problem, goal)
        if not limitedByDepth:
            return []
        depth += 1

# depth-limited search
#   return the goal node
#   return False if we already searched the whole graph
def dls(problem, depth):

    limitedByDepth = False
    explored = set() # explored states
    frontier = util.Stack() # Nodes to expand
    S = Node(problem.getStartState(), None, None, 0)
    frontier.push(S)

    while not frontier.isEmpty():
        elem = frontier.pop() # elem is a Node
        if problem.goalTest(elem.state):
            return elem, True

        # do depth checking here
        currentDepth = len(backtrack(problem, elem))
        if currentDepth == depth:
            limitedByDepth = True
            continue

        # add elem to explored
        explored.add(elem.state)

        #printStack(frontier)
        actions = problem.getActions(elem.state)
        for action in actions:
            child_state = problem.getResult(elem.state, action) # returns state
            child = Node(child_state, elem, action, 0)

            # look for node in stack w/ same child state
            sameNode = getFromStack(frontier, child_state)

            if not sameNode and child_state not in explored:
                frontier.push(child)

    if frontier.isEmpty():
        return None, limitedByDepth


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    explored = [] # list of states
    frontier = util.PriorityQueue() # queue of Nodes
    start = problem.getStartState()
    start_f = heuristic(start, problem)
    S = Node(start, None, None, start_f)
    frontier.push(S, start_f)
    
    while True:
        if frontier.isEmpty():
            return []
        elem = frontier.pop()

        if problem.goalTest(elem.state):
            return backtrack(problem, elem)

        explored.append(elem.state)
        actions = problem.getActions(elem.state)
        for action in actions:
            # create child node
            child_state = problem.getResult(elem.state, action)
            # child_actions = backtrack(problem, elem)
            # child_actions.append(action)
            # child_g = problem.getCostOfActions(child_actions)

            child_g = elem.path_cost - heuristic(elem.state, problem) \
                + problem.getCost(elem.state, action)

            child_h = heuristic(child_state, problem)
            child_f = child_g + child_h
            child = Node(child_state, elem, action, child_f)

            # look for node in queue w/ same child state
            sameNode = getFromPriorQueue(frontier, child_state)

            if not sameNode and child_state not in explored:
                frontier.push(child, child_f)
            elif sameNode and sameNode.path_cost > child_f:
                frontier.update(sameNode, 0) # set old node to 0 cost
                frontier.pop() # remove the old node
                frontier.push(child, child_f) # add in new child node

###### HELPER FUNCTIONS ######

# returns list of actions to get from start to elem
def backtrack(problem, elem): # works correctly
    start = problem.getStartState()
    child = copy.copy(elem)
    actions = []
    while child.state != start:
        actions.insert(0, child.action)
        child = child.parent
    return actions

# finds node in stack w/ state
def getFromStack(stack, state):
    ret = None
    nodelist = util.Stack()
    # find node
    while not stack.isEmpty():
        node = stack.pop()
        nodelist.push(node)
        if node.state == state:
            ret = node
            break
    # restore stack
    while not nodelist.isEmpty():
        stack.push( nodelist.pop() )
    return ret

# finds node in queue w/ state
def getFromQueue(queue, state):
    ret = None
    nodelist = []
    # find node
    while not queue.isEmpty():
        node = queue.pop()
        nodelist.append(node)
        if node.state == state:
            ret = node
            break
    # restore queue
    for n in nodelist:
        queue.push(n)
    return ret

# finds node in priority queue w/ state
def getFromPriorQueue(queue, state):
    ret = None
    nodelist = []
    # find node
    while not queue.isEmpty():
        node = queue.pop()
        nodelist.append(node)
        if node.state == state:
            ret = node
            break
    # restore queue
    for n in nodelist:
        queue.push(n, n.path_cost)
    return ret

# print actions of all nodes in queue
def printPriorQueue(queue):
    nodelist = []
    li = []
    # get node list and li
    while not queue.isEmpty():
        node = queue.pop()
        nodelist.append(node)
        li.append(node.action)
    print(li)
    # restore queue
    for n in nodelist:
        queue.push(n, n.path_cost)

# print actions of all nodes in stack
def printStack(stack):
    nodelist = util.Stack()
    li = []
    # get node list and li
    while not stack.isEmpty():
        node = stack.pop()
        nodelist.push(node)
        li.append(node.action)
    
    print(li)
    
    # restore stack
    while not nodelist.isEmpty():
        stack.push( nodelist.pop() )

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
