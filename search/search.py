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

## problem
## state
## path
## cost
## ready_nodes
## covered_states
def Checker(**node):
    # ready_nodes = node["ready_nodes"] #util.Stack()
    # covered_states = node["covered_states"]

    # TODO: 重构代码
    #     def checker(node):
    # (state, path, cost) = node

    print(node["state"])
    if node["problem"].isGoalState(node["state"]):
        return node["path"]
    else:
        if node["state"] not in node["covered_states"]:
            node["covered_states"].append(node["state"])
            for sub_state, sub_action, sub_cost in node["problem"].getSuccessors(node["state"]):
                node["ready_nodes"].push([sub_state, node["path"] + [sub_action], node["cost"] + sub_cost])
    if not node["ready_nodes"].isEmpty():
        (node["state"], node["path"], node["cost"]) = node["ready_nodes"].pop()
        return Checker(**node)

## problem
## state
## path
## cost
## ready_nodes
## covered_states

# return checker((problem.getStartState(), [], 0))


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

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    ## problem
    ## state
    ## path
    ## cost
    ## ready_nodes
    ## covered_states
    return Checker(problem=problem,
                   state=problem.getStartState(),
                   path=[],
                   cost=0,
                   ready_nodes=util.Stack(),
                   covered_states=[])

    # ready_nodes = util.Stack()
    # covered_states = []
    # def checker(node):
    #     (state, path, cost) = node
    #     if problem.isGoalState(state):
    #         return path
    #     else:
    #         if state not in covered_states:
    #             covered_states.append(state)
    #             for sub_state, sub_action, sub_cost in problem.getSuccessors(state):
    #                 ready_nodes.push([sub_state, path + [sub_action], cost + sub_cost])
    #     if not ready_nodes.isEmpty():
    #         return checker(ready_nodes.pop())
    # return checker((problem.getStartState(), [], 0))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return Checker(problem=problem,
                   state=problem.getStartState(),
                   path=[],
                   cost=0,
                   ready_nodes=util.Queue(),
                   covered_states=[])


    # ready_nodes = util.Queue()
    # covered_states = []
    # def checker(node):
    #     (state, path, cost) = node
    #     if problem.isGoalState(state):
    #         return path
    #     else:
    #         if state not in covered_states:
    #             covered_states.append(state)
    #             for sub_state, sub_action, sub_cost in problem.getSuccessors(state):
    #                 ready_nodes.push([sub_state, path + [sub_action], cost + sub_cost])



    # fringe = util.Queue()
    # start = [problem.getStartState(), 0, []]
    # fringe.push(start)  # queue push at index_0
    # closed = []
    # while not fringe.isEmpty():
    #     [state, cost, path] = fringe.pop()
    #     print("a new path:")
    #     print(path)
    #     if problem.isGoalState(state):
    #         print("final path:")
    #         print(path)
    #         return path
    #     if state not in closed:
    #         closed.append(state)
    #         print("pro path:")
    #         print(problem.getSuccessors(state))
    #         for child_state, child_action, child_cost in problem.getSuccessors(state):
    #             new_cost = cost + child_cost
    #             new_path = path + [child_action]
    #             fringe.push([child_state, new_cost, new_path])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
