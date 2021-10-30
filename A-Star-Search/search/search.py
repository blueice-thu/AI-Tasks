"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # TODO: Finished
    exploreSet = []
    solution = []

    def dfs(currentState):
        if problem.isGoalState(currentState):
            return solution
        exploreSet.append(currentState)
        expandList = problem.expand(currentState)
        for expand in expandList:
            nextState, nextAction, _ = expand
            if nextState in exploreSet:
                continue
            solution.append(nextAction)
            if dfs(nextState):
                return True
            solution.pop()
        return False

    currentState = problem.getStartState()
    if dfs(currentState):
        return solution
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # TODO: Finished
    currentState = problem.getStartState()
    frontier = util.Queue()
    frontier.push(currentState)
    exploreSet = []
    parent = {}
    parent[currentState] = None
    while not frontier.isEmpty():
        state = frontier.pop()
        if problem.isGoalState(state):
            solution = []
            while parent[state] is not None:
                solution.append(parent[state][1])
                state = parent[state][0]
            solution.reverse()
            return solution
        exploreSet.append(state)
        expandList = problem.expand(state)
        for expand in expandList:
            nextState, nextAction, _ = expand
            if nextState in exploreSet:
                continue
            parent[nextState] = (state, nextAction)
            frontier.push(nextState)
    return []


def nullHeuristic(state, problem=None):
    """
    A example of heuristic function which estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial. You don't need to edit this function
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # TODO: Finished
    exploreSet = []
    frontier = util.PriorityQueue()
    currentState = problem.getStartState()
    frontier.push(currentState, heuristic(currentState, problem))
    parent = {currentState: None}
    pathCost = {currentState: 0}
    while not frontier.isEmpty():
        if frontier.isEmpty():
            return []
        currentState = frontier.pop()
        if problem.isGoalState(currentState):
            solution = []
            while parent[currentState] is not None:
                solution.append(parent[currentState][1])
                currentState = parent[currentState][0]
            solution.reverse()
            return solution
        exploreSet.append(currentState)
        expandList = problem.expand(currentState)
        for expand in expandList:
            nextState, nextAction, nextCost = expand
            if nextState not in exploreSet and nextState not in frontier.heap:
                parent[nextState] = (currentState, nextAction)
                pathCost[nextState] = pathCost[currentState] + nextCost
                frontier.push(nextState, pathCost[nextState] + heuristic(nextState, problem))
            elif nextState in frontier.heap:
                savedCost = pathCost[nextState]
                newCost = pathCost[currentState] + nextCost
                if newCost < savedCost:
                    parent[nextState] = (currentState, nextAction)
                    pathCost[nextState] = newCost
                    frontier.update(nextState, pathCost[nextState] + heuristic(nextState, problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
