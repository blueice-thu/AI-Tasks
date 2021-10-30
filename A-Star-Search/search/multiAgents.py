from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        we assume ghosts act in turn after the pacman takes an action
        so your minimax tree will have multiple min layers (one for each ghost)
        for every max layer

        gameState.generateChild(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        self.evaluationFunction(state)
        Returns pacman SCORE in current state (useful to evaluate leaf nodes)

        self.depth
        limits your minimax tree depth (note that depth increases one means
        the pacman and all ghosts has already decide their actions)
        """
        from random import randint
        def maxValue(gameState, depth=1):
            if depth > self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            solution = None
            bestValue = float("-inf")
            record = set()  # record values
            selections = gameState.getLegalActions(0)
            for select in selections:
                nextState = gameState.generateChild(0, select)
                newValue = minValue(nextState, 1, depth) - 1
                record.add(newValue)
                if newValue > bestValue:
                    bestValue = newValue
                    solution = select
            if depth == 1:
                if len(record) == 1:  # avoid entering dead cycle
                    return selections[randint(0, len(selections) - 1)]
                return solution
            return bestValue

        def minValue(gameState, ghost, depth):
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            if ghost >= gameState.getNumAgents():
                return maxValue(gameState, depth + 1)
            bestValue = float("inf")
            selections = gameState.getLegalActions(ghost)
            for select in selections:
                nextState = gameState.generateChild(ghost, select)
                newValue = minValue(nextState, ghost + 1, depth)
                bestValue = bestValue if bestValue < newValue else newValue
            return bestValue

        return maxValue(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # TODO
        from random import randint
        def maxValue(gameState, depth=1, alpha=float("-inf"), beta=float("inf")):
            if depth > self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            solution = None
            bestValue = float("-inf")
            selections = gameState.getLegalActions(0)
            record = set()  # record values
            for select in selections:
                nextState = gameState.generateChild(0, select)
                newValue = minValue(nextState, 1, depth, alpha, beta) - 1
                record.add(newValue)
                if newValue > bestValue:
                    bestValue = newValue
                    solution = select
                alpha = alpha if alpha >= bestValue else bestValue
                if beta <= alpha:
                    break
            if depth == 1:
                if len(record) == 1:  # avoid entering dead cycle
                    return selections[randint(0, len(selections) - 1)]
                return solution
            return bestValue

        def minValue(gameState, ghost, depth, alpha=float("-inf"), beta=float("inf")):
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            if ghost >= gameState.getNumAgents():
                return maxValue(gameState, depth + 1, alpha, beta)
            bestValue = float("inf")
            selections = gameState.getLegalActions(ghost)
            for select in selections:
                nextState = gameState.generateChild(ghost, select)
                newValue = minValue(nextState, ghost + 1, depth, alpha, beta)
                bestValue = bestValue if bestValue < newValue else newValue
                beta = beta if beta <= bestValue else bestValue
                if beta <= alpha:
                    break
            return bestValue

        return maxValue(gameState)
