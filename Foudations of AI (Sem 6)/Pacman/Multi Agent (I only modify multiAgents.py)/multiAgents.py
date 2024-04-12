# multiAgents.py
# --------------
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

#resources used for this homework:
#cs188 slide 2023 lec 9 - lec 10
#discuss with b09902077

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()  
       
        ghost_pos = successorGameState.getGhostPositions()
        food_list = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()

        if action == 'Stop':
            score = -100
        else:
            score = 0
            
        if len(food_list) != 0:
            #find closest food
            min = 100000000
            for food in food_list:
                dist = manhattanDistance(newPos, food)
                if dist<min:
                    min = dist #the smaller the better
                
            #find the distance of closest ghost from pacman
            min_ghost_dist = 100000000
            for ghost in newGhostStates:
                ghost_pos = ghost.getPosition()
                dist = manhattanDistance(ghost_pos, newPos)
                if dist < min_ghost_dist:
                    min_ghost_dist = dist #the bigger the better
                    closest_ghost = ghost
            
            if closest_ghost.scaredTimer > min_ghost_dist:
                #in this case the smaller the min_ghost_dist, the better
                return 10000 + 1/min_ghost_dist
            
            if closest_ghost.scaredTimer == 0 and min_ghost_dist == 1:
                score -= 1000
                
            score += successorGameState.getScore() + min_ghost_dist/min
            #print(score, currentGameState.getScore(), successorGameState.getScore(), action)
            return score
        else:
            score += successorGameState.getScore()
            #print(score,currentGameState.getScore(), successorGameState.getScore(), action)
            return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
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
    Your minimax agent (question 2)
    """
    def max_action(self, gameState, depth, index):
        legal_actions = gameState.getLegalActions()
        
        max = -1000000000
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)
        
        for action in legal_actions:
            nextState = gameState.generateSuccessor(index, action)
            
            if index+1 != gameState.getNumAgents():
                #is ghost turn
                ret = self.min_action(nextState, depth, index+1)
            
            else:
                if self.depth != depth:
                    #is pacman turn
                    ret = self.max_action(nextState, depth+1, 0)
                else:
                    #at final action
                    ret = self.evaluationFunction(nextState)
                
            if ret > max:
                max = ret
                next_action = action
        
        #print("max",max, depth, index, gameState.getNumAgents())
        if depth != 1:
            return max
        else:
            return next_action
            
            
    def min_action(self, gameState, depth, index):
        legal_actions = gameState.getLegalActions(index)
        
        min = 100000000
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)
        
        for action in legal_actions:
            nextState = gameState.generateSuccessor(index, action)
            
            if index+1 != gameState.getNumAgents():
                #is ghost turn
                ret = self.min_action(nextState, depth, index+1)
            
            else:
                if self.depth != depth:
                    #is pacman turn
                    ret = self.max_action(nextState, depth+1, 0)
                else:
                    ret = self.evaluationFunction(nextState)
            
            if ret < min:
                min = ret
                
        #print("min",min, depth, index, gameState.getNumAgents())
        return min

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        #each depth is equal to pacman moving + all the ghosts moving
        #pacman -> max, ghosts -> min
        return (self.max_action(gameState, 1, 0))
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_action(self, gameState, depth, index, alpha, beta):
        legal_actions = gameState.getLegalActions()
        
        max = -1000000000
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)
        
        for action in legal_actions:
            nextState = gameState.generateSuccessor(index, action)
            
            if index+1 != gameState.getNumAgents():
                #is ghost turn
                ret = self.min_action(nextState, depth, index+1, alpha, beta)
            
            else:
                if self.depth != depth:
                    #is pacman turn
                    ret = self.max_action(nextState, depth+1, 0, alpha, beta)
                else:
                    #at final action
                    ret = self.evaluationFunction(nextState)
                
            if ret > max:
                max = ret
                next_action = action
                
            if ret > beta:
                if depth != 1:
                    return max
                else:
                    return next_action
                
            if ret > alpha:
                alpha = ret
        
        #print("max",max, depth, index, gameState.getNumAgents())
        if depth != 1:
            return max
        else:
            return next_action
        
    def min_action(self, gameState, depth, index, alpha, beta):
        legal_actions = gameState.getLegalActions(index)
        
        min = 100000000
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)
        
        for action in legal_actions:
            nextState = gameState.generateSuccessor(index, action)
            
            if index+1 != gameState.getNumAgents():
                #is ghost turn
                ret = self.min_action(nextState, depth, index+1, alpha, beta)
            
            else:
                if self.depth != depth:
                    #is pacman turn
                    ret = self.max_action(nextState, depth+1, 0, alpha, beta)
                else:
                    ret = self.evaluationFunction(nextState)
            
            if ret < min:
                min = ret
            
            if ret < alpha:
                return ret
            
            if ret < beta:
                beta = ret
                
        #print("min",min, depth, index, gameState.getNumAgents())
        return min
    
    

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #each depth is equal to pacman moving + all the ghosts moving
        #pacman -> max, ghosts -> min
        return (self.max_action(gameState, 1, 0, -100000000, 100000000))
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_action(self, gameState, depth, index):
        legal_actions = gameState.getLegalActions()
        
        max = -1000000000
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)
        
        for action in legal_actions:
            nextState = gameState.generateSuccessor(index, action)
            
            if index+1 != gameState.getNumAgents():
                #is ghost turn
                ret = self.average_action(nextState, depth, index+1)
            
            else:
                if self.depth != depth:
                    #is pacman turn
                    ret = self.max_action(nextState, depth+1, 0)
                else:
                    #at final action
                    ret = self.evaluationFunction(nextState)
                
            if ret > max:
                max = ret
                next_action = action
        
        #print("max",max, depth, index, gameState.getNumAgents())
        if depth != 1:
            return max
        else:
            return next_action
        
    def average_action(self, gameState, depth, index):
        legal_actions = gameState.getLegalActions(index)
        
        ret = 0
        counter = 0
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)
        
        for action in legal_actions:
            counter += 1
            nextState = gameState.generateSuccessor(index, action)
            
            if index+1 != gameState.getNumAgents():
                #is ghost turn
                ret += self.average_action(nextState, depth, index+1)
            
            else:
                if self.depth != depth:
                    #is pacman turn
                    ret += self.max_action(nextState, depth+1, 0)
                else:
                    ret += self.evaluationFunction(nextState)
            
                
        return (ret/counter)
        
        
        
        
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return (self.max_action(gameState, 1, 0))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    get the manhattan distance of closest food from pacman.
    get the manhattan distance of closest ghost from pacman.
    
    if the ghost is scared, automatically give manhattan distance of closest ghost from pacman + 100
    
    calculate the score using currentGameState.getScore() + min_ghost_dist/(min_food_dist*10)
    
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    food_loc = currentGameState.getFood()
    food_pos = food_loc.asList()
    
    # if len(food_loc) == 0:
    #     return currentGameState.getScore()
    
    ghost_states = currentGameState.getGhostStates() 
    
    #get manhattan distance of closest food from pacman pos
    min_food_dist = 100000000
    for food in food_pos:
        ret = manhattanDistance(food, pacman_pos)
        if ret < min_food_dist:
            min_food_dist = ret #smaller the better
            
    #calculate manhattan distance of closest ghost from pacman pos
    min_ghost_dist = 10000000
    for ghost in ghost_states:
        ghost_pos = ghost.getPosition()
        ret = manhattanDistance(ghost_pos, pacman_pos)
        if ret < min_ghost_dist:
            min_ghost_dist = ret #bigger the better
            closest_ghost = ghost
    
    #closest ghost is scared
    if closest_ghost.scaredTimer > 0:
        min_ghost_dist += 100
    
    #print(currentGameState.getScore() + min_ghost_dist/(min_food_dist*10))
    return (currentGameState.getScore() + min_ghost_dist/(min_food_dist*10))
    

# Abbreviation
better = betterEvaluationFunction

