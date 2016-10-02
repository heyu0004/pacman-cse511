# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions

import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    Food = gameState.getFood()
    self.bound=(Food.width,Food.height)

    legalMoves = gameState.getLegalActions()
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    "Add more of your code here if you want to"
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
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
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    G_Poss=successorGameState.getGhostPositions()
    S_pos=0
    for pos in G_Poss:
        S_pos+=manhattanDistance(newPos,pos)

    distance,x,y=self.FindClosetFood(newPos,newFood)

    return successorGameState.getScore()+(S_pos*0.8-distance)/10

  def FindClosetFood(self,pos,food):
      foodlist=[]
      foodflg=False
      maxr=max(self.bound)

      for r in range(1,maxr-2):
        for i in range(-r,r+1):
            if r==0:
                continue

            x=pos[0]+i
            y1=pos[1]-r
            y2=pos[1]+r

            x,y1=checkBound(self.bound,x,y1)
            x,y2=checkBound(self.bound,x,y2)

            if(food[x][y1]):
                foodlist.append((manhattanDistance(pos,(x,y1)),x,y1))
                foodflg=True

            if(food[x][y2]):
                foodlist.append((manhattanDistance(pos,(x,y2)),x,y2))
                foodflg=True

            y=pos[1]+i
            x1=pos[0]-r
            x2=pos[0]+r

            x1,y=checkBound(self.bound,x1,y)
            x2,y=checkBound(self.bound,x2,y)

            if(food[x1][y]):
                foodlist.append((manhattanDistance(pos,(x1,y)),x1,y))
                foodflg=True

            if(food[x2][y]):
                foodlist.append((manhattanDistance(pos,(x2,y)),x2,y))
                foodflg=True

      if(foodflg):
          dis,x,y=min(foodlist)
          return dis,x,y
      else:
          return 1,0,0

def checkBound(bound,x,y):
    if(x<1):
        x=1

    if(y<1):
        y=1

    if(x>=bound[0]-1):
        x=bound[0]-2

    if(y>=bound[1]-1):
        y=bound[1]-2

    return (x,y)

def scoreEvaluationFunction(currentGameState):
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

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    depth=self.depth
    actions=gameState.getLegalActions(0)
    score=[]

    for action in actions:
        score.append((minfunc(gameState.generatePacmanSuccessor(action),depth,1),action))

    value,action=max(score)

    return action

def maxfunc(gameState,depth):
    if(gameState.isWin() or gameState.isLose()):
        return gameState.getScore()

    actions=gameState.getLegalActions(0)
    actions.remove(Directions.STOP)
    score=[]
    depth-=1;
    if depth>0:
        for action in actions:
            score.append(minfunc(gameState.generatePacmanSuccessor(action),depth,1))
    else:
        for action in actions:
            score.append(gameState.getScore())

    return max(score)

def minfunc(gameState,depth,agentID):
    if(gameState.isWin() or gameState.isLose()):
        return gameState.getScore()

    actions=gameState.getLegalActions(agentID)
    score=[]
    if gameState.getNumAgents()>(agentID+1):
        for action in actions:
            score.append(minfunc(gameState.generateSuccessor(agentID,action),depth,agentID+1))
    else:
        for action in actions:
            score.append(maxfunc(gameState.generateSuccessor(agentID,action),depth))

    return min(score)

#######################################################################

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth=self.depth
        actions=gameState.getLegalActions(0)
        #actions.remove(Directions.STOP)
        score=[]
        maxscore=-9999

        for action in actions:
            score.append((self.betafunc(gameState.generatePacmanSuccessor(action),depth,1,maxscore),action))
            maxscore,act=max(score)

        value,action=max(score)

        #print gameState
        #print score
        #print value
        #sys.exit()

        return action

    def alphafunc(self,gameState,depth,minscore=9999):
        if(gameState.isWin() or gameState.isLose()):
            return gameState.getScore()

        actions=gameState.getLegalActions(0)
        actions.remove(Directions.STOP)
        score=[]
        depth-=1;
        if depth>0:
            nextmaxscore=-9999
            for action in actions:
                value=self.betafunc(gameState.generatePacmanSuccessor(action),depth,1,nextmaxscore)
                if value>minscore:
                    return 9999
                score.append(value)
                nextmaxscore=max(score)
        else:
            for action in actions:
                value=gameState.getScore()
                #value=betterEvaluationFunction(gameState)
                #print value
                if (value>minscore):
                    return 9999
                score.append(value)

        return max(score)

    def betafunc(self,gameState,depth,agentID,maxscore=-9999):
        if(gameState.isWin() or gameState.isLose()):
            #print gameState.getScore()
            return gameState.getScore()

        actions=gameState.getLegalActions(agentID)
        score=[]
        if gameState.getNumAgents()>(agentID+1):
            for action in actions:
                value=self.betafunc(gameState.generateSuccessor(agentID,action),depth,agentID+1)
                if value<maxscore:
                    return -9999
                score.append(value)
        else:
            nextminscore=9999
            for action in actions:
                value=self.alphafunc(gameState.generateSuccessor(agentID,action),depth,nextminscore)
                score.append(value)
                nextminscore=min(score)
        return min(score)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    depth=self.depth
    actions=gameState.getLegalActions(0)
    score=[]

    for action in actions:
        score.append((self.expfunc(gameState.generatePacmanSuccessor(action),depth,1),action))

    value,action=max(score)

    return action

  def maxexpfunc(self,gameState,depth):
      if(gameState.isWin() or gameState.isLose()):
          return self.evaluationFunction(gameState)

      actions=gameState.getLegalActions(0)
      actions.remove(Directions.STOP)
      score=[]
      depth-=1;
      if depth>0:
          for action in actions:
              score.append(self.expfunc(gameState.generatePacmanSuccessor(action),depth,1))
      else:
          for action in actions:
              score.append(self.evaluationFunction(gameState))

      return max(score)

  def expfunc(self,gameState,depth,agentID):
      if(gameState.isWin() or gameState.isLose()):
          return self.evaluationFunction(gameState)

      actions=gameState.getLegalActions(agentID)
      score=[]
      if gameState.getNumAgents()>(agentID+1):
          for action in actions:
              score.append(self.expfunc(gameState.generateSuccessor(agentID,action),depth,agentID+1))
      else:
          for action in actions:
              score.append(self.maxexpfunc(gameState.generateSuccessor(agentID,action),depth))

      return sum(score)/len(score)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"

  PacmanPos = currentGameState.getPacmanPosition()
  Food = currentGameState.getFood()
  Capsules = currentGameState.getCapsules()

  for capusle in Capsules:
      Food[capusle[0]][capusle[1]]=True

  GhostStates = currentGameState.getGhostStates()
  ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

  GhostPoses=currentGameState.getGhostPositions()
  Sumpos=0
  GhostIndex=0
  for GhostPos in GhostPoses:
      distance=manhattanDistance(PacmanPos,GhostPos)
      if distance <2:
          if ScaredTimes[GhostIndex]>0:
              Sumpos-=1/(distance+1)
          else:
              Sumpos+=1/(distance+1)
      GhostIndex+=1

  if not (currentGameState.isWin() or currentGameState.isLose()):
      distance,x,y=FindClosetFood(PacmanPos,Food)
  else:
      return currentGameState.getScore()

  p=1.0/distance
  n=Sumpos*8

      #print
  #print "this turn p="+str(p)+" and n ="+str(n)+" distance="+str(distance)

  return currentGameState.getScore()+(p-n)*8

def FindClosetFood(pos,food):
    foodlist=[]
    foodflg=False
    bound=(food.width,food.height)
    maxr=max(bound)
    #foodtemp=food.deepCopy()
    #for i in range(bound[0]):
        #for j in range(bound[1]):
            #foodtemp[i][j]=False


    for r in range(1,maxr-2):
        for i in range(-r,r+1):
            if r==0:
                continue

            x=pos[0]+i
            y1=pos[1]-r
            y2=pos[1]+r

            x,y1=checkBound(bound,x,y1)
            x,y2=checkBound(bound,x,y2)

            if(food[x][y1]):
                foodlist.append((manhattanDistance(pos,(x,y1)),x,y1))
                foodflg=True

            if(food[x][y2]):
                foodlist.append((manhattanDistance(pos,(x,y2)),x,y2))
                foodflg=True

            #foodtemp[x][y1]=True
            #foodtemp[x][y2]=True

            y=pos[1]+i
            x1=pos[0]-r
            x2=pos[0]+r

            x1,y=checkBound(bound,x1,y)
            x2,y=checkBound(bound,x2,y)

            if(food[x1][y]):
                foodlist.append((manhattanDistance(pos,(x1,y)),x1,y))
                foodflg=True

            if(food[x2][y]):
                foodlist.append((manhattanDistance(pos,(x2,y)),x2,y))
                foodflg=True

            #foodtemp[x1][y]=True
            #foodtemp[x2][y]=True

        if(foodflg):
            dis,x,y=min(foodlist)
            return dis,x,y

    if(not foodflg):
        print "no food around"
        print foodtemp
        print "real*********************"
        print food
        sys.exit()
        return 1,0,0

# Abbreviation
better = betterEvaluationFunction

#####################################################################
#
#                         Contest
#
#####################################################################

class ContestAgent(MultiAgentSearchAgent):

    def registerInitialState(self, state):
        self.Walls=state.getWalls()
        self.Foods=state.getFood()
        self.Bounds=[self.Foods.width,self.Foods.height]
        self.counter=0
        self.lastPaction=Directions.EAST
        self.depth=4
        self.lastGaction=[Directions.STOP for i in range(3)]
        self.lastGpos=[(0,0) for i in range(3)]
        self.ChasingGhostflg=True

    def getAction(self, gameState):
        actions=gameState.getLegalActions(0)
        actions.remove(Directions.STOP)
        scores=[]
        maxscore=-9999

        self.counter+=1
        if (self.counter==10):
            self.depth=5
        if (self.counter==38):
            self.depth=4
        if (self.counter==70):
            self.depth=3

        if(self.counter==25):
            self.ChasingGhostflg=False

        GhostStates = gameState.getGhostStates()
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        self.foodcounter=gameState.getFood().count()
        self.Capcounter=len(gameState.getCapsules())

        ScaredTimes.remove(min(ScaredTimes))
        if min(ScaredTimes)<=self.depth+5:
            self.eatPacflg=True
        else:
            self.eatPacflg=False

        if self.lastPaction in actions:
            index=actions.index(self.lastPaction)
            temp=actions[0]
            actions[0]=self.lastPaction
            actions[index]=temp

        invlastPaction=self.InvAction(self.lastPaction)

        currentGpos=gameState.getGhostPositions()
        for GhostId in range(3):
            if (self.counter>1):
                self.lastGaction[GhostId]=self.Vector2Action(self.lastGpos[GhostId],currentGpos[GhostId])
            self.lastGpos[GhostId]=currentGpos[GhostId]

        for action in actions:
            self.action=action
            self.caprewarded=False
            currentGameState=gameState.generatePacmanSuccessor(action)

            #
            states=[]
            reward=0
            caprewarded=False
            x,y=currentGameState.getPacmanPosition()
            Food = currentGameState.getFood()
            capsulecounter=len(currentGameState.getCapsules())
            if(Food[x][y]):
                reward+=27
            else:
                if(capsulecounter<self.Capcounter and self.eatPacflg):
                    reward+=150
                    caprewarded=True
                reward*=1.1
            score=self.betafunc(currentGameState,self.depth,1,reward,maxscore,caprewarded,action,self.lastGaction)

            if action==invlastPaction and len(actions)>1:
                score-=10

            scores.append((score,action))
            maxscore,bestaction=max(scores)

        #print score
        #print "#########################"
        #for state in states:
            #print states[0]
        #self.lastaction=bestaction
        return bestaction

    def alphafunc(self,gameState,depth,reward,minscore,caprewarded,lastPacaction,lastGaction):
        if gameState.isLose():
            return -8999-depth

        actions=gameState.getLegalActions(0)
        actions.remove(Directions.STOP)
        score=[]
        depth-=1;
        maxvalue=-9999;
        bestaction=0;

        if lastPacaction in actions:
            index=actions.index(lastPacaction)
            temp=actions[0]
            actions[0]=lastPacaction
            actions[index]=temp

        for action in actions:
            currentGameState=gameState.generatePacmanSuccessor(action)
            x,y=currentGameState.getPacmanPosition()
            Food = currentGameState.getFood()
            capsulecounter=len(currentGameState.getCapsules())

            if(Food[x][y]):
                reward+=27
            else:
                if(self.Capcounter>capsulecounter) and self.eatPacflg and (not caprewarded):
                    reward+=150
                    caprewarded=True
            value=self.betafunc(currentGameState,depth,1,reward*1.1,maxvalue,caprewarded,action,lastGaction)

            if value>minscore:
                bestaction=action
                return 9000
            if value>maxvalue:
                maxvalue=value

        if(depth==self.depth-1):
            self.lastPaction=bestaction

        return maxvalue

    def betafunc(self,gameState,depth,agentID,reward,maxscore,caprewarded,lastPacaction,lastGaction):
        if gameState.isWin():
            return 9000+depth
        if gameState.isLose():
            return -8999

        actions=gameState.getLegalActions(agentID)
        if len(actions)>1:
            InvAct=self.InvAction(lastGaction[agentID-1])
            if InvAct in actions:
                actions.remove(InvAct)

        score=[]
        minvalue=9999

        if gameState.getNumAgents()>(agentID+1):
            for action in actions:
                value=self.betafunc(gameState.generateSuccessor(agentID,action),depth,agentID+1,reward,-9999,caprewarded,lastPacaction,lastGaction)
                if (value<maxscore):
                    return -9000
                if value<minvalue:
                    minvalue=value
        else:
            if depth>1:
                for action in actions:
                    value=self.alphafunc(gameState.generateSuccessor(agentID,action),depth,reward,minvalue,caprewarded,lastPacaction,lastGaction)
                    if value<minvalue:
                        minvalue=value
            else:
                for action in actions:
                    value=self.evaluateFun(gameState,reward,caprewarded)
                    if value<minvalue:
                        minvalue=value

        return minvalue

    def evaluateFun(self,currentGameState,reward,caprewarded):
        if currentGameState.isWin() :
            return currentGameState.getScore()+10
        if currentGameState.isLose():
            return -8999

        Food = currentGameState.getFood()
        x,y = currentGameState.getPacmanPosition()
        GhostPoses=currentGameState.getGhostPositions()
        capsulecounter=len(currentGameState.getCapsules())
        P_FDis=0.5

        if(Food[x][y]):
            reward+=27

        #
        GhostStates=currentGameState.getGhostStates()
        GhostPoses=[ghoststate.getPosition() for ghoststate in GhostStates if ghoststate.scaredTimer>0]
        for ghostpos in GhostPoses:
            if ghostpos[0]>int(ghostpos[0]):
                GhostPoses.append((ghostpos[0]+0.5,ghostpos[1]))
                GhostPoses.append((ghostpos[0]-0.5,ghostpos[1]))
            elif ghostpos[1]>int(ghostpos[1]):
                GhostPoses.append((ghostpos[0],ghostpos[1]+0.5))
                GhostPoses.append((ghostpos[0],ghostpos[1]-0.5))
        #
        if self.ChasingGhostflg:
            if len(GhostPoses)>1:
                distance=self.FindClosetFooducs(currentGameState,GhostPoses,2)
            else:
                distance=self.FindClosetFooducs(currentGameState,GhostPoses,1)
        else:
            distance=self.FindClosetFooducs(currentGameState,GhostPoses,0)
        #
        reward-=distance*P_FDis

        if(self.eatPacflg):
            if(capsulecounter<self.Capcounter) and (not caprewarded):
                reward+=150
        else:
            if (capsulecounter<self.Capcounter):
                reward-=600

        #print currentGameState
        return currentGameState.getScore()+reward

    def FindClosetFooducs(self,currentState,ghost,triger):

        directions = {(0, 1),(1, 0),(0, -1),(-1, 0)}

        Food=currentState.getFood()
        Walls=currentState.getWalls()
        statesVisited=set()
        fringe=util.PriorityQueue()
        pos=currentState.getPacmanPosition()
        Capsules=currentState.getCapsules()

        fringe.push((pos,0),0)

        while not fringe.isEmpty():
            pos,cost=fringe.pop()
            if triger>0:
                if triger==2:
                    if pos in ghost:
                        return cost
                else:
                    if pos in Capsules:
                        return cost
            else:
                if Food[pos[0]][pos[1]]:
                    return cost
            if pos not in statesVisited:
                statesVisited.add(pos)
                for dx,dy in directions:
                    nextx, nexty = int(pos[0] + dx), int(pos[1] + dy)
                    if not Walls[nextx][nexty]:
                        nextpos=(nextx,nexty)
                        if nextpos not in statesVisited:
                            fringe.push((nextpos,cost+1),cost+1)
                            #if nextpos in ghost:
                                #fringe.push((nextpos,cost+2),cost+2)
                            #else:
                                #fringe.push((nextpos,cost+1),cost+1)
            else:
                continue;
        else:
            print "somethingwrong"
            print currentState
            print triger
            print ghost
            print sys.exit()
            return None

    def mazeDistance(pos1,pos2):
        return manhattanDistance(pos1,pos2)

    def Vector2Action(self,lastGpos,currentGpos):
        vector=(currentGpos[0]-lastGpos[0],currentGpos[1]-lastGpos[1])
        if(vector==(0,1)):
            return Directions.NORTH
        if(vector==(0-1)):
            return Directions.SOUTH
        if(vector==(1,0)):
            return Directions.EAST
        if(vector==(-1,0)):
            return Directions.WEST
        return Directions.STOP

    def InvAction(self,Action):
        if(Action==Directions.SOUTH):
            return Directions.NORTH
        if(Action==Directions.NORTH):
            return Directions.SOUTH
        if(Action==Directions.WEST):
            return Directions.EAST
        if(Action==Directions.EAST):
            return Directions.WEST
        return Directions.STOP
