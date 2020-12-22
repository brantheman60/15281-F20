# pacmanPlot.py
# -------------
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


from graphicsDisplay import PacmanGraphics
from graphicsDisplay import InfoPane
import graphicsDisplay
import graphicsUtils
from game import GameStateData
from game import AgentState
from game import Configuration
from game import Directions
from layout import Layout
from tkinter import mainloop
import math
import numpy as np
import time

LINE_COLOR = graphicsUtils.formatColor(0, 1, 0)

def lineBoxIntersection(w1, w2, b, xmin, ymin, xmax, ymax):
    """
    Figure out where line (w1*x+w2*y+b=0) intersections the 
    box (xmin, ymin) -> (xmax, ymax)
    """
        
    point1 = None
    point2 = None
    if w2 == 0:
        x1a = -(w2*ymin + b)*1.0/w1
        x1b = -(w2*ymax + b)*1.0/w1
    
        point1 = (x1a, ymin)
        point2 = (x1b, ymax)
    else:
        x2a = -(w1*xmin + b)*1.0/w2
        x2b = -(w1*xmax + b)*1.0/w2
        
        if w1 == 0:
            point1 = (xmin, x2a)
            point2 = (xmax, x2b)
        else:

            x1a = -(w2*ymin + b)*1.0/w1
            x1b = -(w2*ymax + b)*1.0/w1
            # Point 1
            if x2a < ymin:
                if xmin <= x1a and x1a <= xmax:
                    # Point 1 on bottom edge
                    point1 = (x1a, ymin)
            elif x2a > ymax:
                if xmin <= x1b and x1b <= xmax:
                    # Point 1 on top edge
                    point1 = (x1b, ymax)
            else:
                # Point 1 on left edge
                point1 = (xmin, x2a)
                
            # Point 2
            if point1 is not None:
                if x2b < ymin:
                    # Point 2 on bottom edge
                    point2 = (x1a, ymin)
                elif x2b > ymax:
                    # Point 2 on top edge
                    point2 = (x1b, ymax)
                else:
                    # Point 2 on right edge
                    point2 = (xmax, x2b)                                                
    return (point1, point2)

def plotPoints(x,y):
    """
    Create a Pacman display, plotting the points (x[i],y[i]) for all i in len(x).
    This method will block control and hand it to the displayed window.
           
    x: array or list of N scalar values.
    y: array or list of N scalar values.
    
    >>> x = range(-3,4)
    >>> squared = lambda x : x**2
    >>> y = map(squared, x)
    >>> pacmanPlot.plotPoints(x,y)
   """
    display = PacmanPlot(x,y)
    display.takeControl()

class PacmanPlot(PacmanGraphics):
    def __init__(self, x=None, y=None, zoom=1.0, frameTime=0.0):
        """
        Create and dispaly a pacman plot figure.
        
        If both x and y are provided, plot the points (x[i],y[i]) for all i in len(x).
        
        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.
        
        x: array or list of N scalar values. Default=None, in which case no points will be plotted
        y: array or list of N scalar values. Default=None, in which case no points will be plotted
        """        
        super(PacmanPlot, self).__init__(zoom, frameTime)

        if x is None or y is None:
            width = 23
            height = 23
            xmin = -(width-1)/2+1
            ymin = -(height-1)/2+1
            
            self.initPlot(xmin, ymin, width, height)
        else:
            self.plot(x,y)

    def initPlot(self, xmin, ymin, width, height):
        if graphicsUtils._canvas is not None:
            graphicsUtils.clear_screen()
        
        # Initialize GameStateData with blank board with axes    
        self.width = width
        self.height = height
        self.xShift = -(xmin-1)
        self.yShift = -(ymin-1)
        self.line = None

        self.zoom = min(30.0/self.width, 20.0/self.height)
        self.gridSize = graphicsDisplay.DEFAULT_GRID_SIZE * self.zoom


#         fullRow = ['%']*self.width
#         row = ((self.width-1)/2)*[' '] + ['%'] + ((self.width-1)/2)*[' ']
#         boardText = ((self.height-1)/2)*[row] + [fullRow] + ((self.height-1)/2)*[row]

        numSpaces = self.width-1
        numSpacesLeft = self.xShift
        numSpacesRight = numSpaces-numSpacesLeft

        numRows = self.height
        numRowsBelow = self.yShift
        numRowsAbove = numRows-1-numRowsBelow


        fullRow = ['%']*self.width
        if numSpacesLeft < 0:
            row = [' ']*self.width
        else:
            row = numSpacesLeft*[' '] + ['%'] + numSpacesRight*[' ']
        boardText = numRowsAbove*[row] + [fullRow] + numRowsBelow*[row]

        layout = Layout(boardText)
    
        self.blankGameState = GameStateData()
        self.blankGameState.initialize(layout, 0)
        self.initialize(self.blankGameState)
        title = 'Pacman Plot'
        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

    def plot(self, x, y, weights=None, title='Pacman Plot'):
        """
        Plot the input values x with their corresponding output values y (either true or predicted).
        Also, plot the linear regression line if weights are given; assuming h_w(x) = weights[0]*x + weights[1].
        
        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.
        
        x: array or list of N scalar values.
        y: array or list of N scalar values.
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return
        
        if isinstance(x[0], np.ndarray):
            # Scrape the first element of each data point
            x = [data[0] for data in x]
        
        xmin = int(math.floor(min(x)))
        ymin = int(math.floor(min(y)))
        xmax = int(math.ceil(max(x)))
        ymax = int(math.ceil(max(y)))
        width = xmax-xmin+3
        height = ymax-ymin+3
        self.initPlot(xmin, ymin, width, height)
        
        gameState = self.blankGameState.deepCopy()
                
        gameState.agentStates = []
        
        # Add ghost at each point
        for (px,py) in zip(x,y):
            point = (px+self.xShift, py+self.yShift)
            gameState.agentStates.append( AgentState( Configuration( point, Directions.STOP), False) )

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

    
    def setWeights(self, weights):
        pass        
        
    def takeControl(self):
        """
        Give full control to the window. Blocks current thread. Program will exit when window is closed.
        """
        mainloop()

#     def animate():
#         numSteps = width-2
#         for i in range(numSteps):
#             x = point1[0] + i*dx*1.0/numSteps
#             y = point1[1] + i*dy*1.0/numSteps
#             newPacman = AgentState( Configuration( (x,y), angle), True)
#             display.animatePacman(newPacman, display.agentImages[0][0], display.agentImages[0][1])
#             display.agentImages[0] = (newPacman, display.agentImages[0][1])
#         newPacman = AgentState( Configuration( point2, angle), True)
#         display.animatePacman(newPacman, display.agentImages[0][0], display.agentImages[0][1])
#         display.agentImages[0] = (newPacman, display.agentImages[0][1])


class PacmanPlotRegression(PacmanPlot):
    def __init__(self, zoom=1.0, frameTime=0.0):
        super(PacmanPlotRegression, self).__init__(zoom=zoom, frameTime=frameTime)
        self.addPacmanToLineStart = True
        
    def plot(self, x, y, weights=None, title='Linear Regression'):
        """
        Plot the input values x with their corresponding output values y (either true or predicted).
        Also, plot the linear regression line if weights are given; assuming h_w(x) = weights[0]*x + weights[1].
        
        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.
        
        x: array or list of N scalar values.
        y: array or list of N scalar values.
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return
        
        if isinstance(x[0], np.ndarray):
            # Scrape the first element of each data point
            x = [data[0] for data in x]
        
        xmin = int(math.floor(min(x)))
        ymin = int(math.floor(min(y)))
        xmax = int(math.ceil(max(x)))
        ymax = int(math.ceil(max(y)))
        width = xmax-xmin+3
        height = ymax-ymin+3
        self.initPlot(xmin, ymin, width, height)
        
        gameState = self.blankGameState.deepCopy()
                
        gameState.agentStates = []
        
        # Put pacman in bottom left
        if self.addPacmanToLineStart is True:
            gameState.agentStates.append( AgentState( Configuration( (1,1), Directions.STOP), True) )
            
        # Add ghost at each point
        for (px,py) in zip(x,y):
            point = (px+self.xShift, py+self.yShift)
            gameState.agentStates.append( AgentState( Configuration( point, Directions.STOP), False) )

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

        if weights is not None:
            self.setWeights(weights)

    def setWeights(self, weights):
        """
        Plot the linear regression line for given weights; assuming h_w(x) = weights[0]*x + weights[1].
        
        This will draw on the existing pacman window with the existing points
        
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        weights = np.array(weights)

        if weights.size >= 2:
            w = weights[0]
            b = weights[1]
        else:
            w = float(weights)
            b = 0
            
#         xmin = min(x)
#         xmax = max(x)
#         
#         ymin = w*xmin + b
#         ymax = w*xmax + b        
#         
#         point1 = (xmin+self.xShift, ymin+self.yShift)
#         point2 = (xmax+self.xShift, ymax+self.yShift)

        (point1, point2) = lineBoxIntersection(w, -1, b,
                                               1-self.xShift, 1-self.yShift,
                                               self.width-2-self.xShift, self.height-2-self.yShift)
        
        if point1 is not None and point2 is not None:
            point1 = (point1[0]+self.xShift, point1[1]+self.yShift)
            point2 = (point2[0]+self.xShift, point2[1]+self.yShift)
            
            dx = point2[0]-point1[0]
            dy = point2[1]-point1[1]
            if dx == 0:
                angle = 90 + 180*dy*1.0/abs(dy)
            else:
                angle = math.atan(dy*1.0/dx)*180.0/math.pi
                   
            if self.line is not None:
                graphicsUtils.remove_from_screen(self.line) 
            self.line = graphicsUtils.polygon([self.to_screen(point1), self.to_screen(point2)], LINE_COLOR, filled=0, behind=0)
                
            if self.addPacmanToLineStart is True and len(self.agentImages) > 0:
                # Bring pacman to front of display
                graphicsUtils._canvas.tag_raise(self.agentImages[0][1][0])
                
                # Put pacman at beginning of line
                self.movePacman(point1, angle, self.agentImages[0][1])

            graphicsUtils.refresh()
            graphicsUtils.sleep(1)

class PacmanPlotLogisticRegression1D(PacmanPlot):
    def __init__(self, zoom=1.0, frameTime=0.0):
        super(PacmanPlotLogisticRegression1D, self).__init__(zoom=zoom, frameTime=frameTime)
        self.addPacmanToLineStart = False
       
    def plot(self, x, y, weights=None, title='Logistic Regression'):
        """
        Plot the 1D input points, data[i], colored based on their corresponding labels (either true or predicted).
        Also, plot the logistic function fit if weights are given.
    
        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.
    
        x: list of 1D points, where each 1D point in the list is a 1 element numpy.ndarray
        y: list of N labels, one for each point in data. Labels can be of any type that can be converted
            a string.
        weights: array of 2 values the first one is the weight on the data and the second value is the bias weight term.
        If there are only 1 values in weights,
            the bias term is assumed to be zero.  If None, no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return
        
        # Process data, sorting by label
        possibleLabels = list(set(y))
        sortedX = {}
        for label in possibleLabels:
            sortedX[label] = []
    
        for i in range(len(x)):
            sortedX[y[i]].append(x[i])
    
        xmin = int(math.floor(min(x)))
        xmax = int(math.ceil(max(x)))
        ymin = int(math.floor(0))-1
        ymax = int(math.ceil(1))
        width = xmax-xmin+3
        height = ymax-ymin+3
        self.initPlot(xmin, ymin, width, height)
        
        gameState = self.blankGameState.deepCopy()
                
        gameState.agentStates = []
        
        # Put pacman in bottom left
        if self.addPacmanToLineStart is True:
            gameState.agentStates.append( AgentState( Configuration( (1,1), Directions.STOP), True) )
            
        # Add ghost at each point
        for (py, label) in enumerate(possibleLabels):
            pointsX = sortedX[label]
            for px in pointsX:
                point = (px+self.xShift, py+self.yShift)
                agent = AgentState( Configuration( point, Directions.STOP), False)
                agent.isPacman = 1-py 
                gameState.agentStates.append(agent)

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

        if weights is not None:
            self.setWeights(weights)

    def setWeights(self, weights):
        """
        Plot the logistic regression line for given weights
        
        This will draw on the existing pacman window with the existing points
        
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        weights = np.array(weights)

        if weights.size >= 2:
            w = weights[0]
            b = weights[1]
        else:
            w = float(weights)
            b = 0
            
        xmin = 1 - self.xShift
        xmax = self.width-2 - self.xShift
    
        x = np.linspace(xmin, xmax,30)
        y = 1.0/(1+np.exp(-(w*x+b)))
        x += self.xShift
        y += self.yShift

        if self.line is not None:
            for obj in self.line:
                graphicsUtils.remove_from_screen(obj)
                 
        self.line = []
        
        prevPoint = self.to_screen((x[0],y[0]))
        for i in xrange(1,len(x)):
            point = self.to_screen((x[i],y[i]))
            self.line.append(graphicsUtils.line(prevPoint, point, LINE_COLOR))
            prevPoint = point
            
#         prevPoint = self.to_screen((x[0],y[0]))
#         for i in xrange(1,len(x)):
#             point = self.to_screen((x[i],y[i]))
#             line.append(graphicsUtils.line(prevPoint, point, LINE_COLOR, filled=0, behind=0)
#                         
#             prevPoint = point

            
        if self.addPacmanToLineStart is True and len(self.agentImages) > 0:
            # Bring pacman to front of display
            graphicsUtils._canvas.tag_raise(self.agentImages[0][1][0])
            
            # Put pacman at beginning of line
            if w >= 0:
                self.movePacman((x[0]-0.5,y[0]), Directions.EAST, self.agentImages[0][1])
            else:
                self.movePacman((x[-1]+0.5,y[-1]), Directions.WEST, self.agentImages[0][1])

        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

class PacmanPlotClassification2D(PacmanPlot):
    def __init__(self, zoom=1.0, frameTime=0.0):
        super(PacmanPlotClassification2D, self).__init__(zoom=zoom, frameTime=frameTime)
       
    def plot(self, x, y, weights=None, title='Linear Classification'):
        """
        Plot the 2D input points, data[i], colored based on their corresponding labels (either true or predicted).
        Also, plot the linear separator line if weights are given.
    
        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.
    
        x: list of 2D points, where each 2D point in the list is a 2 element numpy.ndarray
        y: list of N labels, one for each point in data. Labels can be of any type that can be converted
            a string.
        weights: array of 3 values the first two are the weight on the data and the third value is the bias
        weight term. If there are only 2 values in weights, the bias term is assumed to be zero.  If None,
        no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return
        
        # Process data, sorting by label
        possibleLabels = list(set(y))
        sortedX1 = {}
        sortedX2 = {}
        for label in possibleLabels:
            sortedX1[label] = []
            sortedX2[label] = []
    
        for i in range(len(x)):
            sortedX1[y[i]].append(x[i][0])
            sortedX2[y[i]].append(x[i][1])
    
        x1min = float("inf")
        x1max = float("-inf")
        for x1Values in sortedX1.values():
            x1min = min(min(x1Values), x1min)
            x1max = max(max(x1Values), x1max)
        x2min = float("inf")
        x2max = float("-inf")
        for x2Values in sortedX2.values():
            x2min = min(min(x2Values), x2min)
            x2max = max(max(x2Values), x2max)

        x1min = int(math.floor(x1min))
        x1max = int(math.ceil(x1max))
        x2min = int(math.floor(x2min))
        x2max = int(math.ceil(x2max))

        width = x1max-x1min+3
        height = x2max-x2min+3
        self.initPlot(x1min, x2min, width, height)
        
        gameState = self.blankGameState.deepCopy()
                
        gameState.agentStates = []
        
        # Add ghost/pacman at each point
        for (labelIndex, label) in enumerate(possibleLabels):
            pointsX1 = sortedX1[label]
            pointsX2 = sortedX2[label]
            for (px, py) in zip(pointsX1, pointsX2):
                point = (px+self.xShift, py+self.yShift)
                agent = AgentState( Configuration( point, Directions.STOP), False)
                agent.isPacman = (labelIndex==0) 
                if labelIndex==2:
                    agent.scaredTimer = 1
                gameState.agentStates.append(agent)

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

        if weights is not None:
            self.setWeights(weights)

    def setWeights(self, weights):
        """
        Plot the logistic regression line for given weights
        
        This will draw on the existing pacman window with the existing points
        
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        weights = np.array(weights)

        w1 = weights[0]
        w2 = weights[1]
        if weights.size >= 3:
            b = weights[2]
        else:
            b = 0
            
        # Line functions
        # Line where w1*x1 + w2*x2 + b = 0
        # x2 = -(w1*x1 + b)/w2  or
        # x1 = -(w2*x2 + b)/w1

        # Figure out where line intersections bounding box around points

        (point1, point2) = lineBoxIntersection(w1, w2, b,
                                               1-self.xShift, 1-self.yShift,
                                               self.width-2-self.xShift, self.height-2-self.yShift)

        if point1 is not None and point2 is not None:
            point1 = (point1[0]+self.xShift, point1[1]+self.yShift)
            point2 = (point2[0]+self.xShift, point2[1]+self.yShift)
            
            if self.line is not None:
                graphicsUtils.remove_from_screen(self.line) 
            self.line = graphicsUtils.polygon([self.to_screen(point1), self.to_screen(point2)], LINE_COLOR, filled=0, behind=0)
                
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

class PacmanPlotLP(PacmanGraphics):
    def __init__(self, constraints=[], infeasiblePoints=[], feasiblePoints=[], optimalPoint=None, costVector=None, zoom=1.0, frameTime=0.0):
        """
        Create and dispaly a pacman plot figure.
        
        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.
        
        constraints: list of inequality constraints, where each constraint w1*x + w2*y <= b is represented as a tuple ((w1, w2), b)
        infeasiblePoints (food): list of points where each point is a tuple (x, y)
        feasiblePoints (power): list of points where each point is a tuple (x, y)
        optimalPoint (pacman): optimal point as a tuple (x, y)
        costVector (shading): cost vector represented as a tuple (c1, c2), where cost is c1*x + c2*x
        """        
        super(PacmanPlotLP, self).__init__(zoom, frameTime)

        xmin = 100000
        ymin = 100000
        xmax = -100000
        ymax = -100000

        for point in feasiblePoints:
            if point[0] < xmin:
                xmin = point[0]
            if point[0] > xmax:
                xmax = point[0]
            if point[1] < ymin:
                ymin = point[1]
            if point[1] > ymax:
                ymax = point[1]

        if len(feasiblePoints) == 0:
            for point in infeasiblePoints:
                if point[0] < xmin:
                    xmin = point[0]
                if point[0] > xmax:
                    xmax = point[0]
                if point[1] < ymin:
                    ymin = point[1]
                if point[1] > ymax:
                    ymax = point[1]

        xmin = int(math.floor(xmin)) - 3
        ymin = int(math.floor(ymin)) - 3
        xmax = int(math.ceil(xmax)) + 3
        ymax = int(math.ceil(ymax)) + 3
        width = xmax-xmin+1
        height = ymax-ymin+1

#        p = feasiblePoints[2]
#        print("p={}".format(p))
#        print("feasible={}".format(self.pointFeasible(p, constraints)))
#        g = self.cartesianToLayout(xmin, ymin, xmax, ymax, p)
#        print("g={}".format(g))
#        gr = (int(round(g[0])), int(round(g[1])))
#        p2 = self.layoutToCartesian(xmin, ymin, xmax, ymax, gr)
#        print("p2={}".format(p2))
#        print("p2 feasible={}".format(self.pointFeasible(p2, constraints)))

        layoutLists = self.blankLayoutLists(width, height)

        self.addInfeasibleGhosts(layoutLists, constraints, xmin, ymin, xmax, ymax)

        layoutLists = self.changeBorderGhostsToWall(layoutLists)
        
        for point in infeasiblePoints:
            self.addCartesianPointToLayout(layoutLists, point, '.', xmin, ymin, xmax, ymax)

        for point in feasiblePoints:
            self.addCartesianPointToLayout(layoutLists, point, 'o', xmin, ymin, xmax, ymax)

        if optimalPoint is not None:
            self.addCartesianPointToLayout(layoutLists, optimalPoint, 'P', xmin, ymin, xmax, ymax)

        if graphicsUtils._canvas is not None:
            graphicsUtils.clear_screen()
        
        # Initialize GameStateData with blank board with axes    
        self.width = width
        self.height = height

        self.zoom = min(30.0/self.width, 20.0/self.height)
        self.gridSize = graphicsDisplay.DEFAULT_GRID_SIZE * self.zoom

        maxNumGhosts = 10000
        layout = Layout(layoutLists)
        self.blankGameState = GameStateData()
        self.blankGameState.initialize(layout, maxNumGhosts)
        self.initialize(self.blankGameState)
        title = 'Pacman Plot LP'
        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()
        graphicsUtils.sleep(1)

        if costVector is not None:
            self.shadeCost(layoutLists, constraints, costVector, feasiblePoints, xmin, ymin, xmax, ymax)

    def takeControl(self):
        """
        Give full control to the window. Blocks current thread. Program will exit when window is closed.
        """
        mainloop()

    def pointCost(self, costVector, point):
        return costVector[0]*point[0] + costVector[1]*point[1]

    def shadeCost(self, layout, constraints, costVector, feasiblePoints, xmin, ymin, xmax, ymax):
        baseColor = [1.0, 0.0, 0.0]

        costs = [self.pointCost(costVector, point) for point in feasiblePoints]
        minCost = min(costs)
        maxCost = max(costs)
        costSpan = maxCost - minCost

        allFeasiblePoints = self.getFeasibleLayoutPoints(layout, constraints, xmin, ymin, xmax, ymax)

        # The feasible points themselves may have been gridded to infeasible grid points,
        # but we want to make sure they are shaded too.
        #cornerPoints = [self.cartesianToLayout(xmin, ymin, xmax, ymax, point) for point in feasiblePoints]
        cornerPoints = self.getLayoutPointsWithSymbol(layout, ('o', 'P'))

        gridPointsToShade = cornerPoints + allFeasiblePoints

        for gridPoint in gridPointsToShade:
            point = self.layoutToCartesian(xmin, ymin, xmax, ymax, gridPoint)

            relativeCost = (self.pointCost(costVector, point) - minCost) * 1.0 / costSpan
            
            # Whoops our grid points are flipped top-bottom from what to_screen expects
            screenPos = self.to_screen((gridPoint[0], len(layout) - gridPoint[1] - 1)) 

            cellColor = [0.25 + 0.5*relativeCost*channel for channel in baseColor]

            graphicsUtils.square(screenPos,
                    0.5 * self.gridSize,
                    color=graphicsUtils.formatColor(*cellColor),
                    filled=1, behind=2) 

        graphicsUtils.refresh()
        graphicsUtils.sleep(1)


    def layoutToCartesian(self, xmin, ymin, xmax, ymax, point):
        xnew = point[0] + xmin
        ynew = (ymax-ymin) - point[1] + ymin

        return (xnew, ynew)

    def cartesianToLayout(self, xmin, ymin, xmax, ymax, point):
        xnew = point[0] - xmin
        ynew = (ymax-ymin) - (point[1] - ymin)
    
        return (xnew, ynew)
    
    def printLayout(self, layout):
        print('-'*(len(layout[0])+2))
        for row in layout:
            print('|'+''.join(row)+'|')
        print('-'*(len(layout[0])+2))
    
    def blankLayoutLists(self, width, height):
        layout = []
        for _ in range(height):
            row = [' ']*width
            layout.append(row)
    
        return layout
    
    def roundPoint(self, p):
        return ( int(round(p[0])), int(round(p[1])) )
        
    def setLayoutWall(self, layout, point):
        self.setLayoutPoint(layout, point, '%')

    def getLayoutPointsWithSymbol(self, layout, symbolSet):
        points = []
        for gy in range(len(layout)):
            for gx in range(len(layout[0])):
                if layout[gy][gx] in symbolSet:
                    points.append((gx, gy))

        return points

    def getLayoutSymbol(self, layout, point):
        point = self.roundPoint(point)
    
        if point[0] >= 0 and point[0] < len(layout[0]) and point[1] >= 0 and point[1] < len(layout):
            return layout[point[1]][point[0]]

        return None

    def setLayoutPoint(self, layout, point, symbol):
        point = self.roundPoint(point)
    
        if point[0] >= 0 and point[0] < len(layout[0]) and point[1] >= 0 and point[1] < len(layout):
            layout[point[1]][point[0]] = symbol
            return True

        return False

    def distance(self, p1, p2):
        vec = ( p2[0]-p1[0], p2[1]-p1[1] )
        vecLen = math.sqrt(vec[0]**2 + vec[1]**2)
        return vecLen
        
    def addLineToLayout(self, layout, p1, p2):
        radius=1
        STEPS_PER_UNIT = 10
    
        fullVec = ( p2[0]-p1[0], p2[1]-p1[1] )
        fullVecLen = math.sqrt(fullVec[0]**2 + fullVec[1]**2)
     
        stepVec = ( fullVec[0]/fullVecLen/STEPS_PER_UNIT, fullVec[1]/fullVecLen/STEPS_PER_UNIT )
        numSteps = int(math.ceil(fullVecLen)*STEPS_PER_UNIT)
    
        self.setLayoutWall(layout, p1)
        point = p1
        while point != p2:
            # Generate four connected points
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            nextPoints = [(point[0]+delta[0], point[1]+delta[1]) for delta in deltas]
            distances = [self.distance(nextPoint, p2) for nextPoint in nextPoints]
            minDist = distances[0]
            minIndex = 0
            for i, dist in enumerate(distances):
                if dist < minDist:
                    minDist = dist
                    minIndex = i
            #print("distances={}, minIndex={}".format(distances, minIndex))

            point = nextPoints[minIndex]
            self.setLayoutWall(layout, point)
            #print("p1={}, point={}, p2={}".format(p1, point, p2))
            
        #for n in range(1, numSteps+1):
        #    x = p1[0] + n*stepVec[0]
        #    y = p1[1] + n*stepVec[1]
        #    self.setLayoutWall(layout, (x,y))
    
    def getCartesianSymbol(self, layout, point, xmin, ymin, xmax, ymax):
        point = self.cartesianToLayout(xmin, ymin, xmax, ymax, point)
        return self.getLayoutSymbol(layout, point)
        
    def addCartesianPointToLayout(self, layout, point, symbol, xmin, ymin, xmax, ymax):
        point = self.cartesianToLayout(xmin, ymin, xmax, ymax, point)
        return self.setLayoutPoint(layout, point, symbol)

    def addCartesianLineToLayout(self, layout, w1, w2, b, xmin, ymin, xmax, ymax):
        (p1, p2) = lineBoxIntersection(w1, w2, b, xmin, ymin, xmax, ymax)
        p1 = self.cartesianToLayout(xmin, ymin, xmax, ymax, p1)    
        p2 = self.cartesianToLayout(xmin, ymin, xmax, ymax, p2)    
    
        self.addLineToLayout(layout, p1, p2)
    
    def pointFeasible(self, point, constraints):
        EPSILON = 1e-6
        for constraint in constraints:
            if constraint[0][0]*point[0] + constraint[0][1]*point[1] > constraint[1] + EPSILON:
                #print("Infeasible: point={}, constraint={}".format(point, constraint))
                #print("\t{}*{} + {}*{} = {}".format(constraint[0][0], point[0], constraint[0][1], point[1], constraint[0][0]*point[0] + constraint[0][1]*point[1]))
                return False

        return True

    def getFeasibleLayoutPoints(self, layout, constraints, xmin, ymin, xmax, ymax):
        height = len(layout)
        width = len(layout[0])

        layoutPoints = []

        for gy in range(height):
            for gx in range(width):
                point = self.layoutToCartesian(xmin, ymin, xmax, ymax, (gx, gy))
                if self.pointFeasible(point, constraints):
                    layoutPoints.append((gx, gy))

        return layoutPoints

    def addInfeasibleGhosts(self, layout, constraints, xmin, ymin, xmax, ymax):
        numGhosts = 0

        height = len(layout)
        width = len(layout[0])

        for gy in range(height):
            for gx in range(width):
                point = self.layoutToCartesian(xmin, ymin, xmax, ymax, (gx, gy))
                if not self.pointFeasible(point, constraints):
                    self.setLayoutPoint(layout, (gx, gy), 'G')
                    numGhosts += 1
        
        return numGhosts

    def isSymbolNeighbor(self, layout, point, symbols):
        deltas = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]    

        neighborSymbols = [self.getLayoutSymbol(layout, (point[0]+delta[0], point[1]+delta[1])) for delta in deltas]
        for symbol in symbols:
            if symbol in neighborSymbols:
                return True        
    
        return False

    def changeBorderGhostsToWall(self, layout):
        width = len(layout[0])
        height = len(layout)

        newLayout = self.blankLayoutLists(width, height)
    
        for gy in range(height):
            for gx in range(width):
                symbol = self.getLayoutSymbol(layout, (gx, gy))
                newLayout[gy][gx] = symbol

                if symbol == 'G':
                    if self.isSymbolNeighbor(layout, (gx, gy), (' ', 'o', 'P')):
                        newLayout[gy][gx] = '%'

        return newLayout
    
if __name__ == '__main__':
    """
    Demo code
    """
    # LP

    #constraints = [((1, 1), 10), ((-1, 1), 10), ((1, -1), 10), ((-1, -1), 10)]
    #infeasiblePoints = [(-10, -10), (-10, 10), (10, 10), (10, -10)]
    #feasiblePoints = [(-10, 0), (0, 10), (10, 0), (0, -10)]
    #optimalPoint = feasiblePoints[0]
    #costVector = (1, 0.5)

    constraints = [((0, -1), 0), ((-1, 0), 0), ((0.25, 1), 16), ((1.5, 1), 40)]
    infeasiblePoints = [(0, 40)]
    feasiblePoints = [(0, 0), (0, 16), (40/1.5, 0), (24/1.25, 16-(0.25*(24/1.5)))]
    optimalPoint = feasiblePoints[3]
    costVector = (-1, -1)

    display = PacmanPlotLP(constraints=constraints,
            feasiblePoints=feasiblePoints,
            infeasiblePoints=infeasiblePoints,
            optimalPoint=optimalPoint,
            costVector=costVector)

    display.takeControl()
    exit()

    time.sleep(2)

    # Regression
         
    display = PacmanPlotRegression()
        
    x = np.random.normal(0,1,10)
    display.plot(x*3, x**3)
    time.sleep(2)
       
    for i in range(6):
        display.setWeights([(5-i)/5.0])
        time.sleep(1)
    time.sleep(1)
            
    # With offset
    display.setWeights([0, -5])
    time.sleep(2)
    
    # Classification 2D
     
    display = PacmanPlotClassification2D()
      
    # Generate labeled points
    means = ((4,4), (-4,4), (0,-4))
    labelNames= ('A','B','C')
    labels = []
    data = []
    for i in range(15):
        labelIndex = np.random.randint(len(labelNames))
        labels.append(labelNames[labelIndex])
        mean = np.array(means[labelIndex])
        data.append(np.random.normal(mean,1,mean.shape))
   
    display.plot(data, labels)
    time.sleep(2)
   
    for i in range(8):
        display.setWeights([4,i])
        time.sleep(1)
  
   
    # With offset and horizontal separator
    display.setWeights([0, 1, -3])
    time.sleep(2)
    
    # Logistic Regression
        
    display = PacmanPlotLogisticRegression1D()
       
    # Generate labeled points
    means = (1, -5)
    labelNames= ('A','B')
    labels = []
    data = []
    for i in range(15):
        labelIndex = np.random.randint(len(labelNames))
        labels.append(labelNames[labelIndex])
        mean = np.array(means[labelIndex])
        data.append(np.random.normal(mean,3,mean.shape))
   
    display.plot(data, labels)
    time.sleep(2)
   
    for i in range(8):
        display.setWeights(4-i)
        time.sleep(1)
    time.sleep(1)
   
    # With offset and horizontal separator
    display.setWeights([-3, -6])

#     # Just some extra tests
#     display = PacmanPlotRegression()
#     display.plot([-2, 2, 2], [2, 2, -2], [4,0])
#     display = PacmanPlotClassification2D()
#     display.plot([np.array([-1,1]), np.ones(2), np.array([1,-1])], [0,1,2], [1,0])

    display.takeControl()
    # Blocked until the window is closed
    
