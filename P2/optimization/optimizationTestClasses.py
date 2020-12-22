# optimizationTestClasses.py
# --------------------------
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


from hashlib import sha1
import testClasses

from collections import defaultdict
from pprint import PrettyPrinter
pp = PrettyPrinter()

from pacman import GameState
from ghostAgents import RandomGhost
import random
import math
import traceback
import sys
import os
import time
import layout
import pacman
import pacmanPlot
import ast

import numpy as np
import util
from util import FixedRandom

class UnitTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(UnitTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "{}.preamble".format(self.getPath()), 'exec')
        self.test = compile(testDict['test'], "{}.test".format(self.getPath()), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        try:
            solution = float(solutionDict['result'])
        except ValueError:
            solution = solutionDict['result']
            solution = solution.replace('[','')
            solution = solution.replace(']','')
            solution = solution.split(' ')
            solution = [s for s in solution if s!='']
            for i in range(len(solution)):
                solution[i] = float(solution[i])
            solution = np.array(solution)

        error = result - solution
        errorNorm = np.linalg.norm(np.array(error))
        if errorNorm < self.tolerance:
            grades.addMessage('PASS: ' + self.path)
            grades.addMessage('\t' + self.success)
            if self.partialPoints > 0:
                print("                    ({} of {} points)".format(self.partialPoints, self.partialPoints))
                grades.addPoints(self.partialPoints)
            return True
        else:
            grades.addMessage('FAIL: ' + self.path)
            grades.addMessage('\t' + self.failure)
            grades.addMessage('\tstudent result: "{}"'.format(result))
            grades.addMessage('\tcorrect result: "{}"'.format(solutionDict['result']))
        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(0, self.partialPoints))
        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "{}"\n'.format(output))
        # print '>>>>>>>>>>>>>>>>>>result:"{}"\n'.format(output)
        handle.write('result:"{}"\n'.format(output))
        handle.close()
        return True



class PointTest(testClasses.TestCase):
    """
    Unit test for code that returns a tuple of floats.
    Checks that result matches solution within a given tolerance.
    """

    def __init__(self, question, testDict):
        super(PointTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "{}.preamble".format(self.getPath()), 'exec')
        self.test = compile(testDict['test'], "{}.test".format(self.getPath()), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result_point = self.evalCode(moduleDict)

        solution_point = ast.literal_eval(solutionDict['result'])

        numDims = len(solution_point)
        if len(result_point) != numDims:
            return self.fail(grades, result_point, solution_point)

        # Make sure each solution point exists in result
        error = np.array(result_point) - np.array(solution_point)
        errorNorm = np.linalg.norm(error)
        if errorNorm > self.tolerance:
            return self.fail(grades, result_point, solution_point)
                
        # Otherwise pass
        grades.addMessage('PASS: ' + self.path)
        grades.addMessage('\t' + self.success)
        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(self.partialPoints, self.partialPoints))
            grades.addPoints(self.partialPoints)
        return True

    def fail(self, grades, result, solution):
        grades.addMessage('FAIL: ' + self.path)
        grades.addMessage('\t' + self.failure)
        grades.addMessage('\tstudent result: "{}"'.format(result))
        grades.addMessage('\tcorrect result: "{}"'.format(solution))

        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(0, self.partialPoints))

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "{}"\n'.format(output))
        # print '>>>>>>>>>>>>>>>>>>result:"{}"\n'.format(output)
        handle.write('result:"{}"\n'.format(output))
        handle.close()
        return True



class ListOfPointsTest(testClasses.TestCase):
    """
    Unit test for code that returns a list of tuples of floats.
    Checks that result matches solution within a given tolerance.
    """

    def __init__(self, question, testDict):
        super(ListOfPointsTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "{}.preamble".format(self.getPath()), 'exec')
        self.test = compile(testDict['test'], "{}.test".format(self.getPath()), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result_list = self.evalCode(moduleDict)

        solution_list = ast.literal_eval(solutionDict['result'])

        if len(result_list) != len(solution_list):
            return self.fail(grades, result_list, solution_list)

        numDims = len(solution_list[0])
        for result_point in result_list:
            if len(result_point) != numDims:
                return self.fail(grades, result_list, solution_list)

        # Make sure each solution point exists in result
        for solution_point in solution_list:
            sol_in_result = False
            for result_point in result_list:
                error = np.array(result_point) - np.array(solution_point)
                errorNorm = np.linalg.norm(error)
                if errorNorm < self.tolerance:
                    sol_in_result = True
                    break
            if not sol_in_result:
                return self.fail(grades, result_list, solution_list)
                
        # Otherwise pass
        grades.addMessage('PASS: ' + self.path)
        grades.addMessage('\t' + self.success)
        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(self.partialPoints, self.partialPoints))
            grades.addPoints(self.partialPoints)
        return True

    def fail(self, grades, result, solution):
        grades.addMessage('FAIL: ' + self.path)
        grades.addMessage('\t' + self.failure)
        grades.addMessage('\tstudent result: "{}"'.format(result))
        grades.addMessage('\tcorrect result: "{}"'.format(solution))

        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(0, self.partialPoints))

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "{}"\n'.format(output))
        # print '>>>>>>>>>>>>>>>>>>result:"{}"\n'.format(output)
        handle.write('result:"{}"\n'.format(output))
        handle.close()
        return True



class AllPointsTest(testClasses.TestCase):
    """
    Unit test for code that returns a list of tuples of floats.
    Checks that result matches solution within a given tolerance.
    """

    def __init__(self, question, testDict):
        super(AllPointsTest, self).__init__(question, testDict)
        self.showPlot = not question.display.checkNullDisplay()
        self.moduleStr = testDict['module']
        self.functionStr = testDict['function']
        self.constraints = ast.literal_eval(testDict['constraints'])
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        module = moduleDict[self.moduleStr]
        function = getattr(module, self.functionStr)

        timeout = 60
        timed_function = util.TimeoutFunction(function, timeout)
        try:
            result = timed_function(self.constraints)
        except util.TimeoutFunctionException:
            result = "Timed out after {} seconds".format(timeout)

        return result

    def plotPoints(self, points, sleep=1):
        if not self.showPlot:
            return
        if len(points) == 0 or len(points[0]) != 2:
            return

        if "Feasible" in self.functionStr:
            display = pacmanPlot.PacmanPlotLP(constraints=self.constraints, feasiblePoints=points)
        else:
            display = pacmanPlot.PacmanPlotLP(constraints=self.constraints, infeasiblePoints=points)
    
        time.sleep(sleep)

    def execute(self, grades, moduleDict, solutionDict):
        result_list = self.evalCode(moduleDict)

        solution_list = ast.literal_eval(solutionDict['result'])

        if isinstance(result_list, str) and "Timed out" in result_list:
            return self.fail(grades, result_list, solution_list)

        if len(result_list) != len(solution_list):
            return self.fail(grades, result_list, solution_list)

        if len(solution_list) > 0:
            numDims = len(solution_list[0])
            for result_point in result_list:
                if len(result_point) != numDims:
                    return self.fail(grades, result_list, solution_list)

        # Make sure each solution point exists in result
        for solution_point in solution_list:
            sol_in_result = False
            for result_point in result_list:
                error = np.array(result_point) - np.array(solution_point)
                errorNorm = np.linalg.norm(error)
                if errorNorm < self.tolerance:
                    sol_in_result = True
                    break
            if not sol_in_result:
                return self.fail(grades, result_list, solution_list)
                
        # Otherwise pass
        grades.addMessage('PASS: ' + self.path)
        grades.addMessage('\t' + self.success)
        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(self.partialPoints, self.partialPoints))
            grades.addPoints(self.partialPoints)
        self.plotPoints(result_list)
        return True

    def fail(self, grades, result, solution):
        grades.addMessage('FAIL: ' + self.path)
        grades.addMessage('\t' + self.failure)
        grades.addMessage('\tstudent result: "{}"'.format(result))
        grades.addMessage('\tcorrect result: "{}"'.format(solution))

        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(0, self.partialPoints))

        self.plotPoints(result, sleep=5)
        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "{}"\n'.format(output))
        # print '>>>>>>>>>>>>>>>>>>result:"{}"\n'.format(output)
        handle.write('result:"{}"\n'.format(output))
        handle.close()
        return True



class PointObjTest(testClasses.TestCase):
    """
    Unit test for code that returns a list of tuples of floats.
    Checks that result matches solution within a given tolerance.
    """

    def __init__(self, question, testDict):
        super(PointObjTest, self).__init__(question, testDict)
        self.showPlot = not question.display.checkNullDisplay()
        self.moduleStr = testDict['module']
        self.functionStr = testDict['function']
        if 'constraints' in testDict:
            self.constraints = ast.literal_eval(testDict['constraints'])
        else:
            self.constraints = None
        if 'cost' in testDict:
            self.cost = ast.literal_eval(testDict['cost'])
        else:
            self.cost = None
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        module = moduleDict[self.moduleStr]
        function = getattr(module, self.functionStr)

        timeout = 180
        timed_function = util.TimeoutFunction(function, timeout)
        try:
            if self.constraints is None or self.cost is None:
                result = timed_function()
            else:
                result = timed_function(self.constraints, self.cost)
        except util.TimeoutFunctionException:
            result = "Timed out after {} seconds".format(timeout)

        return result


    def getFeasiblePoints(self, moduleDict):
        module = moduleDict[self.moduleStr]
        function = getattr(module, "findFeasibleIntersections")
        if self.constraints is None or self.cost is None:
            return []
        else:
            return function(self.constraints)

    def plotPoints(self, moduleDict, pointObj, sleep=1):
        if not self.showPlot:
            return
        if pointObj is None or len(pointObj[0]) != 2:
            return

        feasiblePoints = self.getFeasiblePoints(moduleDict)

        if len(feasiblePoints) >= 2:
            display = pacmanPlot.PacmanPlotLP(constraints=self.constraints, feasiblePoints=feasiblePoints,
                    optimalPoint=pointObj[0], costVector=self.cost)
    
        time.sleep(sleep)

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)

        solution = ast.literal_eval(solutionDict['result'])

        if solution is None and result is not None:
            return self.fail(grades, moduleDict, result, solution)

        if result is None and solution is not None:
            return self.fail(grades, moduleDict, result, solution)

        if isinstance(result, str) and "Timed out" in result:
            return self.fail(grades, moduleDict, result, solution)

        if solution is not None:
            numDims = len(solution[0])
            result_point = result[0]
            if len(result_point) != numDims:
                return self.fail(grades, moduleDict, result, solution)

            solution_point = solution[0]
            error = np.array(result_point) - np.array(solution_point)
            errorNorm = np.linalg.norm(error)
            if errorNorm > self.tolerance:
                return self.fail(grades, moduleDict, result, solution)

            result_obj = result[1]
            solution_obj = solution[1]
            if np.abs(result_obj - solution_obj) > self.tolerance:
                return self.fail(grades, moduleDict, result, solution)
                
        # Otherwise pass
        grades.addMessage('PASS: ' + self.path)
        grades.addMessage('\t' + self.success)
        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(self.partialPoints, self.partialPoints))
            grades.addPoints(self.partialPoints)
        self.plotPoints(moduleDict, result)
        return True

    def fail(self, grades, moduleDict, result, solution):
        grades.addMessage('FAIL: ' + self.path)
        grades.addMessage('\t' + self.failure)
        grades.addMessage('\tstudent result: "{}"'.format(result))
        grades.addMessage('\tcorrect result: "{}"'.format(solution))

        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(0, self.partialPoints))

        self.plotPoints(moduleDict, result, sleep=5)
        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "{}"\n'.format(output))
        # print '>>>>>>>>>>>>>>>>>>result:"{}"\n'.format(output)
        handle.write('result:"{}"\n'.format(output))
        handle.close()
        return True



class FoodDistributionTest(testClasses.TestCase):
    """
    Unit test for code that returns a list of tuples of floats.
    Checks that result matches solution within a given tolerance.
    """

    def __init__(self, question, testDict):
        super(FoodDistributionTest, self).__init__(question, testDict)
        self.showPlot = not question.display.checkNullDisplay()
        self.moduleStr = testDict['module']
        self.functionStr = 'foodDistribution' 
        self.truck_limit = ast.literal_eval(testDict['truck_limit'])
        self.W = ast.literal_eval(testDict['W'])
        self.C = ast.literal_eval(testDict['C'])
        self.T = ast.literal_eval(testDict['T'])
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        module = moduleDict[self.moduleStr]
        function = getattr(module, self.functionStr)

        timeout = 300
        timed_function = util.TimeoutFunction(function, timeout)
        try:
            result = timed_function(self.truck_limit, self.W, self.C, self.T)
        except util.TimeoutFunctionException:
            result = "Timed out after {} seconds".format(timeout)

        return result

    def getFeasiblePoints(self, moduleDict):
        module = moduleDict[self.moduleStr]
        function = getattr(module, "findFeasibleIntersections")
        if self.constraints is None or self.cost is None:
            return []
        else:
            return function(self.constraints)

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)

        solution = ast.literal_eval(solutionDict['result'])

        if solution is None and result is not None:
            return self.fail(grades, result, solution)

        if result is None and solution is not None:
            return self.fail(grades, result, solution)

        if isinstance(result, str) and "Timed out" in result:
            return self.fail(grades, result, solution)

        if solution is not None:
            numDims = len(solution[0])
            result_point = result[0]
            if len(result_point) != numDims:
                return self.fail(grades, result, solution)

            solution_point = solution[0]
            error = np.array(result_point) - np.array(solution_point)
            errorNorm = np.linalg.norm(error)
            if errorNorm > self.tolerance:
                return self.fail(grades, result, solution)

            result_obj = result[1]
            solution_obj = solution[1]
            if np.abs(result_obj - solution_obj) > self.tolerance:
                return self.fail(grades, result, solution)
                
        # Otherwise pass
        grades.addMessage('PASS: ' + self.path)
        grades.addMessage('\t' + self.success)
        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(self.partialPoints, self.partialPoints))
            grades.addPoints(self.partialPoints)
        return True

    def fail(self, grades, result, solution):
        grades.addMessage('FAIL: ' + self.path)
        grades.addMessage('\t' + self.failure)
        grades.addMessage('\tstudent result: "{}"'.format(result))
        grades.addMessage('\tcorrect result: "{}"'.format(solution))

        if self.partialPoints > 0:
            print("                    ({} of {} points)".format(0, self.partialPoints))

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "{}"\n'.format(output))
        # print '>>>>>>>>>>>>>>>>>>result:"{}"\n'.format(output)
        handle.write('result:"{}"\n'.format(output))
        handle.close()
        return True



class EvalAgentTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalAgentTest, self).__init__(question, testDict)
        self.layoutName = testDict['layoutName']
        self.agentName = testDict['agentName']
        self.ghosts = eval(testDict['ghosts'])
        self.maxTime = int(testDict['maxTime'])
        self.seed = int(testDict['randomSeed'])
        self.numGames = int(testDict['numGames'])
        self.numTraining = int(testDict['numTraining'])

        self.scoreMinimum = int(testDict['scoreMinimum']) if 'scoreMinimum' in testDict else None
        self.nonTimeoutMinimum = int(testDict['nonTimeoutMinimum']) if 'nonTimeoutMinimum' in testDict else None
        self.winsMinimum = int(testDict['winsMinimum']) if 'winsMinimum' in testDict else None

        self.scoreThresholds = [int(s) for s in testDict.get('scoreThresholds','').split()]
        self.nonTimeoutThresholds = [int(s) for s in testDict.get('nonTimeoutThresholds','').split()]
        self.winsThresholds = [int(s) for s in testDict.get('winsThresholds','').split()]

        self.maxPoints = sum([len(t) for t in [self.scoreThresholds, self.nonTimeoutThresholds, self.winsThresholds]])
        self.agentArgs = testDict.get('agentArgs', '')

    def execute(self, grades, moduleDict, solutionDict):
        startTime = time.time()

        agentType = getattr(moduleDict['qlearningAgents'], self.agentName)
        agentOpts = pacman.parseAgentArgs(self.agentArgs) if self.agentArgs != '' else {}
        agent = agentType(**agentOpts)

        lay = layout.getLayout(self.layoutName, 3)

        disp = self.question.getDisplay()

        random.seed(self.seed)
        games = pacman.runGames(lay, agent, self.ghosts, disp, self.numGames, False, numTraining=self.numTraining, catchExceptions=True, timeout=self.maxTime)
        totalTime = time.time() - startTime

        stats = {'time': totalTime, 'wins': [g.state.isWin() for g in games].count(True),
                 'games': games, 'scores': [g.state.getScore() for g in games],
                 'timeouts': [g.agentTimeout for g in games].count(True), 'crashes': [g.agentCrashed for g in games].count(True)}

        averageScore = sum(stats['scores']) / float(len(stats['scores']))
        nonTimeouts = self.numGames - stats['timeouts']
        wins = stats['wins']

        def gradeThreshold(value, minimum, thresholds, name):
            points = 0
            passed = (minimum == None) or (value >= minimum)
            if passed:
                for t in thresholds:
                    if value >= t:
                        points += 1
            return (passed, points, value, minimum, thresholds, name)

        results = [gradeThreshold(averageScore, self.scoreMinimum, self.scoreThresholds, "average score"),
                   gradeThreshold(nonTimeouts, self.nonTimeoutMinimum, self.nonTimeoutThresholds, "games not timed out"),
                   gradeThreshold(wins, self.winsMinimum, self.winsThresholds, "wins")]

        totalPoints = 0
        for passed, points, value, minimum, thresholds, name in results:
            if minimum == None and len(thresholds)==0:
                continue

            # print passed, points, value, minimum, thresholds, name
            totalPoints += points
            if not passed:
                assert points == 0
                self.addMessage("{} {} (fail: below minimum value {})".format(value, name, minimum))
            else:
                self.addMessage("{} {} ({} of {} points)".format(value, name, points, len(thresholds)))

            if minimum != None:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < {}:  fail".format(minimum))
                if len(thresholds)==0 or minimum != thresholds[0]:
                    self.addMessage("    >= {}:  0 points".format(minimum))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= {}:  {} points".format(threshold, idx+1))
            elif len(thresholds) > 0:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < {}:  0 points".format(thresholds[0]))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= {}:  {} points".format(threshold, idx+1))

        if any([not passed for passed, _, _, _, _, _ in results]):
            totalPoints = 0

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for {}.\n'.format(self.path))
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True



