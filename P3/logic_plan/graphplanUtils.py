# graphplanUtils.py
# -----------------
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


# Translation of GraphPlan algorithm (Blum & Furst).
# Originally implemented in Ruby by David S. Touretzky, October 2012
# Translated to Python by Jonathan Li and further extended by
# David S. Touretzky, October 2018

### Constants for built-in types, predicates, and actions
INT = 'int'

EQUAL = 'equal'
NOT_EQUAL = 'not_equal'
LESS_THAN = 'less_than'
LESS_EQUAL = 'less_equal'
SUM = 'sum'

NOOP = 'noop'

# To Do:  check that goals contain no variables
# Rename demo files, e.g., tire-problem.py, blocks-problems.py

def safeIndex(l, f):
    '''
    let l be a list of items [x_i | 0 <= i <= len(l)]

    safeIndex(l,f) returns the index of the first x_i
    such that f(x_i) == True

    if such an element does not exist, safeIndex returns None
    '''

    for i in range(len(l)):
        if f(l[i]):
            return i
    return None

class Instance:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return '<Instance %s: %s>' % (self.name, self.kind)

    def __eq__(self, other):
        return (
            isinstance(other, Instance) and
            self.name == other.name and
            self.kind == other.kind
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Defining __eq__ causes a class to be unhashable by default,
        must explicitly re-define __hash__ if __eq__ is defined
        '''
        return hash(str(self))

class Variable:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind

    def __str__(self):
        return '?' + str(self.name)

    def __repr__(self):
        return '<Variable %s: %s>' % (self.name, self.kind)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.name == other.name and self.kind == other.kind


def Proposition(pred, *args):
    return UniqueProposition.new(pred, args, True)

def NotProposition(pred, *args):
    return UniqueProposition.new(pred, args, False)

class UniqueProposition:
    prophash = dict()

    def __init__(self, pred, args=[], value=True):
        self.pred = pred
        self.args = args
        self.value = value
        self.propnum = None
        if len([arg for arg in args if isinstance(arg, Variable)]) > 0:
            return    # Term contains variables, so don't hash it
        signature = (pred, tuple(args), value)   # can't hash lists in python!
        UniqueProposition.prophash[signature] = self
        self.propnum = len(UniqueProposition.prophash)

    @classmethod
    def new(cls, pred, args, value=True):
        signature = (pred, tuple(args), value)
        prop = UniqueProposition.prophash.get(signature, None)
        if prop is not None:
            return prop
        else:
            return UniqueProposition(pred, args, value)

    def __repr__(self):
        if self.value:
            sign = ''
        else:
            sign = '~'

        argstring = '('
        argsep = ''

        for arg in self.args:
            argstring += argsep + str(arg)
            argsep = ','

        argstring += ')'
        return sign + str(self.pred) + argstring

    def __eq__(self, other):
        return (
            isinstance(other, UniqueProposition) and
            self.pred == other.pred and
            self.args == other.args and
            self.value == other.value
        )

    def __invert__(self):
        return self.new(self.pred, self.args, not self.value)

    def __hash__(self):
        '''
        Defining __eq__ causes a class to be unhashable by default,
        must explicitly re-define __hash__ if __eq__ is defined
        '''
        return hash(str(self))
    

    def instantiate(self, bindings):
        newargs = [bindings.get(x,x) for x in self.args]
        return UniqueProposition.new(self.pred, newargs, self.value)
    
def validate_terms(terms):
    for term in terms:
        if not isinstance(term, UniqueProposition):
            raise ValueError('Not a Proposition: %s' % term)
        if not isinstance(term.pred, str):
            raise ValueError('Predicate names must be strings, not %s in %s' %
                             (repr(term.pred), repr(term)))
        for arg in term.args:
            if not isinstance(arg, (Variable, Instance)):
                raise ValueError('Arguments must be variables or instances, not %s in %s' %
                                 (repr(arg), repr(term)))

class Operator:
    def __init__(self, name, precs, adds, deletes):
        self.name = name
        self.params = []
        self.precs = precs
        self.adds = adds
        self.deletes = deletes

        self.validate_operator()

    def __str__(self):
        return str(self.name) + "(" + str(self.params) + ")"

    def __repr__(self):
        return str(self)

    def validate_operator(self):
        validate_terms(self.precs)
        validate_terms(self.adds)
        validate_terms(self.deletes)
        params = set()
        for prec in self.precs:
            for arg in prec.args:
                if isinstance(arg, Variable):
                    params.add(arg)
        self.params = params
        for term in self.adds + self.deletes:
            for arg in term.args:
                if isinstance(arg, Variable) and arg not in params:
                    raise ValueError('Variable %s must be bound in the preconditions, in %s' %
                                     (repr(arg), repr(term)))

    def generate_bindings(self, instance_table):
        # Generate a list of dicts for possible variable bindings for
        # this operator.  Along the way, check all preconditions to
        # eliminate as many binding combinations as possible.
        bindings = [{}]
        for param in self.params:
            newbinds = []
            for inst in instance_table[param.kind]:
                bindings_copy = [binding.copy() for binding in bindings]
                for binding in bindings_copy:
                    binding[param] = inst
                bindings_copy = [binding for binding in bindings_copy if self.check_binding(binding)]
                newbinds = newbinds + bindings_copy
            bindings = newbinds
        #print self.name,"#candidate bindings=",len(bindings)
        return bindings

    def check_binding(self, binding):
        # Returns True if the binding is compatible with all precondition tests.
        # If the binding doesn't cover some variables, those tests will be skipped.
        for prec in self.precs:
            if (prec.pred == EQUAL) or (prec.pred == NOT_EQUAL):
                arg1 = prec.args[0]
                arg2 = prec.args[1]
                if arg1 in binding: arg1 = binding[arg1]
                if arg2 in binding: arg2 = binding[arg2]
                if isinstance(arg1, Instance) and isinstance(arg2, Instance):
                    if (arg1 == arg2):
                        if (prec.pred == NOT_EQUAL) == prec.value: return False
                    elif (prec.pred == EQUAL) == prec.value: return False
            elif (prec.pred == LESS_THAN) or (prec.pred == LESS_EQUAL):
                arg1 = prec.args[0]
                arg2 = prec.args[1]
                if arg1 in binding: arg1 = binding[arg1]
                if arg2 in binding: arg2 = binding[arg2]
                if isinstance(arg1, Instance) and isinstance(arg2, Instance):
                    if prec.pred == LESS_THAN:
                        if ((arg1.name < arg2.name) != prec.value): return False
                    elif prec.pred == LESS_EQUAL:
                        if ((arg1.name <= arg2.name) != prec.value): return False
            elif prec.pred == SUM:
                arg1 = prec.args[0]
                arg2 = prec.args[1]
                arg3 = prec.args[2]
                if arg1 in binding: arg1 = binding[arg1]
                if arg2 in binding: arg2 = binding[arg2]
                if arg3 in binding: arg3 = binding[arg3]
                if (isinstance(arg1, Instance) and
                    isinstance(arg2, Instance) and
                    isinstance(arg3, Instance)):
                    if (arg1.name + arg2.name == arg3.name) != prec.value: return False
        return True

    def generate_bindprops(self, propnodes, instance_table):
        bindings = self.generate_bindings(instance_table)
        bindprops = []
        builtins = set([EQUAL, NOT_EQUAL, LESS_THAN, LESS_EQUAL, SUM])
        for bind in bindings:
            props = []
            for prec in self.precs:
                if prec.pred not in builtins:
                    bprec = prec.instantiate(bind)
                    pnodes = [pn for pn in propnodes if pn.prop == bprec]
                    if (len(pnodes) == 0):
                        props = None
                        break
                    else:
                        props.append(pnodes[0])
            bindprops = (bindprops + [BindProp(bind, props)] if props is not None else bindprops)
        return bindprops


class BindProp:
    def __init__(self, bindings, propnodes):
        self.bindings = bindings
        self.propnodes = propnodes

    def __str__(self):
        return 'Bind<' + str(self.bindings) + ';' + str(self.propnodes) + '>'

    def __repr__(self):
        return str(self)

    def preconditions_excluded(self):
        # return the first propNode for which its
        # excluded preconditions are also in self.propnodes
        for pn in self.propnodes:
            for q in pn.excludes:
                if q in self.propnodes:
                    return pn
        return None

class PropNode:
    node_counter = 0

    def __init__(self, prop, level):
        PropNode.node_counter += 1
        self.prop = prop
        self.level = level
        self.propnum = prop.propnum
        self.adders = []
        self.deleters = []
        self.users = []
        self.excludes = []

    def __repr__(self):
        pname = str(self.prop)
        return pname

    def dump_str(self):
        return '%3d:%s' % (self.propnum, str(self))

    def excludes_prop(self, pnode):
        for anode in self.adders:
            for anode2 in pnode.adders:
                if anode2 not in anode.excludes:
                    return None

        return True

    ## defining the comparison operators between propNodes ##
    ## equivalent of Ruby's <=> ##

    def __eq__(self, other):
        if not isinstance(other, PropNode):
            return False
        return self.prop.propnum == other.prop.propnum

    def __ne__(self, other):
        if isinstance(other, PropNode):
            return self.prop.propnum != other.prop.propnum
        return True
        
    def __lt__(self, other):
        return self.prop.propnum < other.prop.propnum

    def __le__(self, other):
        return self.prop.propnum <= other.prop.propnum

    def __gt__(self, other):
        return self.prop.propnum > other.prop.propnum

    def __ge__(self, other):
        return self.prop.propnum >= other.prop.propnum

    def __hash__(self):
        '''
        Defining __eq__ causes a class to be unhashable by default,
        must explicitly re-define __hash__ if __eq__ is defined
        '''
        return hash(str(self))

class ActionNode:
    node_counter = 0

    def __init__(self, name, level, binds, precs):
        self.name = name
        self.level = level
        ActionNode.node_counter += 1
        self.number = ActionNode.node_counter
        self.binds = binds
        self.precs = precs
        self.adds = []
        self.deletes = []
        self.excludes = []

    def __repr__(self):
        if self.name == NOOP:
            return str(self.name) + ' ' + str(self.precs[0])
        else:
            args = '(' + ', '.join(['%s: %s' % (str(var),str(val)) for (var,val) in self.binds.items()]) + ')'
            return str(self.name) + ' ' + args

    def __lt__(self,other):
        return self.name < other.name

    def dump_str(self):
        return '%3d-%s' % (self.number, str(self))

    def print_name(self):
        return self.name

    def deletes_addition(self, anode):
        # determines if the intersection is non-empty
        intersection = [x for x in self.deletes if x in anode.adds]
        return len(intersection) > 0

    def deletes_precondition(self, anode):
        # does deleted prop at level i+1 match precondition prop at level i?
        for d in self.deletes:
            for p in anode.precs:
                if d.prop == p.prop:
                    return True
        return False

    def competing_needs(self, anode):
        for p in self.precs:
            for p2 in anode.precs:
                if p2 in p.excludes:
                    return True
        return False

class GraphPlanProblem:
    def __init__(self, name, instances, operators, initial, goals):
        self.name = name
        self.instances = instances
        self.operators = operators
        self.initial = initial
        self.goals = goals
        self.propLevels = []
        self.actionLevels = []
        self.level = -1
        self.unsolvable_goalsets = [dict()]
        self.result = None

        self.generate_instance_table()
        self.populate_prop_level_0()

        validate_terms(initial)
        validate_terms(goals)

    def generate_instance_table(self):
        self.instance_table = dict()
        for inst in self.instances:
            if inst.kind not in self.instance_table:
                self.instance_table[inst.kind] = []
            if inst not in self.instance_table[inst.kind]:
                self.instance_table[inst.kind].append(inst)

    def populate_prop_level_0(self):
        self.propLevels = [[]]
        for prop in self.initial:
            self.propLevels[0].append(PropNode(prop, 0))

    def extend_graph_level(self):
        # Generate next levels of action and proposition nodes
        self.actionLevels.append([])
        self.propLevels.append([])
        self.unsolvable_goalsets.append(dict())
        self.level += 1
        print("Generating level", self.level, '-> level', self.level+1)

        self.generate_noop_actions()

        for op in self.operators:
            self.generate_operator_actions(op)

        self.generate_action_excludes_links()
        self.generate_proposition_excludes_links()

    def generate_noop_actions(self):
        for pnode in self.propLevels[-2]:
            anode = ActionNode(NOOP, pnode.level, dict(), [pnode])
            pnode.users.append(anode)
            pnode2 = PropNode(pnode.prop, 1 + pnode.level)
            anode.adds.append(pnode2)
            pnode2.adders.append(anode)
            self.propLevels[-1].append(pnode2)
            self.actionLevels[-1].append(anode)

    def generate_operator_actions(self, op):
        bindprops = op.generate_bindprops(self.propLevels[-2], self.instance_table)
        # print op.name, "#bindprops=", len(bindprops)
        bindprops = [bp for bp in bindprops
                     if bp.preconditions_excluded() is None]
        self.actionLevels[-1] += [self.generate_action_node(op,bp)
                                  for bp in bindprops]

    def generate_action_node(self, op, bp):
        # op = operator, bp = bindProp
        anode = ActionNode(op.name, self.level, bp.bindings, bp.propnodes)
        for pnode in bp.propnodes:
            pnode.users.append(anode)

        anode.adds = [
            self.get_propnode(prop.instantiate(bp.bindings), self.level+1) for prop in op.adds
        ]
        for pnode in anode.adds:
            pnode.adders.append(anode)

        anode.deletes = [
            self.get_propnode(prop.instantiate(bp.bindings), self.level+1) for prop in op.deletes
        ]
        for pnode in anode.deletes:
            pnode.deleters.append(anode)
            
        return anode

    def generate_action_excludes_links(self):
        for anode in self.actionLevels[-1]:
            for bnode in self.actionLevels[-1]:
                if anode != bnode:
                    if (
                            anode.deletes_addition(bnode) or
                            anode.deletes_precondition(bnode) or
                            anode.competing_needs(bnode)
                    ):
                        if (bnode not in anode.excludes):
                            anode.excludes.append(bnode)
                        if (anode not in bnode.excludes):
                            bnode.excludes.append(anode)

    def generate_proposition_excludes_links(self):
        for pnode in self.propLevels[-1]:
            for pnode2 in self.propLevels[-1]:
                if pnode != pnode2 and pnode.excludes_prop(pnode2):
                    pnode.excludes.append(pnode2)
    

    def leveled_off(self):
        '''
        The graph has leveled off if the last level contains no new
        propositions, and the same number of exclusion relationships as the
        level before it
        '''
        if len(self.propLevels) < 2:
            return None

        if len(self.propLevels[-1]) != len(self.propLevels[-2]):
            return None

        this_excludes = 0
        for prop in self.propLevels[-1]:
            this_excludes += len(prop.excludes)
            
        prev_excludes = 0
        for prop in self.propLevels[-2]:
            prev_excludes += len(prop.excludes)

        if this_excludes != prev_excludes:
            return None

        return True

    def get_existing_propnode(self, prop, level):
        for pnode in self.propLevels[level]:
            if pnode.prop == prop:
                return pnode
        return None

    def get_propnode(self, prop, level):
        pnode = self.get_existing_propnode(prop, level)
        if pnode is not None:
            return pnode
        pnode = PropNode(prop, level)
        self.propLevels[level].append(pnode)
        return pnode

    def solve(self):
        print('Solving', self.name, '...')
        PropNode.node_counter = 0
        ActionNode.node_counter = 0
        if self.level != -1:
            raise ValueError('Problem has already been solved.')
        while not self.terminate():
            self.extend_graph_level()
            goals = [self.get_existing_propnode(g, self.level+1)
                     for g in self.goals]
            if None in goals:
                print('Some goal props not instantiated at level %d' % \
                      (self.level+1))
            elif self.mutually_exclusive(goals):
                print('Some goals are mutually exclusive at level %d' % \
                      (self.level+1))
            else:
                result = self.solve_goals(goals, [], goals, [])
                if result is not None:
                    print('Generated %d proposition nodes and %d action nodes total.' % \
                          (PropNode.node_counter, ActionNode.node_counter))
                    return result
                else:
                    print('No solution at level %d with %d proposition nodes and %d action nodes' % \
                        (self.level + 1, PropNode.node_counter, ActionNode.node_counter))
        print('Problem has no solution!  Generated %d proposition nodes and %d action nodes total.' % \
              (PropNode.node_counter, ActionNode.node_counter))
        print(goals)
        return None

    '''
    solve_goals will recurse on the tail of the goals_remaining, but we
    still need to keep the complete goal set around for the 
    minimal_action_set test
    '''

    def solve_goals(self, goals, goals_remaining, new_goals, selected_actions):
        # time to advance to the next level?
        if len(goals_remaining) == 0:
            if new_goals[0].level == 0:
                print('Solution found at level %d' % (self.level + 1))
                return selected_actions
            else:
                new_goals.sort()
                new_goals = tuple(new_goals) # needs to be a hashable type!
                if self.unsolvable_goalsets[new_goals[0].level].get(new_goals,None):
                    #print "Found unsolvable: ", new_goals
                    return None
                result = self.solve_goals(new_goals, new_goals,
                                     [], [[]] + selected_actions)
                if result is not None:
                    return result
                else:
                    # no action solves this goal set, so mark as unsolvable
                    #print "--> Added unsolvable:", new_goals
                    self.unsolvable_goalsets[new_goals[0].level][new_goals] = True
                    return None

        # Work on the next goal at the current level
        goal = goals_remaining[0]
        actions = goal.adders
        for a in actions:
            # Make sure this action isn't excluded by any actions already selected
            if safeIndex(selected_actions[0], lambda s: a in s.excludes) is None:
                new_actions = [selected_actions[0] + [a]] + selected_actions[1:]
                if ((not self.forward_cutoff(goals_remaining[1:], new_actions[0]))
                    and self.minimal_action_set(goals, new_actions[0])):
                    next_goals_remaining = [g for g in goals_remaining
                                            if g not in a.adds]
                    next_new_goals = new_goals + [g for g in a.precs if
                                                  g not in new_goals]
                    result = self.solve_goals(goals, next_goals_remaining,
                                              next_new_goals, new_actions)
                    if result is not None:
                        self.result = result
                        return result

        return None

    def mutually_exclusive(self, goals):
        '''
        Return True if any two goals are mutually exclusive
        '''
        for i in range(len(goals)):
            g1 = goals[i]
            for j in range(i+1, len(goals)):
                g2 = goals[j]
                if g2 in g1.excludes:
                    return True
        return False

    def forward_cutoff(self, goals, actions):
        '''
        Return True if some goals has been cut off by this aciton set,
        i.e. every action that could achieve it has been excluded
        '''
        for g in goals:
            possible_achievers = g.adders
            for a in actions:
                possible_achievers = [
                    a for a in possible_achievers
                    if a not in a.excludes
                ]
                if len(possible_achievers) == 0:
                    return True
        return False

    def minimal_action_set(self, goals, actions):
        '''
        Return False if removing any one action still satisfies all the goals.
        This can happen if actions have overlapping effects
        '''
        for a in actions:
            to_be_achieved = goals
            reduced_actions = [a2 for a2 in actions if a2 != a]
            for a2 in reduced_actions:
                to_be_achieved = [a3 for a3 in to_be_achieved
                                  if a3 not in a2.adds]
                if len(to_be_achieved) == 0:
                    print('Non-minimal action set:', actions)
                    return False
        return True

    def terminate(self):
        if not self.leveled_off():
            return False
        elif (len(self.unsolvable_goalsets[-1]) !=
              len(self.unsolvable_goalsets[-2])):
            
            return False
        return True

    def display(self):
        if self.result:
            print("Plan found:")
            for level in self.result:
                for action in level:
                    if action.name != NOOP:
                        print('%3d:' % action.level, action)
        else:
            print("No plan found.")

    def getactions(self):
        actions = []
        if self.result:
            for level in self.result:
                for action in level:
                    if action.name != NOOP:
                        actions.append(action)
            return actions
        else:
            return []

    def dump(self):
        print("Dump of planning graph:")
        for level in range(len(self.propLevels)):
            print(('*'*16) + (' Level %d ' % level) + ('*'*16))
            print('Proposition nodes: (' + str(len(self.propLevels[level])) + ')')
            self.propLevels[level].sort()
            for p in self.propLevels[level]:
                pname = p.dump_str()
                print(' ' + pname),
                if (len(p.excludes) > 0):
                    p.excludes.sort()
                    print('excl (%d)'  % len(p.excludes)),
                    for e in p.excludes:
                        print('%d' % (e.prop.propnum)),
                print()

            if level < len(self.actionLevels):
                print('Action nodes: (%d)' % len(self.actionLevels[level]))
                for a in self.actionLevels[level]:
                    aname = a.dump_str()
                    print(' ' + aname),
                    if len(a.excludes) > 0:
                        a.excludes.sort()
                        print('excl (%d):' % len(a.excludes)),
                        for e in a.excludes:
                            print('%d' % e.number),
                    print()
            print()
        return None
