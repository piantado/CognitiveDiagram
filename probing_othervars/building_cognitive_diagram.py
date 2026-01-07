# -*- coding: utf-8 -*-

import cognitive_diagram as cd
from types import MethodType
import itertools


def ordinal(n):
    if n == 1:
        return ""
    elif 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} "

def build_diagram_WCST():
    
    # task variables
    stimuli_type = ['NO', 'NX', 'SO', 'SX', 'CO', 'CX']
    action_type = list({stimulus[0] for stimulus in stimuli_type})
    
    # build a diagram
    task = cd.Task("WCST", stimuli_type, action_type)
    task.set_initial("".join(sorted(action_type)))

    # add states
    for comb_len in range(1, len(action_type) + 1):
        for comb in itertools.combinations(action_type, comb_len):
            task.add_state("".join(sorted(comb)))

    # add edges
    for currState in task.states():
        for stimulus in stimuli_type:
            rule = stimulus[:-1]
            
            if stimulus[-1] == "O":
                nextState = rule
                action = rule
            elif rule not in currState:
                nextState = currState
                action = currState[0]
            elif len(currState) == 1:
                nextState = "".join(sorted(action_type))
                action = action_type[0]
            else:
                nextState = currState.replace(rule, "")
                action = nextState[0]

            task.add_edge(currState, stimulus, action, nextState)
            
    print(len(task.states()))
    
    # make it minimal
    task_coloring = task.minimalColoring()
    task_minimal = task.coloringToCognitiveDiagram(task_coloring)
    
    print(len(task_minimal.states()))
    
    # define functional variables
    task_minimal.othervars = ["last correct rule"]
    
    def generate_func_vars(self, samples):
        
        func_vars = dict()
        for othervar in self.othervars:
            func_vars[othervar] = []
            
        for sample in samples:
            last_correct = next((stimulus for stimulus in reversed(sample) if stimulus[-1] == "O"), None)
            func_vars[self.othervars[0]].append(last_correct)
        
        return func_vars
    
    task_minimal.generate_func_vars = MethodType(generate_func_vars, task_minimal)
    
    return task_minimal

def build_diagram_MaxNum():
    
    # task variables
    stimuli_type = [str(i) for i in range(1, 10)]   # if include zero or negatives, results in error at sorted(comb)
    nth = 2     # nth biggest digit in the sequence
    action_type = [str(i) for i in range(1, 10)]    # different than the actual actions that appear
    
    # build a diagram
    task = cd.Task("MaxNum", stimuli_type, action_type)
    task.nth = nth
    task.set_initial("0" * nth)   # if leave as a blank, results in error at sorted(...)[1:]
    
    # add states
    for comb_len in range(1, nth + 1):
        for comb in itertools.combinations_with_replacement(stimuli_type, comb_len):
            task.add_state("0" * (nth - comb_len) + "".join(sorted(comb)))        
    
    # add edges
    for currState in task.states():
        for stimulus in stimuli_type:
            nextState = "".join(sorted(currState + stimulus)[1:])
            action = nextState[0]
            task.add_edge(currState, stimulus, action, nextState)
            
    print(len(task.states()))
    
    # make it minimal
    task_coloring = task.minimalColoring()
    task_minimal = task.coloringToCognitiveDiagram(task_coloring)
    
    print(len(task_minimal.states()))
    
    # define functional variables
    assert (nth - 1) > 0
    task_minimal.othervars = [f"{ordinal(nth)}largest", f"{ordinal(nth - 1)}largest"]
    
    def generate_func_vars(self, samples):
        
        func_vars = dict()
        for othervar in self.othervars:
            func_vars[othervar] = []
            
        for sample in samples:
            sample_sorted = "".join(sorted(sample))
            func_vars[self.othervars[0]].append(sample_sorted[-nth])
            func_vars[self.othervars[1]].append(sample_sorted[-(nth - 1)])
        
        return func_vars
    
    task_minimal.generate_func_vars = MethodType(generate_func_vars, task_minimal)
    
    return task_minimal
    
def build_diagram_Nback():
    
    # task variables
    stimuli_type = [str(i) for i in range(1, 10)]
    nth = 2
    action_type = [str(i) for i in range(1, 10)]
    
    # build a diagram
    task = cd.Task("Nback", stimuli_type, action_type)
    task.nth = nth
    task.set_initial("0" * nth)
    
    # add states
    for comb_len in range(1, nth + 1):
        for comb in itertools.product(stimuli_type, repeat = comb_len):
            task.add_state("0" * (nth - comb_len) + "".join(comb))
              
    # add edges
    for currState in task.states():
        for stimulus in stimuli_type:
            nextState = currState[1:] + stimulus
            action = currState[0]
            task.add_edge(currState, stimulus, action, nextState)
    
    print(len(task.states()))

    # make it minimal
    task_coloring = task.minimalColoring()
    task_minimal = task.coloringToCognitiveDiagram(task_coloring)
    
    print(len(task_minimal.states()))
    
    # define functional variables
    assert (nth - 1) >= 0
    task_minimal.othervars = [f"{nth} positions back", f"{nth - 1} positions back"]
    
    def generate_func_vars(self, samples):
        
        func_vars = dict()
        for othervar in self.othervars:
            func_vars[othervar] = []
            
        for sample in samples:
            func_vars[self.othervars[0]].append(sample[-nth])
            func_vars[self.othervars[1]].append(sample[-(nth - 1)])
        
        return func_vars
    
    task_minimal.generate_func_vars = MethodType(generate_func_vars, task_minimal)
    
    return task_minimal