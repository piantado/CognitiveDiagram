import cognitive_diagram as cd
from types import MethodType
import itertools
import numpy as np

def build_cognitive_diagram(task_type, if_minimal = False):
    
    if task_type == "WCST":

        # task_variables
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


    elif task_type == "MaxNum":

        # task variables
        nth = 1     # nth biggest digit in the sequence
        stimuli_type = [str(i) for i in range(1, 10)]   # if include zero or negatives, results in error at sorted(comb)
        action_type = [str(i) for i in range(0, 10)]    # different than the actual actions that appear
    
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


    elif task_type == "Nback":

        # task variables
        nth = 3
        stimuli_type = [str(i) for i in range(1, nth + 1)]
        action_type = [str(0), str(1)]
    
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
                action = str(0) if currState[0] == stimulus else str(1) 
                task.add_edge(currState, stimulus, action, nextState)
    
        
    elif task_type == "FiniteSet":

        element_type = [str(i) for i in range(1, 4)]
        stimuli_type = [stim for elem in element_type for stim in [str(elem), "r" + str(elem)]]
        action_type = element_type + ["0"]
    
        # build a diagram
        task = cd.Task("FiniteSet", stimuli_type, action_type)
        task.set_initial("")
    
        # add states
        for comb_len in range(1, len(element_type) + 1):
            for comb in itertools.combinations(element_type, comb_len):
                task.add_state("".join(sorted(comb)))

        # add edges
        for currState in task.states():
            for stimulus in stimuli_type:

                if stimulus.startswith("r"):
                    nextState = currState.replace(stimulus[1:], "")
                    action = stimulus[1:] if stimulus[1:] in currState else "0"
                else:
                    stim_set = set(currState)
                    stim_set.add(stimulus)
                    nextState = ("".join(sorted(stim_set)))
                    action = "0"
        
                task.add_edge(currState, stimulus, action, nextState)


    elif task_type == "FiniteList":

        stimuli_type = set()
        element_type = [str(i) for i in range(1, 3)]
        action_type = element_type + ["0", "-"]

        for mode in ["r", "w"]:
            for elem in element_type:

                elem = "-" if mode == "r" else elem

                for pos in range(0, len(element_type)):
                    stimuli_type.add(mode + elem + str(pos))

        stimuli_type = list(stimuli_type)
    
        # build a diagram
        task = cd.Task("FiniteList", stimuli_type, action_type)
        task.set_initial("-" * len(element_type))
    
        # add states
        action_type_with_null = element_type + ["-"]
        states = list(itertools.product(action_type_with_null, repeat = len(element_type)))
        for state in states:
            task.add_state("".join(state))
              
        # add edges
        for currState in task.states():
            for stimulus in stimuli_type:

                if stimulus.startswith("r"):
                    nextState = currState
                    action = currState[int(stimulus[-1])]
                else:
                    nextState = list(currState)
                    nextState[int(stimulus[-1])] = stimulus[1]
                    nextState = "".join(nextState)
                    action = "0"
        
                task.add_edge(currState, stimulus, action, nextState)


    elif task_type == "FiniteStack":

        max_depth = 3
        element_type = [str(i) for i in range(1, 3)]
        stimuli_type = ["p" + elem for elem in element_type] + ["pop"]
        action_type = element_type + ["0"]

        # build a diagram
        task = cd.Task("FiniteStack", stimuli_type, action_type)
        task.set_initial("")

        # add states
        for depth in range(1, max_depth + 1):
            for state in itertools.product(element_type, repeat = depth):
                task.add_state("".join(state))
        task.add_state("E")
    
        # add edges
        for currState in task.states():
            for stimulus in stimuli_type:

                if currState == "E":
                    nextState = "E"
                    action = "0"
                elif stimulus.startswith("p"):
                    if len(currState) >= max_depth:
                        nextState = "E"
                        action = "0"
                    else:
                        nextState = currState + stimulus[-1]
                        action = "0"
                elif stimulus == "pop":
                    if len(currState) < 1:
                        nextState = currState
                        action = "0"
                    else:
                        nextState = currState[:-1]
                        action = currState[-1]

                task.add_edge(currState, stimulus, action, nextState)

    # elif  # add task type
    # elif
    # elif
                
    # make it minimal
    if if_minimal:
        task_coloring = task.minimalColoring()
        task_minimal = task.coloringToCognitiveDiagram(task_coloring)
        return task_minimal
    else:
        return task

