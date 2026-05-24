class UniqueColoring:

    def __init__(self):
      self.number = dict()
      self.nextNum = 0
    
    def getNum(self, state):
      if state not in self.number:
        self.number[state] = self.nextNum
        self.nextNum += 1
    
      return self.number[state]

class Task:
    def __init__(self, task_type, stimuli_type, action_type):
        self.task_type = task_type
        self.stimuli_type = stimuli_type
        self.action_type = action_type
        self.nextState = dict()
        self.action = dict()
        self.initial = None
        self.nth = None
        
    def states(self):
        return self.nextState.keys()
      
    def add_state(self, state):
        self.nextState[str(state)] = dict()
        self.action[str(state)] = dict()
        
    def set_initial(self, state):
        self.initial = str(state)
        self.add_state(str(state))
      
    def add_multipleStates(self, seqState: list):
        for state in seqState:
          self.add_state(str(state))

    def add_edge(self, currState, stimuli, action, nextState):
        self.nextState[str(currState)][str(stimuli)] = nextState
        self.action[str(currState)][str(stimuli)] = action
        
    def printDiagram(self):
        for state in self.states():
          print("State ", state, " goes to: ", self.nextState[state], ", with action: ", self.action[state])
     
    def seqActions(self, stimuli):
        curr_state = self.initial
        action = []
        state = []
        
        assert curr_state is not None
        
        for stimulus in stimuli:
          action.append(self.action[str(curr_state)][str(stimulus)])
          curr_state = self.nextState[str(curr_state)][str(stimulus)]
          state.append(curr_state)
        
        return action, state
    
    def minimalColoring(self):
        stateNum = dict()
        coloring = UniqueColoring()
          
        for state in self.states():
          pairs = tuple(sorted((stimulus, action) for stimulus, action in self.action[state].items())) # combination of (stimulus, action) pairs
          stateNum[state] = coloring.getNum(pairs) # each state gets different number if they have different combinations
          
        while True:
          newStateNum = dict()
          coloring = UniqueColoring()
          
          for state in self.states():
            # combination of (current state type, stimulus, next state type)
            pairs = tuple(sorted((stateNum[state], stimulus, stateNum[nextState]) for stimulus, nextState in self.nextState[state].items()))
            newStateNum[state] = coloring.getNum(pairs)
          
          # iterate until it can't be shrinked anymore
          if newStateNum == stateNum:
            break
          else:
            stateNum = newStateNum
          
        return stateNum

    def coloringToCognitiveDiagram(self, stateNum: dict):
    
        cd_temp = Task(self.task_type, self.stimuli_type, self.action_type)
        cd_temp.add_multipleStates(list(set(stateNum.values())))
        cd_temp.set_initial(stateNum[self.initial])
          
        pairChecked = set()
          
        for state in self.states():
          for stimulus, nextState in self.nextState[state].items():
            pair = (stateNum[state], stimulus)
          
            if pair not in pairChecked:
              cd_temp.add_edge(stateNum[state], stimulus, self.action[state][stimulus], stateNum[nextState])
              pairChecked.add(pair)
          
        return cd_temp
    
    def to_dict(self) -> dict:
        
        task_dict = {}
        for k, v in self.__dict__.items():
            if callable(v):
                continue
            task_dict[k] = v
            
        return task_dict
    
    @classmethod
    def from_dict(cls, task_dict: dict) -> "Task":
        
        obj = cls(task_dict["task_type"], task_dict["stimuli_type"], task_dict["action_type"])
        obj.__dict__.update(task_dict)
        
        return obj
    