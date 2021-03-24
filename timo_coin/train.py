import pickle
import os
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .callbacks import features_to_state_number
from .callbacks import state_finder

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EXTRA_REWARD = 0
REWARD_DISTANCE = 15
# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
alpha = 0.5
gamma = 0.6
REPETION_SEARCH = 5


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.repition = deque(maxlen=3)
    #self.samefeature = deque(maxlen=REPETION_SEARCH)
    self.states = np.array([])
    self.coins_collected = REPETION_SEARCH
    max_distance = 3  # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES, TAMBIEN EN CALLBACKS
    possible_outcomes = max_distance * 2 + 1
    q_values = np.ones((16*possible_outcomes**2 * max_distance*2 + 16*4*30 + 1, 6))
    #q_values = np.ones((201411,6))
    q_values[:, 5] = -40000
    q_values[:, 4] = -40000
    if self.train and os.path.isfile("q_values.npy"):
        #np_load_old = np.load
        # modify the default parameters of np.load
        #np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        # call load_data with allow_pickle implicitly set to true
        q_values = np.load("q_values.npy")
        print("File was loaded")
        # restore np.load for future normal usage
        #np.load = np_load_old
    self.q_values = q_values

def symmetry_up_down(features):
    u, d, l, r, m1x, m1y, n, near, distance, state = features
    if n == 1:
        n = 4
    if n ==4:
        n = 1
    if n ==2:
        n = 3
    if n == 3:
        n = 2
    return (d, u, l, r, m1x, -m1y, n, near, distance, state)

def action_symmetry_up_down(action):
    if action == 0:
        action = 2
    if action == 2:
        action = 0
    return action

def symmetry_left_right(features):
    u, d, l, r, m1x, m1y, n, near, distance, state = features
    if n == 1:
        n = 2
    if n ==2:
        n = 1
    if n ==4:
        n = 3
    if n == 3:
        n = 4
    return (u, d, r, l, -m1x, m1y, n, near, distance, state)

def action_symmetry_left_right(action):
    if action ==1:
        action = 3
    if action ==3:
        action = 1
    return action

def symmetry_x_y(features):
    u, d, l, r, m1x, m1y, n, near, distance, state = features
    return (l, r, u, d, m1y, m1x, n, near, distance, state)
def action_symmetry_x_y(action):
    if action ==2:
        action = 1
    if action ==1:
        action = 2
    if action ==0:
        action = 3
    if action ==0:
        action = 3
    return action

def use_symmetry(features, action, next_features, reward, self):
    if features is None or next_features is None:
        return
    u, d, l, r, m1x, m1y, n, near, distance, state = features
    u_next, d_next, l_next, r_next, m1x_next, m1y_next, n_next, near_next, distance_next, state_next = next_features
    if state == False or state_next == False or action == None:
        return
    old_value = self.q_values[features_to_state_number(features), action]
    print("Old Value: " + str(old_value))
    print(action)
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(next_features)])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        print("New Value: " + str(new_value))
        self.q_values[features_to_state_number(features), action] = new_value

    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    #(d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    old_value = self.q_values[state_number_changed, action_changed ]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 3, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 4, interchange of x and y:
    if m1x != m1y and (n == 0):
        features = symmetry_x_y(features) #Ultima vez que usamos todos estos valores, entonces podemos cambiarlo por facilidad
        next_features = symmetry_x_y(next_features)
        state_number_changed = features_to_state_number(features)
        # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
        action = action_symmetry_x_y(action)
        old_value = self.q_values[state_number_changed, action]
        if type(old_value) == type(np.float64(0.3)):
            next_max = np.max(self.q_values[features_to_state_number(next_features)])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            self.q_values[state_number_changed, action] = new_value

        # Symmetry 1, interchange of up and down
        state_number_changed = features_to_state_number(symmetry_up_down(features))
        # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
        action_changed = action_symmetry_up_down(action)
        old_value = self.q_values[state_number_changed, action_changed]
        if type(old_value) == type(np.float64(0.3)):
            next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            self.q_values[state_number_changed, action_changed] = new_value

        # Symmetry 2, interchange of lef and right:
        state_number_changed = features_to_state_number(symmetry_left_right(features))
        # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
        action_changed = action_symmetry_left_right(action)
        old_value = self.q_values[state_number_changed, action_changed]
        if type(old_value) == type(np.float64(0.3)):
            next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            self.q_values[state_number_changed, action_changed] = new_value

        # Symmetry 3, interchange of lef and right:
        state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
        # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
        action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
        old_value = self.q_values[state_number_changed, action_changed]
        if type(old_value) == type(np.float64(0.3)):
            next_max = np.max(
                self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            self.q_values[state_number_changed, action_changed] = new_value


def action_number(action):
    if action == "UP":
        action = 0
    if action == "RIGHT":
        action = 1
    if action == "DOWN":
        action = 2
    if action == "LEFT":
        action = 3
    if action == "WAIT":
        action = 4
    if action == "BOMB":
        action = 5
    return action
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if self_action != None:
        #self.states  = np.append(state_finder(old_game_state))
        #if len(self.states)==5:
            #values, counts = np.unique(self.states, return_counts=True)
        #self.samefeature.append()
        self.transitions.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))
        self.repition.append(state_to_features(old_game_state))
        repition = True
        print(state_finder(old_game_state))

        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
        #print(events)
        features = state_to_features(old_game_state)
        next_features = state_to_features(new_game_state)
        u_old, d_old, l_old, r_old, m1x_old, m1y_old, n_old, near_old, distance_old, state_old = features
        u_new, d_new, l_new, r_new, m1x_new, m1y_new, n_new, near_new, distance_new, state_new = next_features
        next_state = features_to_state_number(next_features)
        state = features_to_state_number(features)

        numpy_events = np.array(events)
        """if np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
            print("coin was collected")
            self.coins_collected = self.coins_collected + 1
            self.steps = 0"""


        ## AQUI ESTABA EL CODIGO PARA LA DISTANCIA
        coin_colected = False
        if next_state != None and state != None:
            next_max = np.max(self.q_values[next_state])
            action = self_action
            action = action_number(action)
            #old_value = self.q_values[state, action]
            if (distance_old<distance_new) and not np.any(np.isin(numpy_events, 'COIN_COLLECTED')): #or (distance_old==distance_new) ; and distance_old>3
                reward = -REWARD_DISTANCE
                #new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                use_symmetry(features, action, next_features, reward, self)
                print("Agent got farther of coin")
            if distance_old>distance_new and not np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
                reward = REWARD_DISTANCE
                #new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                use_symmetry(features, action, next_features, reward, self)
                print("Agent got closer of coin")

        if np.any(np.isin(numpy_events, 'INVALID_ACTION')) and self.transitions:
            if len(self.repition)==1:
                repition = False
            self.repition.pop()
            last_transition = self.transitions.pop()

            features, action, next_features, reward = last_transition
            next_state  = features_to_state_number(next_features)
            state = features_to_state_number(features)
            reward = reward - 50
            if state != None and next_state != None and action != None:
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
                """next_max = np.max(self.q_values[next_state])
                old_value = self.q_values[state, action]
                if type(old_value)== type(np.float64(0.3)):
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    u, d, l, r, m1x, m1y, n, near, distance, state = features
                    
                    use_symmetry(u, d, l, r, m1x, m1y, n, near, distance, state, action, new_value, self)"""

                print("Agent was punished: INVALID MOVE")
                np.save("q_values.npy", self.q_values)
                #print("An invalid move was made and the file was saved")
                self.steps = self.steps - 1
            if state == None:
                print("state = None")
            if next_state == None:
                print("new_state = None")


        #print("Not empty")
        #print("Length of repition deque: " + str(len(self.repition)))
        #print("Length of transition deque: " + str(len(self.transitions)))
        if repition:
            if len(self.repition)==3:
                if self.repition[0]== self.repition[2]:
                    print("Repetion was made #1")
                    while self.repition:
                        self.repition.pop()
                        self.transitions.pop()
            if len(self.repition) == 4:
                if self.repition[0] == self.repition[3]:
                    print("Repetion was made #2")
                    while self.repition:
                        self.repition.pop()
                        self.transitions.pop()
                self.repition.clear()
                #repition = False
                print("Repition was cleared")



        # state_to_features is defined in callbacks.py
        if np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
            coin_colected = True
            while self.transitions:
                features, action, next_features, reward = self.transitions.pop()
                reward = reward + EXTRA_REWARD
                next_state = features_to_state_number(next_features)
                state = features_to_state_number(features)
                if state != None and next_state != None and action !=None:
                    action = action_number(action)
                    use_symmetry(features, action, next_features, reward, self)
                    #next_features = features
                    #features = old_features
                    #before_features, action, features, reward = self.transitions.pop()
                    #action = action_number(action)
                    #use_symmetry(features, action, next_features, reward, self)
                    #next_features = features
                    #features = before_features
                    #use_symmetry(features, action, next_features, reward, self)

            self.transitions.clear()
            self.repition.clear()
            np.save("q_values.npy", self.q_values)
            print("1 Coins were collected and the file was saved")
            self.coins_collected = 0
            self.steps = 0
        """if self.steps >= TRANSITION_HISTORY_SIZE and self.coins_collected == 0:
            features, action, next_features, reward = self.transitions.pop()
            next_state = features_to_state_number(next_features)
            state = features_to_state_number(features)
            reward = reward - 0.01
            if state != None and next_state != None and action !=None:
                while self.transitions:
                    action = action_number(action)
                    next_max = np.max(self.q_values[next_state])
                    old_value = self.q_values[state,action]
                    if type(old_value)== type(np.float64(0.3)):
                        print("Agent was punished: TOO SLOW")
                        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                        u, d, l, r, m1x, m1y,n, near, distance, state = features
                        
                        use_symmetry(u, d, l, r, m1x, m1y, n, near, distance, state, action, new_value, self)

                    features, action, next_features, reward = self.transitions.pop()
                    next_state = features_to_state_number(next_features)
                    state = features_to_state_number(features)
            self.transitions.clear()
            print("deque was cleared")
            np.save("q_values.npy", self.q_values)
            print("0 Coins were collected and the file was saved")
            self.coins_collected = 0
            self.steps = 0"""
    if self_action == None:
        print("Self_action = None")
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_finder(last_game_state), last_action, None, reward_from_events(self, events)))
    end_features = state_to_features(last_game_state)
    #u_end, d_end, l_end, r_end, m1x_end, m1y_end, n_end, near_end, distance_end, state_end = end_features
    end_state = features_to_state_number(end_features)
    action = last_action
    action = action_number(action)
    end_action = action
    end_reward = reward_from_events(self, events)
    if (end_state !=None and end_action !=None) and self.transitions:
        features, action, vorletzter_features, reward = self.transitions.pop()
        if (end_state != None and end_action != None and vorletzter_features != None):
            use_symmetry(vorletzter_features, end_action, end_features, end_reward, self)
        state = features_to_state_number(features)
        action  = action_number(action)
        if state != None and vorletzter_features != None and action != None:
            use_symmetry(features, action, vorletzter_features, reward, self)
        while self.transitions:
            features, action, next_features, reward = self.transitions.pop()
            action = action_number(action)
            use_symmetry(features, action, next_features, reward, self)
    self.transitions.clear()
    print("deque was cleared")
    np.save("q_values.npy", self.q_values)
    print("Game Ended")
    self.transitions.clear()
    self.repition.clear()
    # Store the model
    #with open("my-saved-model.pt", "wb") as file:
     #   pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 25,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.MOVED_DOWN: -.1,
        e.WAITED: -.1,
        e.KILLED_SELF: -100,
        e.INVALID_ACTION:-5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

class MyModel:
    def __init__(self, q_values):
        self.q = q_values
    def propose_action(game_state):
        features = state_finder(game_state)