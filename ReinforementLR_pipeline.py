import copy
import math
from collections import defaultdict
from itertools import product

ids = ["111111111", "222222222"]


DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1









import time
import random
import copy
from collections import defaultdict, deque
from itertools import product

class WizardAgent:
    def __init__(self, initial,
                 alpha=0.05,        # Q-learning step size
                 gamma=0.95,       # Discount factor
                 epsilon=0.2,      # Exploration prob for offline train
                 max_episodes=2000,# Max training episodes
                 train_time_limit=300, # 300s for offline training
                 max_resets=3      # limit on 'reset' usage
                 ):
        """
        A dictionary-based Q-learning Wizard agent with:
          - BFS-based distance features to horcrux & death eaters
          - coverage heuristics
          - partial collision logic
          - large penalty for terminate if horcrux remains
          - limit on reset usage
          - no numpy used

        This version includes checks to avoid NoneType errors for sub-actions.
        """
        self.initial_state = copy.deepcopy(initial)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episodes = max_episodes
        self.train_time_limit = train_time_limit
        self.max_resets = max_resets

        # keep track of resets used in each training episode
        self.resets_used = 0

        # environment info
        self.map = initial["map"]
        self.wizards = initial["wizards"]
        self.horcrux = initial["horcrux"]
        self.death_eaters = initial["death_eaters"]
        self.turns_to_go = initial["turns_to_go"]

        # parse map for passable
        self.grid = {}
        self.reachable = set()
        for r, row in enumerate(self.map):
            for c, val in enumerate(row):
                self.grid[(r,c)] = val
                if val == 'P':
                    self.reachable.add((r,c))

        # dictionary-based linear Q function: weights[feature_name] => float
        self.weights = defaultdict(float)
        self.Q_table={}
        self.last_temprature=None

        # Offline Q-learning
        self.offline_train()

    # --------------------------------------------------------------------------
    # Offline Q-Learning
    # --------------------------------------------------------------------------
    def offline_train(self):
        start_t = time.time()
        max_episodes=3000
        time_limit=290
        episodes = 0
        initial_temp=1.0
        decay_rate=0.99
        min_temp=0.1
        # We'll do at most max_episodes or until train_time_limit
        while episodes < max_episodes:
            time_taken=time.time() - start_t
            if time_taken >= time_limit:
                # Reached the 300s limit
                break
            temprature=max(min_temp,initial_temp*(decay_rate**episodes))
            self.last_temprature=temprature

            # New episode: start from initial
            state = copy.deepcopy(self.initial_state)
            self.resets_used = 0
            done = False

            while not done and (time.time()-start_t)<time_limit:
                # time_taken=time.time()-start_t
                # if time_taken>=time_limit:
                #     break
                actions = self.env_legal_actions(state)
                if not actions:
                    break

                # epsilon-greedy
                if random.random() < self.epsilon:
                    action = random.choice(actions)
                else:
                    # if len(self.map[0])>4 or len(self.map)>4:
                    #     action=self.best_action_boltzmann(state,actions,temprature)
                    # else:
                    action = self.best_action(state, actions)
                next_state, reward, done = self.env_simulate_step(state, action)
                self.q_update(state, action, reward, next_state)

                state = next_state

            episodes += 1

    def act_boltzmann(self, state):
        """
        Called each turn. We pick the action with the highest Q(s,a).
        If multiple actions have the same best Q-value, pick the one
        with the largest number of "destroy" sub-actions.
        If no legal actions => wizard(s) all do 'wait'.
        """
        actions = self.env_legal_actions(state)
        if not actions:
            # fallback => wizards do 'wait'
            w_names = list(state["wizards"].keys())
            return tuple(("wait", w) for w in w_names)

        best_action = None
        best_q = float("-inf")
        best_destroy_count = -1  # track how many "destroy" sub-actions for tie-break

        for action in actions:
            # Extract features and create a stable key
            feats = self.extract_features(state, action)
            feats_key = tuple(sorted(feats.items()))

            # Lookup Q-value, defaulting to 0
            q_val = self.q_value(feats)

            # Count how many times this action includes a "destroy" sub-action
            destroy_count = 0
            if isinstance(action, (tuple, list)):
                for sub_a in action:
                    if isinstance(sub_a, (tuple, list)):
                        if sub_a[0] == "destroy":
                            destroy_count += 1

            # If we find a strictly better Q-value, update best
            if q_val > best_q:
                best_q = q_val
                best_action = action
                best_destroy_count = destroy_count

            # If Q-value ties the current best, prefer the action with more destroys
            elif q_val == best_q and destroy_count > best_destroy_count:
                best_action = action
                best_destroy_count = destroy_count

        return best_action

    def mcts_value_estimate(self, state, rollout_depth, simulation_time,action):
        """
        Run MCTS rollouts from the current state for a limited time,
        and return an estimate of the best (or average) cumulative reward.

        Args:
            state (dict): The current game state.
            rollout_depth (int): How many steps to simulate for each rollout.
            simulation_time (float): Total time (in seconds) to run simulations.

        Returns:
            best_reward (float): The maximum cumulative reward found over all rollouts.
            avg_reward (float): The average cumulative reward over all rollouts.
        """
        start_time = time.time()
        rewards = []

        # Run rollouts until we exceed the simulation time limit.
        while time.time() - start_time < simulation_time:
            total_reward = 0
            current_state = copy.deepcopy(state)
            done = False
            steps = 0
            while not done and steps < rollout_depth and time.time() - start_time < simulation_time:
                current_state, reward, done = self.env_simulate_step(current_state, action)
                actions = self.env_legal_actions(current_state)
                if not actions:
                    break
                # You can use a simple policy (e.g., random or your best action) for the rollout.
                action = random.choice(actions)
                total_reward += reward
                steps += 1
            rewards.append(total_reward)

        if rewards:
            best_reward = max(rewards)
            avg_reward = sum(rewards) / len(rewards)
        else:
            best_reward = 0
            avg_reward = 0

        return best_reward, avg_reward

    def act(self, state):
        """
        Called each turn. We pick the best Q(s,a) action greedily (0 epsilon).
        If no legal actions => wizard(s) all do 'wait'.
        """
        # if len(self.map[0])>5 or len(self.map)>5:
        #     return self.act_boltzmann(state)

        actions = self.env_legal_actions(state)
        if not actions:
            # fallback => wizards do 'wait'
            w_names = list(state["wizards"].keys())
            return tuple(("wait", w) for w in w_names)

        best = self.best_action(state, actions, epsilon=0.0)
        # best_reward, avg_reward = self.mcts_value_estimate(state, 5,3.5,best)
        #
        # # Define a threshold below which we decide it is not worth continuing.
        # reward_threshold = 0.1  # You can tune this value
        #
        # if best_reward < reward_threshold and avg_reward < reward_threshold:
        #     print("MCTS rollout suggests no positive reward; terminating.")
        #     return "terminate"

        return best
    #
    # def best_action_boltzmann(self, state, actions, temperature=1.0):
    #     """
    #     Return an action according to Boltzmann (Softmax) exploration.
    #     Uses Min-Max Scaling to avoid exponent overflow and underflow issues.
    #
    #     :param state: current state
    #     :param actions: list of possible actions
    #     :param temperature: softmax temperature; higher => more randomness
    #     :return: selected action
    #     """
    #     if not actions:
    #         return None  # No actions => no choice
    #
    #     # Compute Q-values
    #     q_values = [self.q_value(self.extract_features(state, a)) for a in actions]
    #
    #     # If all Q-values are the same, return a random action
    #     if len(set(q_values)) == 1:
    #         return random.choice(actions)
    #
    #     # Min-Max Scaling to [0, 1] range
    #     q_min, q_max = min(q_values), max(q_values)
    #
    #     if q_max == q_min:
    #         stable_qs = [0] * len(q_values)  # Avoid division by zero
    #     else:
    #         stable_qs = [(q - q_min) / (q_max - q_min) for q in q_values]  # Normalize
    #
    #     # Adjust scaling for temperature control
    #     stable_qs = [q / temperature for q in stable_qs]  # Prevent division by zero
    #
    #     # Compute exponentials safely
    #     exps = [math.exp(q) for q in stable_qs]
    #
    #     # Normalize to get probabilities
    #     sum_exps = sum(exps)
    #     if sum_exps == 0:
    #         return random.choice(actions)  # Fallback in case of all zero exponents
    #
    #     probs = [e / sum_exps for e in exps]
    #
    #     # Select an action based on probabilities
    #     chosen_action = random.choices(actions, weights=probs, k=1)[0]
    #     return chosen_action

    # --------------------------------------------------------------------------
    # Q-Learning Update
    # --------------------------------------------------------------------------
    def q_update(self, state, action, reward, next_state):
        feats = self.extract_features(state, action)
        old_q = self.q_value(feats)

        next_actions = self.env_legal_actions(next_state)
        if not next_actions:
            target = reward
        else:
            # best_next = max(self.q_value(self.extract_features(next_state, a2))
            #                 for a2 in next_actions)
            max_next_q=float('-inf')
            best_next=float('-inf')
            for a2 in next_actions:
                next_feat=self.extract_features(next_state, a2)
                best_next=max(self.q_value(next_feat),best_next)
                next_feat_key=tuple(sorted(next_feat.items()))
                q_val=self.Q_table.get(next_feat_key,0.0)
                max_next_q=max(max_next_q,q_val)

            target = reward + self.gamma * best_next



        td_error = target - old_q
        for f_name, f_val in feats.items():
            self.weights[f_name] += self.alpha * td_error * f_val

    # def q_value(self, features):
    #     val = 0.0
    #     for fn, fv in features.items():
    #         val += self.weights[fn] * fv
    #
    #     return val
    def q_value(self, features):
        # Compute the dot product: sum(weights_i * feature_i)
        dot = 0.0
        for fn, fv in features.items():
            dot += self.weights[fn] * fv

        # Compute the L2 norm of the feature vector: sqrt(sum(feature_i^2))

        norm = math.sqrt(sum(fv for fv in features.values()))#fv

        # Avoid division by zero
        if norm > 0:
            return dot / norm
        else:
            return dot

    def best_action(self, state, actions, epsilon=0.0):
        if random.random() < epsilon:
            return random.choice(actions)

        best_q = float("-inf")
        best_a = None
        for a in actions:
            feats = self.extract_features(state, a)
            q_val = self.q_value(feats)
            if q_val > best_q:
                best_q = q_val
                best_a = a
        return best_a

    # --------------------------------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------------------------------
    def extract_features(self, state, action):
        """
        Return dict of features:
          - coverage (possible horcrux-locs within 1 step of wizard)
          - collision_prob
          - if action= reset / terminate / destroy
          - # of horcrux left
          - BFS distance to nearest horcrux
          - BFS distance to nearest death eater
        """
        feats = defaultdict(float)

        # coverage
        coverage_val = self.estimate_coverage(state, action)
        feats["coverage"] = coverage_val

        # collision
        coll_prob = self.estimate_collision_prob(state, action)
        feats["collision_prob"] = coll_prob

        # handle single-string actions
        if action == "terminate":
            feats["action_terminate"] = 1.0
            if len(state["horcrux"])>0:
                feats["terminate_with_horcrux"] = 1.0
        elif action == "reset":
            feats["action_reset"] = 1.0
        else:
            # check if multi-subactions
            if isinstance(action,(tuple,list)):
                for sub_a in action:
                    # sub_a might be a string => skip
                    if isinstance(sub_a, (tuple, list)):
                        if sub_a[0] == "destroy":
                            feats["action_destroy"] += 1.0

        # horcrux_count
        feats["horcrux_count"] = float(len(state["horcrux"]))

        # BFS dist => horcrux
        feats["avg_hx_dist"] = self.estimate_avg_dist_to_horcrux(state, action)

        # BFS dist => death eater
        feats["avg_de_dist"] = self.estimate_avg_dist_to_deatheater(state, action)

        return feats

    def estimate_coverage(self, state, action):
        wlocs = self.apply_wizard_actions(state, action)
        possible_spots = set()

        for hx_name, hx_data in state["horcrux"].items():
            if "possible_locations" in hx_data:
                for loc in hx_data["possible_locations"]:
                    possible_spots.add(loc)

        coverage_ct = 0
        for spot in possible_spots:
            for loc in wlocs.values():
                if self.manhattan_dist(loc, spot)<=1:
                    coverage_ct += 1
                    break
        return float(coverage_ct)

    def estimate_collision_prob(self, state, action):
        wlocs = self.apply_wizard_actions(state, action)
        total_prob = 0.0
        for dename, ddata in state["death_eaters"].items():
            path = ddata["path"]
            idx = ddata["index"]
            # possible next indices
            opts = [idx]
            if idx+1<len(path):
                opts.append(idx+1)
            if idx-1>=0:
                opts.append(idx-1)

            p_each = 1.0/len(opts) if opts else 1.0
            for wiz_loc in wlocs.values():
                for choice in opts:
                    if path[choice] == wiz_loc:
                        total_prob += p_each

        if total_prob>1.0:
            total_prob=1.0
        return total_prob

    def estimate_avg_dist_to_horcrux(self, state, action):
        wlocs = self.apply_wizard_actions(state, action)
        hx_locs = []
        for hx_name,hx_data in state["horcrux"].items():
            hx_locs.append(hx_data["location"])
        if not hx_locs:
            return 0.0

        goals = set(hx_locs)
        sumd=0
        count_w=0
        for wloc in wlocs.values():
            d = self.bfs_distance_to_any(wloc, goals)
            if d is None:
                d=10
            sumd+=d
            count_w+=1

        if count_w==0:
            return 10
        return sumd/count_w

    def estimate_avg_dist_to_deatheater(self, state, action):
        wlocs = self.apply_wizard_actions(state, action)

        # gather each DE's current cell
        de_cells=[]
        for dename,ddata in state["death_eaters"].items():
            path=ddata["path"]
            idx=ddata["index"]
            if idx<len(path):
                de_cells.append(path[idx])

        if not de_cells:
            return 0.0

        goals=set(de_cells)
        sumd=0
        wcount=0
        for wloc in wlocs.values():
            d = self.bfs_distance_to_any(wloc, goals)
            if d is None:
                d=10
            sumd+=d
            wcount+=1
        if wcount==0:
            return 0
        return sumd/wcount

    def apply_wizard_actions(self, state, action):
        """
        Return wizard->location after applying sub-actions (move/destroy/wait)
        ignoring environment side-effects. We check types to avoid None error.
        """
        new_locs = {}
        for wn, wd in state["wizards"].items():
            new_locs[wn] = wd["location"]

        if not isinstance(action, (tuple,list)):
            # single string => "reset"/"terminate" => no wizard loc change
            a_list = [action]
        else:
            a_list = action

        for sub_a in a_list:
            if not isinstance(sub_a, (tuple, list)):
                # Possibly a string or None => treat as no wizard movement
                continue

            # now sub_a is presumably ("move", wizardName, (r,c)) or ("destroy", w, h)
            if sub_a[0]=='move':
                wname, dest = sub_a[1], sub_a[2]
                new_locs[wname] = dest
            # if 'destroy' or 'wait' => no location update

        return new_locs

    # --------------------------------------------------------------------------
    # BFS Helpers
    # --------------------------------------------------------------------------
    def manhattan_dist(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def bfs_distance_to_any(self, start, goals):
        visited=set([start])
        queue=deque([(start,0)])
        while queue:
            cell, dist=queue.popleft()
            if cell in goals:
                return dist
            for nbr, can_move in self.env_legal_moves(cell).items():
                if can_move and nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist+1))
        return None

    # --------------------------------------------------------------------------
    # Environment Simulation
    # --------------------------------------------------------------------------
    def env_simulate_step(self, state, action):
        """
        Single-step environment logic:
          - step cost => -0.01
          - if action=reset => revert if resets_used<max_resets
          - if action=terminate => big penalty if horcrux left, else +2
          - else sub-actions => move/destroy/wait
          - random DE movement => collision => -1
          - random horcrux teleport => prob
          - turns_to_go -=1 => done if 0
        """
        next_state = copy.deepcopy(state)
        rew = 0
        done = False

        # small step cost
        rew -= 0.01

        if action == "reset":
            if self.resets_used >= self.max_resets:
                # do nothing => treat as wait
                pass
            else:
                next_state = copy.deepcopy(self.initial_state)
                rew -= 2
                return (next_state, rew, done)

        elif action == "terminate":
            if len(next_state["horcrux"])>0:
                rew -= 10
            else:
                rew += 2
            done = True
            return (next_state, rew, done)

        else:
            # sub actions
            if not isinstance(action, (tuple,list)):
                a_list = [action]
            else:
                a_list = action

            for sub_a in a_list:
                if not isinstance(sub_a, (tuple, list)):
                    continue

                if sub_a[0]=="move":
                    wname, dest = sub_a[1], sub_a[2]
                    next_state["wizards"][wname]["location"] = dest
                elif sub_a[0]=="destroy":
                    wname, hx = sub_a[1], sub_a[2]
                    if hx in next_state["horcrux"]:
                        locW = next_state["wizards"][wname]["location"]
                        locH = next_state["horcrux"][hx]["location"]
                        if locW==locH:
                            rew += 2
                            # next_state["horcrux"].pop(hx, None)

        # random DE movement => collisions => -1
        for dename, ddata in next_state["death_eaters"].items():
            path = ddata["path"]
            idx = ddata["index"]
            opts = [idx]
            if idx+1<len(path):
                opts.append(idx+1)
            if idx-1>=0:
                opts.append(idx-1)
            choice = random.choice(opts)
            ddata["index"] = choice
            cell = path[choice]
            # collision => rew-1
            for wn, wdat in next_state["wizards"].items():
                if wdat["location"]==cell:
                    rew -=1

        # random horcrux teleports
        for hx_name, hx_data in list(next_state["horcrux"].items()):
            pchg = hx_data["prob_change_location"]
            poss = hx_data["possible_locations"]
            if random.random()<pchg:
                hx_data["location"] = random.choice(poss)

        # decrement turns
        next_state["turns_to_go"] = max(0, next_state["turns_to_go"]-1)
        if next_state["turns_to_go"]<=0:
            done=True

        return (next_state, rew, done)

    # --------------------------------------------------------------------------
    # Building Legal Actions
    # --------------------------------------------------------------------------
    def env_legal_actions(self, state):
        """
        Return a list of possible wizard-sub-action combos (cartesian product)
        plus 'reset'/'terminate'. Each wizard can do:
          - ('wait', wizardName)
          - ('destroy', wizardName, horcruxName) if on same cell
          - ('move', wizardName, newCell) if reachable
        Then we add single string 'reset', 'terminate'.
        """
        wizard_subacts = {}
        for wname, wd in state["wizards"].items():
            loc = wd["location"]
            subs = []
            # wait
            subs.append(("wait", wname))
            # destroy if on horcrux
            for hx_name,hx_data in state["horcrux"].items():
                if hx_data["location"]==loc:
                    subs.append(("destroy", wname, hx_name))
            # moves
            moves = self.env_legal_moves(loc)
            for cell, can_go in moves.items():
                if can_go:
                    subs.append(("move", wname, cell))

            wizard_subacts[wname]= subs

        combos = list(product(*(wizard_subacts[w] for w in wizard_subacts)))

        # Add 'reset','terminate'
        combos.append("reset")
        combos.append("terminate")
        return combos

    def env_legal_moves(self, pos):
        r,c=pos
        moves={
            (r-1,c):False,
            (r+1,c):False,
            (r,c-1):False,
            (r,c+1):False
        }
        for cc in moves:
            if cc in self.reachable:
                moves[cc]=True
        return moves

    def can_reach_any_horcrux_in_time(self, st):
        if len(st["horcrux"])==0:
            return False
        tleft= st["turns_to_go"]
        hx_locs=[]
        for hx_name,hx_data in st["horcrux"].items():
            hx_locs.append(hx_data["location"])
        goals=set(hx_locs)
        for wn,wdata in st["wizards"].items():
            d= self.bfs_distance_to_any(wdata["location"], goals)
            if d is not None and d<tleft:
                return True
        return False


class OptimalWizardAgent:
    def __init__(self, initial):
        """
        Initialize the WizardAgent with the provided initial game state.

        Args:
            initial (dict): The initial state of the game, including the map, wizards,
                            Death Eaters, Horcruxes, turns, and other parameters.
        """
        # Store input values
        self.optimal = initial["optimal"]  # Whether to act optimally
        self.map = initial["map"]  # The grid map
        self.wizards = initial["wizards"]  # Wizard positions as a dictionary
        self.horcruxes = initial["horcrux"]  # Horcrux locations and behavior
        self.death_eaters = initial["death_eaters"]  # Death Eater paths and indices
        self.turns_to_go = initial["turns_to_go"]  # Number of turns remaining
        self.score = 0  # Initialize score to 0

        # Parse the map into a dictionary for quick access
        self.grid = {}
        for i, row in enumerate(self.map):
            for j, cell in enumerate(row):
                self.grid[(i, j)] = cell

        # Compute passable tiles for quick lookup
        self.reachable = set((i, j) for (i, j), cell in self.grid.items() if cell == 'P')

        # Initialize Death Eater positions
        # self.death_eater_positions = {
        #     name: death_eater["path"][death_eater["index"]]
        #     for name, death_eater in self.death_eaters.items()
        # }
        self.horcrux_pos = {}
        self.turns=0

        for k,v in self.horcruxes.items():
            self.horcrux_pos[k] = v['possible_locations']


        # Store initial state for reset functionality
        self.initial_state = initial
        # self.initial_start = {
        #     initial['']
        # }
        self.pi={}
        self.value_function, self.police = self.value_iteration(initial,self.turns_to_go,0.9)
        # print(f'policy {self.police}')

    def act(self, state):
        """
        Decide the next action for the wizards based on the optimal policy.

        Args:
            state (dict): The current state of the game.

        Returns:
            tuple: The optimal action(s) for this turn.
        """
        # Hash the current state to retrieve from the policy
        state_hash = self.hash_state(state)
        turns_left = state["turns_to_go"]
        # Check if the state exists in the policy


        if state_hash in self.police:
            # if self.should_terminate(state,turns_left-1) is True:
            #      return 'terminate'

            optimal_action = self.police[state_hash][turns_left]

            self.turns+=1
            # self.turns+=1
            return optimal_action
        else:
            # If no policy is found, fallback to a default action
            actions = []
            for wizard_name in state["wizards"].keys():
                actions.append(("wait", wizard_name))
            return tuple(actions)

    def calc_reward(self, state, next_state, actions):
        """
        Calculate the reward for transitioning from `state` to `next_state` using `actions`.

        Args:
            state (dict): The current state.
            next_state (dict): The resulting state after applying actions.
            actions (list): The actions taken.

        Returns:
            int: The calculated reward.
        """
        rewards = 0

        # Check global actions
        for a in actions:
            if a == "reset":
                rewards -= 2  # Reset penalty
            elif a[0] == "destroy":
                rewards += 2  # Reward for destroying a Horcrux

        # Check penalties for Death Eater collisions in the resulting state
        for wizard_name, wizard_data in next_state["wizards"].items():
            wiz_location = wizard_data["location"]
            for death_eater in next_state["death_eaters"].values():
                index = death_eater["index"]
                if death_eater["path"][index] == wiz_location:
                    rewards -= 1  # Penalty for Death Eater collision

        return rewards

    def legal_moves_function(self,wizard_position):
        """
        Generate all legal moves for a wizard based on their current position.

        Args:
            wizard_position (tuple): The current position of the wizard as (x, y).
            reachable_tiles (set): A set of tiles that are passable ('P').

        Returns:
            dict: A dictionary mapping positions to a boolean indicating whether the move is legal.
        """
        x, y = wizard_position

        # Define potential moves (up, down, left, right)
        moves = {
            (x - 1, y): False,  # Up
            (x + 1, y): False,  # Down
            (x, y - 1): False,  # Left
            (x, y + 1): False  # Right
        }

        # Check if each move is within reachable tiles
        for pos in moves.keys():
            if self.map[pos[0]][pos[1]]=='P':
                moves[pos] = True
        return moves

    def death_eater_positions(self, state):
        """
        Compute all possible positions (indices) for each Death Eater based on their movement rules.

        Args:
            state (dict): Current game state containing Death Eater information.

        Returns:
            dict: A dictionary mapping each Death Eater's name to a list of possible indices.
        """
        positions = {}

        for death_eater_name, death_eater in state["death_eaters"].items():
            path = death_eater["path"]
            index = death_eater["index"]
            possible_indices = []

            # Add the current position (stay in place)
            possible_indices.append(index)

            # Add the forward position if it exists
            if index + 1 < len(path):
                possible_indices.append(index + 1)

            # Add the backward position if it exists
            if index - 1 >= 0:
                possible_indices.append(index - 1)

            # Map the Death Eater's name to their possible indices
            positions[death_eater_name] = possible_indices

        return positions
    def legal_actions(self, state):
        """
        Generate all legal actions for the current state.

        Args:
            state (dict): The current game state.

        Returns:
            list: A list of all possible action combinations for the state.
        """
        wizards_actions = {}

        # Generate actions for each wizard
        for wizard_name, wizard_data in state["wizards"].items():
            position = wizard_data["location"]
            wizards_actions[wizard_name] = []

            # Add 'wait' action
            wizards_actions[wizard_name].append(('wait', wizard_name))

            # Add 'reset' action
            # wizards_actions[wizard_name].append('reset')
            # wizards_actions[wizard_name].append('reset')

            # Add 'terminate' action (global, so it’s added once for consistency)
            # wizards_actions[wizard_name].append('terminate')

            # Add 'destroy' actions if the wizard is on a Horcrux
            for horcrux_name, horcrux_data in state["horcrux"].items():
                if horcrux_data["location"] == position:
                    wizards_actions[wizard_name].append(('destroy', wizard_name, horcrux_name))

            # Add 'move' actions for valid moves
            legal_moves = self.legal_moves(position)
            for move, is_valid in legal_moves.items():
                if is_valid:
                    wizards_actions[wizard_name].append(('move', wizard_name, move))

        # Combine all wizard actions into global actions using Cartesian product
        all_actions=list(product(*(wizards_actions[wizard] for wizard in wizards_actions)))
        # all_actions.append('reset')
        # all_actions.append('terminate')
        return all_actions

    def transition_function(self, state, actions):
        transitions = []
        _, current_state, current_reward = state
        transitions_state = []
        death_eaters_pos = self.death_eater_positions(current_state)
        for k,v in self.horcrux_pos.items():
            for horc_pos in v:
                for k1,v1 in death_eaters_pos.items():
                    for death_index in v1:
                        copy_state = copy.deepcopy(current_state)
                        # print(copy_state)
                        copy_state['horcrux'][k]['location'] = horc_pos
                        copy_state['death_eaters'][k1]['index'] = death_index
                        transitions_state.append(copy_state)

        for trans_state in transitions_state:
            all_rewards = 0
            for action in actions:
                next_state = copy.deepcopy(trans_state)
                reward = 0

                if action == "reset":
                    # turns = current_state["turns_to_go"]
                    next_state = copy.deepcopy(self.initial_state)
                    # next_state["turns_to_go"] = turns
                    reward -= 2

                    return [(action, next_state, reward)]
                elif action=='terminate':
                    continue
                elif action[0] == "move":
                    wizard_name, move_to = action[1], action[2]
                    next_state["wizards"][wizard_name]["location"] = move_to

                elif action[0] == "destroy":
                    wizard_name, horcrux_name = action[1], action[2]

                    if next_state["wizards"][wizard_name]["location"] == next_state["horcrux"][horcrux_name]["location"]:
                        reward += 2

                all_rewards+=reward
                # Decrease turns and add the resulting state
                # next_state["turns_to_go"] = current_state["turns_to_go"] - 1
                transitions.append((action, next_state, reward))#all_rewards instead of reward
        # print(transitions)

        return transitions

    def calculate_probabilities(self, state, next_state):
        prev_horc = []
        prev_death = []
        new_horc = []
        new_death = []
        for k, v in state["horcrux"].items():
            if len(v['possible_locations']) > 1:
                prev_horc.append((k, v['location'], v["prob_change_location"], v['possible_locations'], True))
            else:
                prev_horc.append((k, v['location'], v["prob_change_location"], v['possible_locations'], False))
        for k_d, v_d in state["death_eaters"].items():
            if v_d['index'] > 0 and v_d['index'] < len(v_d['path']) - 1:
                prev_death.append((k_d, v_d['index'], 1 / 3))
            else:
                prev_death.append((k_d, v_d['index'], 1 / 2))
        for k_h, v_h in next_state["horcrux"].items():
            if len(v_h['possible_locations']) > 1:
                prev_horc.append((k_h, v_h['location'], v_h["prob_change_location"], v_h['possible_locations'], True))
            else:
                prev_horc.append((k_h, v_h['location'], v_h["prob_change_location"], v_h['possible_locations'], False))

        p = 1
        for prev in prev_horc:
            if prev in new_horc:
                p = p * ((1 - prev[2]) + prev[2] * (1 / len(prev[3])))
            else:
                if prev[-1] == True:
                    p = p * prev[2] * (1 / (len(prev[3])))
        for death in prev_death:
            p = p * death[2]
        return p



    # def value_iteration(self, initial_state, horizon, discount=1.0):
    #     """
    #     Perform value iteration for the wizard game.
    #
    #     Args:
    #         initial_state (dict): The initial state of the game.
    #         horizon (int): The number of turns to consider.
    #         discount (float): The discount factor (0 <= discount <= 1).
    #
    #     Returns:
    #         tuple: (value_function, policy)
    #             - value_function: Dictionary mapping state hashes to their values.
    #             - policy: Dictionary mapping state hashes to optimal actions.
    #     """
    #     # Generate the state space
    #     states = self.get_states(initial_state)
    #     # print(f'space {states}')
    #     # print(len(states))
    #     # Initialize value function and policy
    #     value_function = {
    #         self.hash_state(state[1]): [0] * (horizon + 1)
    #         for state in states
    #     }
    #     policy = {
    #         self.hash_state(state[1]): [None] * horizon
    #         for state in states
    #     }
    #
    #     # Add terminal states to the value function
    #     for state in states:
    #         state_hash = self.hash_state(state[1])
    #         # if state[1]["turns_to_go"] == 0:
    #         #     value_function[state_hash][0] = 0  # Set terminal state value to 0
    #
    #     # print(value_function)
    #
    #     # Perform backward iteration
    #     for t in range(horizon-1, -1, -1):  # Start from the last turn and work backward
    #         for current_action, current_state, current_reward in states:
    #             state_hash = self.hash_state(current_state)
    #             max_value = float("-inf")
    #             best_action = None
    #
    #             # Get all legal actions for the current state
    #             actions = self.legal_actions(current_state)
    #
    #             # Evaluate each action
    #             for action in actions:
    #                 expected_value = 0
    #
    #                 # Get resulting states and probabilities
    #                 resulting_states = self.transition_function((current_action, current_state, current_reward), action)
    #                 for next_action, next_state, reward in resulting_states:
    #                     next_state_hash = self.hash_state(next_state)
    #
    #                     # if next_state["turns_to_go"] == 0:
    #                     #     continue  # Skip terminal states
    #                     prob = self.calculate_probabilities(current_state, next_state)
    #                     # expected_value += prob * value_function[next_state_hash][t + 1]
    #
    #                     # print(value_function["((('Harry Potter', (0, 2)),), (('Diary', (0, 0)), ('Nagini', (0, 2))), (('Snape', 1), ('random_de', 0)), 0)"])
    #                     expected_value += prob * (reward + value_function[next_state_hash][t + 1])
    #
    #                 # Update maximum value and best action
    #                 expected_value=current_reward+expected_value
    #                 if expected_value > max_value:
    #                     max_value = expected_value
    #                     best_action = action
    #
    #             # Store the best value and action for the current state and time
    #             value_function[state_hash][t] = max_value
    #             # print(f'state hash {state_hash}')
    #             policy[state_hash][t] = best_action
    #
    #     return value_function, policy
    def tie_break_action(self,best_actions,best_val,actions_vals):
        max_destroys=float('-inf')
        max_moves=float('-inf')
        best_actions_destroys=None
        best_actions_moves=None
        for i,actions in enumerate(best_actions):
            if actions_vals[i]==best_val:
                count_destroys=0
                count_moves=0
                for action in actions:
                    if action[0]=='destroy':
                        count_destroys+=1
                    if action[0]=='move':
                        count_moves+=1
                if count_destroys>max_destroys:
                    max_destroys=count_destroys
                    best_actions_destroys=actions
                if count_moves>max_moves:
                    max_moves=count_moves
                    best_actions_moves=actions
        if max_destroys>0:
            return best_actions_destroys
        return None





    def value_iteration(self, initial_state, horizon, discount=1.0):
        """
        Perform value iteration for the wizard game.

        Args:
            initial_state (dict): The initial state of the game.
            horizon (int): The number of turns to consider.
            discount (float): The discount factor (0 <= discount <= 1).

        Returns:
            tuple: (value_function, policy)
                - value_function: Dictionary mapping state hashes to their values.
                - policy: Dictionary mapping state hashes to optimal actions.
        """
        # Generate the state space
        states = self.get_states(initial_state)
        # print(f'space {states}')
        # print(len(states))
        # Initialize value function and policy
        value_function = {
            self.hash_state(state[1]): [0] * (horizon + 1)
            for state in states
        }
        policy = {
            self.hash_state(state[1]): [None] * (horizon+1)
            for state in states
        }

        # Perform backward iteration
        for t in range(horizon+1):  # Start from the last turn and work backward
            for current_action, current_state, current_reward in states:
                state_hash = self.hash_state(current_state)
                max_value = float("-inf")
                best_action = None
                if t==0:
                    value_function[state_hash][t] = current_reward
                    continue

                # Get all legal actions for the current state
                actions = self.legal_actions(current_state)
                actions_values=[0]*len(actions)
                # Evaluate each action
                for i,action in enumerate(actions):
                    expected_value = 0

                    # Get resulting states and probabilities
                    resulting_states = self.transition_function((current_action, current_state, current_reward), action)
                    for next_action, next_state, reward in resulting_states:
                        next_state_hash = self.hash_state(next_state)

                        # if next_state["turns_to_go"] == 0:
                        #     continue  # Skip terminal states
                        prob = self.calculate_probabilities(current_state, next_state)
                        # expected_value += prob * value_function[next_state_hash][t + 1]

                        # print(value_function["((('Harry Potter', (0, 2)),), (('Diary', (0, 0)), ('Nagini', (0, 2))), (('Snape', 1), ('random_de', 0)), 0)"])
                        expected_value += prob * (reward+value_function[next_state_hash][t-1])

                    # Update maximum value and best action
                    expected_value=current_reward+expected_value
                    actions_values[i]=expected_value
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = action

                # Store the best value and action for the current state and time

                value_function[state_hash][t] = max_value
                # print(f'state hash {state_hash}')
                best_action_=self.tie_break_action(actions,max_value,actions_values)
                if best_action_ is None:
                    best_action_=best_action
                policy[state_hash][t] = best_action_

        return value_function, policy

    def get_states(self, initial):
        queue = [(None, initial, 0)]
        colored_states = set()
        state_space = []

        while queue:
            # print(queue[0])
            current_action, current_state, current_reward = queue.pop(0)
            state_hash = self.hash_state(current_state)

            # if current_state["turns_to_go"] <= 0:
            #     continue  # Skip states with no turns remaining

            if state_hash not in colored_states:
                colored_states.add(state_hash)
                state_space.append((current_action, current_state, current_reward))

                # Generate all legal actions for the current state
                all_actions = self.legal_actions(current_state)

                for actions in all_actions:
                    resulting_states = self.transition_function(
                        (current_action, current_state, current_reward), actions
                    )
                    for result in resulting_states:
                        queue.append(result)

            # print(f"Total States Generated: {len(state_space)}")
        return state_space



    def hash_state(self, state):
        """
        Create a unique and hashable representation of the state.

        Args:
            state (dict): The state dictionary.

        Returns:
            tuple: A hashable representation of the state.
        """
        return (
            tuple(sorted((wizard, tuple(data["location"])) for wizard, data in state["wizards"].items())),
            tuple(sorted((horcrux, tuple(data["location"])) for horcrux, data in state["horcrux"].items())),
            tuple(sorted((death_eater, data["index"]) for death_eater, data in state["death_eaters"].items())),
            # state["turns_to_go"],
        )


    def compute_reachable(self):
        reachable = set()
        for i,j in self.grid:
            if self.grid[(i,j)] == 'P':
                reachable.add((i,j))
        return reachable

    def legal_moves(self,wiz_loc):
        moves = {(wiz_loc[0]-1,wiz_loc[1]): False,(wiz_loc[0]+1,wiz_loc[1]): False,(wiz_loc[0],wiz_loc[1]-1): False,(wiz_loc[0],wiz_loc[1]+1): False}
        for k,v in moves.items():
            if k in self.reachable:
                moves[k] = True
        return moves








