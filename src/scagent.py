import random
import math
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from QLearningTable import QLearningTable
from ScLogger import ScLogger

import NEUTRAL
import TERRAN
import REWARD

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_SCREEN = [0]
MAP_MAXSIZE = 83

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]


class SmartAgent(base_agent.BaseAgent):
    # depot_x = 5

    def __init__(self):
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_score = 0
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_action = None
        self.previous_state = None

        self.depot_x = 5
        self.depot_y = 5

    # def transformLocation(self, x, x_distance, y, y_distance):
    #     if not self.base_top_left:
    #         return [max(x - x_distance, 0), max(y - y_distance, 0)]
    #
    #     return [min(x + x_distance, MAP_MAXSIZE), min(y + y_distance, MAP_MAXSIZE)]

    def step(self, obs):

        super(SmartAgent, self).step(obs)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        unit_type = obs.observation['screen'][_UNIT_TYPE]
        depot_y, depot_x = (unit_type == TERRAN.SUPPLY_DEPOT).nonzero()

        supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == TERRAN.BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        killed_unit_score = obs.observation['score_cumulative'][5]
        score = obs.observation['score_cumulative'][0]
        killed_building_score = obs.observation['score_cumulative'][6]

        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply,
        ]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += REWARD.KILL_UNIT

            if killed_unit_score < self.previous_killed_unit_score:
                reward += REWARD.LOST_UNIT

            if killed_building_score > self.previous_killed_building_score:
                reward += REWARD.KILL_BUILDING

            # if 0 != reward:

            tt = math.exp(-np.logaddexp(0, -(score - self.previous_score)))

            self.qlearn.learn(str(self.previous_state), self.previous_action, tt, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_score = score
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        if smart_action == ACTION_DO_NOTHING:
            # ScLogger.logbo(smart_action)
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == TERRAN.SCV).nonzero()

            if unit_y.any() and len(unit_y) > 1:
                i = random.randrange(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]

                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT and supply_depot_count == 0:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                ScLogger.logbo(smart_action)
                target = self.findLocationForBuilding(obs,TERRAN.SUPPLY_DEPOT_SIZE)
                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                ScLogger.logbo(smart_action)
                target = self.findLocationForBuilding(obs,TERRAN.BARRACKS_SIZE)
                return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])

        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == TERRAN.BARRACKS).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_MARINE:
            ScLogger.logbo(smart_action)
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [[1]])

        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [[0]])

        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [[1], [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [[1], [21, 24]])

        return actions.FunctionCall(_NO_OP, [])

    def findLocationForBuilding(self, obs, size):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        minefield_y, minefield_x = (unit_type == NEUTRAL.MINERALFIELD).nonzero()
        max_y, max_x = unit_type.shape
        distance = 6
        chance = 10
        while True:
            s_target_x = int(self.depot_x + np.random.choice([-1, 0, 1], 1) * distance)
            s_target_y = int(self.depot_y + np.random.choice([-1, 0, 1], 1) * distance)
            within_map = (0 < s_target_x < max_x - size) and (0 < s_target_y < max_y - size)
            buildings = [NEUTRAL.MINERALFIELD] + TERRAN.BUILDINGS
            ScLogger.log(buildings)
            area = unit_type[s_target_y:s_target_y + 6][s_target_x:s_target_x + 6]
            space_available = not any(x in area for x in buildings)
            within_mineral_field = (min(minefield_y) < s_target_y < max(minefield_y)) and (
                min(minefield_x) < s_target_x < 11 + max(minefield_x))
            chance += -1
            if within_map and space_available and not within_mineral_field:
                self.depot_x = s_target_x
                self.depot_y = s_target_y
                s_target = np.array([s_target_x, s_target_y])
                ScLogger.log(s_target)
                break
            if chance == 0:
                distance += 1
                chance = 10
        ScLogger.log("Go go building")
        return s_target
