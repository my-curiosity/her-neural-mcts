import numpy as np

from src.game.game_history import GameHistory
from src.syntax_tree.syntax_tree import SyntaxTree
import copy
from src.residual.get_residual_of_equation import create_new_data_frame
import pandas as pd
import tensorflow as tf
from src.utils.logging import get_log_obj
from src.utils.error import NonFiniteError
from src.generate_datasets.dataset_generator import constant_dict_to_string


class Hindsight:
    """
    Starts with a state corresponding to a finished formula.
    Calls on the forward pass the predecessors and adds the actions that created the final state.
    When initial node is reached the corresponding syntax tree is constructed.
    With the new tree the corresponding y-values are calculated.
    In the backward pass the new dataset is added to the history object.
    """

    def __init__(self, action_size, dataset_columns, grammar, args, mcts=None):
        self.action_size = action_size
        self.dataset_columns = dataset_columns
        self.grammar = grammar
        self.args = args
        self.logger = get_log_obj(args=args, name="Hindsight")
        self.created_samples = 0
        self.complete_syntax_tree = None
        self.hindsight_data_without_residual = (None,)
        self.mcts = mcts

    def create_hindsight_history(self, final_state):
        """
        starting method to get the hindsight history
        In the forward pass we generate the instances, where syntax tree and
        data does not fit.
        On the backward pass we create the dataset which fits with the syntax tree.
        :param final_state:
        :return:
        """
        try:
            self.complete_syntax_tree = None
            self.hindsight_data_without_residual = None
            history_forward = GameHistory()
            history_backward = GameHistory()
            history_forward.observed_returns = []
            history_backward.observed_returns = []
            self.constant_dict = final_state.syntax_tree.constants_in_tree
            self.visit_previous_state(
                state=final_state,
                history_forward=history_forward,
                history_backward=history_backward,
                actions=[],
            )
            history_forward.found_equation = f"{self.hindsight_true_equation}_f"
            history_backward.found_equation = f"{self.hindsight_true_equation}_b"
            self.created_samples += 1
            return history_forward, history_backward
        except AssertionError:
            self.logger.debug(
                f"Equation can not be evaluated"
                f"the equation is: {final_state.syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n"
            )
            return [], []
        except FloatingPointError:
            self.logger.debug(
                f"In the calculation of the hindsight a FloatingPointError occur"
                f"the equation is: {final_state.syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n"
                f"the dataset is: {final_state.observation['data_frame']} "
            )
            return [], []
        except RuntimeError as e:
            self.logger.debug(
                f"In the calculation of the hindsight a RuntimeError occur."
                f"The error is {e} "
                f"No history is added"
            )
            return [], []
        except NonFiniteError:
            self.logger.debug(
                f"In the calculation of the hindsight a NonFinite Element occur."
                f"No history added"
            )
            return [], []

    def visit_previous_state(self, state, history_forward, history_backward, actions):
        """
        adds the action which created this state to the action list.
        It will first create the forward instance and call its previous state.
        On the way back the backward instance is created.
        :param state:
        :param history_forward:
        :param history_backward:
        :param actions:
        :return:
        """
        actions.insert(0, state.production_action)
        previous_state = state.previous_state
        self.add_forward_history(history_forward, previous_state, state)
        history_forward, history_backward, hindsight_data = self.forward_pass(
            actions, history_forward, history_backward, previous_state
        )

        history_forward, history_backward, hindsight_data = self.backward_pass(
            hindsight_data, history_forward, history_backward, previous_state, state
        )

        return history_forward, history_backward, hindsight_data

    def add_forward_history(self, history_forward, previous_state, state):
        """
        adds the forward history. When the reward of the final state is
        not high enough a uniform distribution is added to the buffer
        :param history_forward:
        :param previous_state:
        :param state:
        :return:
        """
        if self.mcts:
            possible_actions = np.nonzero(
                self.mcts.valid_moves_for_s[previous_state.hash]
            )[0]
            if state.reward < 0.5:
                mcts_distribution = np.zeros(shape=self.action_size)
                probability_one_action = 1 / len(possible_actions)
                for action in possible_actions:
                    mcts_distribution[action] = probability_one_action
            else:
                counts = np.zeros(shape=self.action_size)
                for action in possible_actions:
                    key = (previous_state.hash, action)
                    if key in self.mcts.times_edge_s_a_was_visited:
                        counts[action] = self.mcts.times_edge_s_a_was_visited[key]
                mcts_distribution = counts / np.sum(counts)
            previous_state.action = state.production_action
            previous_state.reward = state.reward * self.args.gamma
            history_forward.observed_returns.append(state.reward)
            history_forward.capture(
                state=previous_state, pi=mcts_distribution, r=state.reward, v=None
            )

    def backward_pass(
        self, hindsight_data, history_forward, history_backward, previous_state, state
    ):
        """
        Adds a state with the new generated data set to the buffer
        :param hindsight_data:
        :param history_forward:
        :param history_backward:
        :param previous_state:
        :param state:
        :return:
        """
        state_hindsight = copy.deepcopy(previous_state)
        state_hindsight.observation["data_frame"] = hindsight_data
        state_hindsight.observation["true_equation"] = self.hindsight_true_equation
        state_hindsight.observation["true_equation_hash"] = (
            self.hindsight_true_equation_hash
        )
        state_hindsight.action = state.production_action
        hindsight_distribution = np.zeros(self.action_size)
        hindsight_distribution[state.production_action] = 1
        if state_hindsight.residual_calculated:
            df = copy.deepcopy(self.hindsight_data_without_residual)
            y_calc = self.complete_syntax_tree.evaluate_subtree(
                node_id=state_hindsight.syntax_tree.start_node.node_id,
                dataset=self.hindsight_data_without_residual,
            )
            df["y"] = y_calc
            hindsight_data = df
            state_hindsight.observation["data_frame"] = df
        history_backward.observed_returns.append(1)
        history_backward.capture(
            state=state_hindsight, pi=hindsight_distribution, r=1, v=None
        )
        return history_forward, history_backward, hindsight_data

    def forward_pass(self, actions, history_forward, history_backward, previous_state):
        if check_for_first_state(previous_state):
            hindsight_data = self.create_hindsight_data(
                actions=actions, original_data=previous_state.observation["data_frame"]
            )
        else:
            # call previous state
            history_forward, history_backward, hindsight_data = (
                self.visit_previous_state(
                    state=previous_state,
                    history_forward=history_forward,
                    history_backward=history_backward,
                    actions=actions,
                )
            )
        return history_forward, history_backward, hindsight_data

    def create_hindsight_data(self, actions, original_data):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        for action in actions:
            syntax_tree.expand_node_with_action(
                node_id=syntax_tree.nodes_to_expand[0], action=action
            )
        syntax_tree.constants_in_tree = self.constant_dict
        hind_sight_df = copy.deepcopy(original_data)
        y_calc = syntax_tree.evaluate_subtree(
            node_id=syntax_tree.start_node.node_id,
            dataset=hind_sight_df,
        )
        if np.all(np.isfinite(y_calc)):
            c_string_backward = constant_dict_to_string(syntax_tree)
            equation_string = f"{syntax_tree.rearrange_equation_infix_notation(-1)[1]}"
            self.hindsight_true_equation = f"{equation_string}_{c_string_backward}"
            self.hindsight_true_equation_hash = equation_string.strip()
            hind_sight_df["y"] = y_calc
            self.complete_syntax_tree = syntax_tree
            self.hindsight_data_without_residual = hind_sight_df
            return hind_sight_df
        else:
            self.logger.info(
                f"In calculation of hindsight  y_calc a non finite element happen. "
            )
            raise RuntimeError


def check_for_first_state(state):
    if state.previous_state:
        return False
    else:
        return True
