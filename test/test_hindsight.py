import unittest

import pandas as pd
from src.game.find_equation_game import FindEquationGame
from pcfg import PCFG
import tensorflow as tf
from src.utils.logging import get_log_obj
import random
from src.residual.get_residual_of_equation import get_residual_of_equation
from src.hindsight.hindsight import Hindsight
import numpy as np
from src.mcts.classic_mcts import ClassicMCTS


def get_empty_list(arg):
    return []


class TestHindsightExperienceReplay(unittest.TestCase):
    def setUp(self) -> None:
        grammar_string = """
                    S -> Constant [0.2]
                    S -> Variable [0.1]
                    S -> '+' S S [0.2]
                    S -> '-' S S [0.1]
                    S -> '*' S S [0.1]
                    S -> '/' S S [0.1]
                    S -> 'sin' S_ [0.1]
                    S -> 'cos' S_ [0.1]
                    Constant -> 'c' [0.15]
                    Constant -> '2' [0.2]
                    Constant -> '3' [0.2]
                    Constant -> '4' [0.15]
                    Constant -> '5' [0.15]
                    Constant -> '6' [0.15]
                    Variable -> 'x_0' [1.0]
                    S_ -> Constant [0.2]
                    S_ -> '+' S_ S_ [0.4]
                    S_ -> '-' S_ S_ [0.2]
                    S_ -> '*' S_ S_ [0.1]
                    S_ -> '/' S_ S_ [0.1]
                    """

        self.grammar = PCFG.fromstring(grammar_string)

        class Namespace:
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 10
        self.args.max_num_nodes_in_syntax_tree = 10
        self.args.max_constants_in_tree = 5
        self.args.max_constants_in_tree = 5
        self.args.number_equations = 10
        self.args.num_calls_sampling = 10
        self.args.sample_with_noise = False
        self.args.how_to_select_node_to_delete = "random"

        self.args.precision = "float32"
        self.logger = get_log_obj(args=self.args, name="test_logger")
        self.args.logging_level = 40
        self.args.equation_preprocess_class = "PandasPreprocess"
        self.args.max_len_datasets = 10

        self.args.data_path = "test/saved_object"
        self.args.tree_representation = "tree_structure"
        self.args.max_tokens_equation = 64
        self.args.batch_size_loading = 1
        self.args.minimum_reward = -1
        self.args.gamma = 1
        self.args.max_elements_in_best_list = 10
        self.args.build_syntax_tree_token_based = False

    def test_hindsight(self):
        random.seed(42)
        np.random.seed(0)
        dataset_true = np.array(
            [
                [0.3, 6.0],
                [1.3777778, 6.0],
                [2.4555554, 6.0],
                [3.5333333, 6.0],
                [4.611111, 6.0],
                [5.688889, 6.0],
                [6.766667, 6.0],
                [7.8444443, 6.0],
                [8.922222, 6.0],
                [10.0, 6.0],
            ]
        )
        tf.random.set_seed(seed=42)
        self.find_equation_game = FindEquationGame(
            grammar=self.grammar,
            args=self.args,
        )
        self.mcts = ClassicMCTS(
            game=self.find_equation_game, rule_predictor=None, args=self.args
        )
        self.game = FindEquationGame(self.grammar, self.args, train_test_or_val="train")

        state_0 = self.find_equation_game.getInitialState()
        state_0.observation["data_frame"] = pd.DataFrame(
            dataset_true, columns=["x_0", "y"]
        )
        hash_0 = self.find_equation_game.getHash(state=state_0)
        self.mcts.valid_moves_for_s[hash_0] = self.game.getLegalMoves(state_0).astype(
            bool
        )
        state_0.residual_calculated = False

        state_1, reward = self.find_equation_game.getNextState(
            state=state_0, action=4
        )  # '*' S S
        hash_1 = self.find_equation_game.getHash(state=state_1)
        self.mcts.valid_moves_for_s[hash_1] = self.game.getLegalMoves(state_1).astype(
            bool
        )
        state_1.residual_calculated = False

        state_2, reward = self.find_equation_game.getNextState(
            state=state_1, action=0
        )  # Constant
        hash_2 = self.find_equation_game.getHash(state=state_2)
        self.mcts.valid_moves_for_s[hash_2] = self.game.getLegalMoves(state_2).astype(
            bool
        )
        state_2.residual_calculated = False

        state_3, reward = self.find_equation_game.getNextState(
            state=state_2, action=8
        )  # c
        hash_3 = self.find_equation_game.getHash(state=state_3)
        self.mcts.valid_moves_for_s[hash_3] = self.game.getLegalMoves(state_3).astype(
            bool
        )
        state_3.residual_calculated = False

        state_4, reward = self.find_equation_game.getNextState(
            state=state_3, action=1
        )  # Variable
        hash_4 = self.find_equation_game.getHash(state=state_4)
        self.mcts.valid_moves_for_s[hash_4] = self.game.getLegalMoves(state_4).astype(
            bool
        )
        state_4.residual_calculated = False

        state_5, reward = self.find_equation_game.getNextState(
            state=state_4, action=14
        )  # x_0
        hash_5 = self.find_equation_game.getHash(state=state_5)
        state_5.residual_calculated = False

        hindsight = Hindsight(
            action_size=self.find_equation_game.action_size,
            dataset_columns=self.find_equation_game.dataset_columns,
            grammar=self.grammar,
            args=self.args,
            mcts=self.mcts,
        )
        y_soll = (
            dataset_true[:, 0] * state_5.syntax_tree.constants_in_tree["c_0"]["value"]
        )
        history_forward, history_backward = hindsight.create_hindsight_history(
            final_state=state_5
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[0]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[1]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[2]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[3]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[4]["data_frame"]["y"], decimal=2
        )

    def test_hindsight_residual(self):
        random.seed(42)
        dataset_true = np.array(
            [
                [0.3, 6.0],
                [1.3777778, 6.0],
                [2.4555554, 6.0],
                [3.5333333, 6.0],
                [4.611111, 6.0],
                [5.688889, 6.0],
                [6.766667, 6.0],
                [7.8444443, 6.0],
                [8.922222, 6.0],
                [10.0, 6.0],
            ]
        )
        tf.random.set_seed(seed=42)
        self.find_equation_game = FindEquationGame(
            grammar=self.grammar,
            args=self.args,
        )
        self.mcts = ClassicMCTS(
            game=self.find_equation_game, rule_predictor=None, args=self.args
        )
        self.game = FindEquationGame(self.grammar, self.args, train_test_or_val="train")

        state_0 = self.find_equation_game.getInitialState()
        state_0.observation["data_frame"] = pd.DataFrame(
            dataset_true, columns=["x_0", "y"]
        )
        hash_0 = self.find_equation_game.getHash(state=state_0)
        self.mcts.valid_moves_for_s[hash_0] = self.game.getLegalMoves(state_0).astype(
            bool
        )
        state_0.residual_calculated = False

        state_1, reward = self.find_equation_game.getNextState(
            state=state_0, action=4
        )  # '*' S S
        hash_1 = self.find_equation_game.getHash(state=state_1)
        self.mcts.valid_moves_for_s[hash_1] = self.game.getLegalMoves(state_1).astype(
            bool
        )
        state_1.residual_calculated = False

        state_2, reward = self.find_equation_game.getNextState(
            state=state_1, action=0
        )  # Constant
        state_2 = get_residual_of_equation(
            state=state_2,
            function_to_get_current_tree_representation_int=self.find_equation_game.reader.map_tree_representation_to_int,
            logger=self.logger,
        )
        hash_2 = self.find_equation_game.getHash(state=state_2)
        self.mcts.valid_moves_for_s[hash_2] = self.game.getLegalMoves(state_2).astype(
            bool
        )
        state_2.residual_calculated = True

        state_3, reward = self.find_equation_game.getNextState(
            state=state_2, action=8
        )  # c
        state_3 = get_residual_of_equation(
            state=state_3,
            function_to_get_current_tree_representation_int=self.find_equation_game.reader.map_tree_representation_to_int,
            logger=self.logger,
        )
        hash_3 = self.find_equation_game.getHash(state=state_3)
        self.mcts.valid_moves_for_s[hash_3] = self.game.getLegalMoves(state_3).astype(
            bool
        )
        state_3.residual_calculated = True

        state_4, reward = self.find_equation_game.getNextState(
            state=state_3, action=1
        )  # Variable
        hash_4 = self.find_equation_game.getHash(state=state_4)
        self.mcts.valid_moves_for_s[hash_4] = self.game.getLegalMoves(state_4).astype(
            bool
        )
        state_4.residual_calculated = False

        state_5, reward = self.find_equation_game.getNextState(
            state=state_4, action=14
        )  # x_1     y = c * x_1
        self.find_equation_game.getHash(state=state_5)
        state_5.residual_calculated = False

        hindsight = Hindsight(
            action_size=self.find_equation_game.action_size,
            dataset_columns=self.find_equation_game.dataset_columns,
            grammar=self.grammar,
            args=self.args,
            mcts=self.mcts,
        )
        y_soll = (
            dataset_true[:, 0] * state_5.syntax_tree.constants_in_tree["c_0"]["value"]
        )
        y_soll_residual = dataset_true[:, 0]
        history_forward, history_backward = hindsight.create_hindsight_history(
            final_state=state_5
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[0]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[1]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll, history_backward.observations[2]["data_frame"]["y"], decimal=2
        )
        np.testing.assert_almost_equal(
            y_soll_residual,
            history_backward.observations[3]["data_frame"]["y"],
            decimal=2,
        )
        np.testing.assert_almost_equal(
            y_soll_residual,
            history_backward.observations[4]["data_frame"]["y"],
            decimal=2,
        )
