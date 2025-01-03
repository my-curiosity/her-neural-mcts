"""
Main function of the codebase. This file is intended to call different parts of our pipeline based on console arguments.

To add new games to the pipeline, add your string_query-class constructor to the 'game_from_name' function.
https://github.com/kaesve/muzero
"""

import warnings
import random
from datetime import datetime
from src.config import Config
from src.coach import Coach
from src.neural_nets.equation.equation_rule_predictor_skeleton import (
    EquationRulePredictorSkeleton,
)
from src.neural_nets.bit_flip.bit_flip_predictor import BitFlipPredictor
from src.neural_nets.point_maze.point_maze_predictor import PointMazePredictor
from src.game.find_equation_game import FindEquationGame
from src.game.gym_game import GymGame, make_env
from src.mcts.classic_mcts import ClassicMCTS
from src.mcts.amex_mcts import AmEx_MCTS
import tensorflow as tf
import numpy as np
import wandb
from definitions import ROOT_DIR

from src.utils.copy_weights import copy_dataset_encoder_weights_from_pretrained_agent
from src.utils.get_grammar import get_grammar_from_string
from src.equation_modules.generate_datasets.grammars import get_grammars

warnings.filterwarnings("ignore")


def run():
    args = Config.arguments_parser()
    args.ROOT_DIR = ROOT_DIR
    time_string = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    unique_dir = f"{time_string}_{args.seed}"
    wandb_path = ROOT_DIR / ".wandb" / args.experiment_name / f"{unique_dir}"
    wandb_path.mkdir(parents=True, exist_ok=True)
    wandb.init(
        config=args.__dict__,
        project=args.project_name,
        sync_tensorboard=True,
        tensorboard=True,
        dir=wandb_path,
        mode=args.wandb,
        name=args.experiment_name,
    )
    wandb.log({"Job_ID": args.job_id})

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    grammar = get_grammar_from_string(
        string=get_grammars(args.grammar_search), args=args
    )

    if args.game == "gym":
        game = GymGame(
            args,
            make_env(
                env_str=args.gym_env_str,
                max_episode_steps=args.gym_max_episode_steps,
                minimum_reward=args.minimum_reward,
                maximum_reward=args.maximum_reward,
            ),
        )
    else:  # equation
        game = FindEquationGame(grammar, args, train_test_or_val="train")

    learn_a0(game=game, args=args, run_name=args.experiment_name)
    wandb.log({f"successful": True})


def learn_a0(game, args, run_name):
    """
    Train an AlphaZero agent on the given environment with the specified configuration. If specified within the
    configuration file, the function will load in a previous model along with previously generated data.
    :param args:
    :param game: Game Instance of a Game class that implements environment logic. Train agent on this environment.
    :param run_name: str Run name to store data by and annotate results.
    """
    print("Testing:", ", ".join(run_name.split("_")))

    # Extract neural network and algorithm arguments separately
    if args.game == "gym":
        if args.gym_env_str.startswith("BitFlip"):
            rule_predictor_train = BitFlipPredictor(game=game, args=args)
        elif args.gym_env_str.startswith("PointMaze"):
            rule_predictor_train = PointMazePredictor(game=game, args=args)
        else:
            raise NotImplementedError
    else:  # equation
        rule_predictor_train = EquationRulePredictorSkeleton(
            args=args, reader_train=game.reader
        )

    checkpoint_train, manager_train = load_pretrained_net(
        args=args, rule_predictor=rule_predictor_train, game=game
    )

    if args.mcts_engine == "Endgame":
        search_engine = AmEx_MCTS
    elif args.mcts_engine == "Normal":
        search_engine = ClassicMCTS
    else:
        raise AssertionError(f"Engine: {args.mcts_engine} not defined!")

    c = Coach(
        game=game,
        rule_predictor=rule_predictor_train,
        args=args,
        search_engine=search_engine,
        run_name=run_name,
        checkpoint_train=checkpoint_train,
        checkpoint_manager=manager_train,
    )

    c.learn()


def load_pretrained_net(args, rule_predictor, game):
    experiment_name = f"{args.experiment_name}/{args.seed}"
    net = rule_predictor.net if args.game == "equation" else rule_predictor.net.model
    checkpoint_path_current_model = (
        ROOT_DIR / "saved_models" / args.data_path / experiment_name
    )
    print(f"Model will be saved at {checkpoint_path_current_model}")

    checkpoint_current_model = tf.train.Checkpoint(step=tf.Variable(1), net=net)
    manager_train = tf.train.CheckpointManager(
        max_to_keep=30,
        step_counter=checkpoint_current_model.step,
        checkpoint=checkpoint_current_model,
        directory=str(checkpoint_path_current_model / "tf_ckpts"),
        checkpoint_interval=10,
    )
    initialize_net(args, checkpoint_current_model, game)

    if args.load_pretrained and len(args.path_to_complete_model) > 0:
        restore_path = ROOT_DIR / args.path_to_complete_model
        if restore_path.suffix != "":
            raise RuntimeError(
                f"Your path to the complete model has an suffix: {restore_path.suffix} \n "
                f"the restore operation wants to have the path in the form *path_to_checkpoint/tf_chpts/ckpt-x* \n"
                f" Most likely you add the path to the index file \n"
                f"Your path is: {restore_path}"
            )

        checkpoint_current_model.restore(f"{restore_path}")
        print("Restored from {}".format(f"{ROOT_DIR / args.path_to_complete_model }"))

    elif args.load_pretrained and manager_train.latest_checkpoint:
        checkpoint_current_model.restore(
            manager_train.latest_checkpoint
        ).assert_consumed()

        print("Restored from {}".format(manager_train.latest_checkpoint))
    else:
        # checkpoint_current_model.restore(manager_train.latest_checkpoint)
        print("Initializing from scratch.")

    copy_dataset_encoder_weights_from_pretrained_agent(
        args=args, checkpoint_current_model=checkpoint_current_model, game=game
    )

    return checkpoint_current_model, manager_train


def initialize_net(args, checkpoint_current_model, game):
    net = checkpoint_current_model.net

    if isinstance(game, FindEquationGame):
        iterator = game.reader.get_datasets()
        data_dict = next(iterator)
        prepared_syntax_tree = [
            np.zeros(shape=args.max_tokens_equation, dtype=np.float32)
        ]
        net(
            input_encoder_tree=prepared_syntax_tree,
            input_encoder_measurement=[data_dict["data_frame"]],
        )
    else:
        pass


def get_run_name(config_name: str, architecture: str, game_name: str) -> str:
    """Macro function to wrap various ModelConfig properties into a run name."""
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config_name}_{architecture}_{game_name}_{time}"


if __name__ == "__main__":
    run()
