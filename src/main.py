"""
Main function of the codebase. This file is intended to call different parts of our pipeline based on console arguments.

To add new games to the pipeline, add your string_query-class constructor to the 'game_from_name' function.
https://github.com/kaesve/muzero
"""

import warnings
import random
import gym
from datetime import datetime
from src.config import Config
from src.coach import Coach
from src.game.bitflip_env import BitFlipEnv  # required for gym.make
from src.neural_nets.equation_rule_predictor_skeleton import (
    EquationRulePredictorSkeleton,
)
from src.neural_nets.bitflip_rule_predictor_skeleton import BitFlipRulePredictorSkeleton
from src.game.find_equation_game import FindEquationGame
from src.game.gym_game import GymGame
from src.mcts.classic_mcts import ClassicMCTS
from src.mcts.amex_mcts import AmEx_MCTS
import tensorflow as tf
import numpy as np
import wandb
from definitions import ROOT_DIR
from src.utils.copy_weights import copy_dataset_encoder_weights_from_pretrained_agent
from src.utils.get_grammar import get_grammar_from_string
from src.generate_datasets.grammars import get_grammars

warnings.filterwarnings("ignore")


def run():
    args = Config.arguments_parser()
    args.ROOT_DIR = ROOT_DIR
    time_string = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    unique_dir = f"{time_string}_{args.seed}"
    wandb_path = ROOT_DIR / ".wandb" / args.experiment_name / f"{unique_dir}"
    wandb_path.mkdir(parents=True, exist_ok=True)
    wandb.init(
        entity="my_cur10s1ty-tu-darmstadt",
        config=args.__dict__,
        project="her-neural-mcts",
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

    if args.game == "equation_discovery":
        game = FindEquationGame(grammar, args, train_test_or_val="train")
        game_test = FindEquationGame(grammar, args, train_test_or_val="test")
    else:
        game = GymGame(
            args,
            gym.make(id=args.game, max_episode_steps=args.bitflip_max_steps, args=args),
        )
        game_test = GymGame(
            args,
            gym.make(id=args.game, max_episode_steps=args.bitflip_max_steps, args=args),
        )

    learn_a0(game=game, args=args, run_name=args.experiment_name, game_test=game_test)
    wandb.log({f"successful": True})


def learn_a0(game, args, run_name: str, game_test) -> None:
    """
    Train an AlphaZero agent on the given environment with the specified configuration. If specified within the
    configuration file, the function will load in a previous model along with previously generated data.
    :param game_test:
    :param args:
    :param game: Game Instance of a Game class that implements environment logic. Train agent on this environment.
    :param run_name: str Run name to store data by and annotate results.
    """
    print("Testing:", ", ".join(run_name.split("_")))

    # Extract neural network and algorithm arguments separately
    if args.game == "equation_discovery":
        rule_predictor_train = EquationRulePredictorSkeleton(
            args=args, reader_train=game.reader
        )
        rule_predictor_test = EquationRulePredictorSkeleton(
            args=args, reader_train=game_test.reader
        )
    elif args.game == "bitflip":
        rule_predictor_train = BitFlipRulePredictorSkeleton(args=args)
        rule_predictor_test = BitFlipRulePredictorSkeleton(args=args)
    else:
        rule_predictor_train = None
        rule_predictor_test = None

    checkpoint_train, manager_train = load_pretrained_net(
        args=args, rule_predictor=rule_predictor_train, game=game
    )
    checkpoint_test, _ = load_pretrained_net(
        args=args, rule_predictor=rule_predictor_test, game=game
    )
    if args.MCTS_engine == "Endgame":
        search_engine = AmEx_MCTS
    elif args.MCTS_engine == "Normal":
        search_engine = ClassicMCTS
    else:
        raise AssertionError(f"Engine: {args.MCTS_engine} not defined!")

    c = Coach(
        game=game,
        game_test=game_test,
        rule_predictor=rule_predictor_train,
        rule_predictor_test=rule_predictor_test,
        args=args,
        search_engine=search_engine,
        run_name=run_name,
        checkpoint_train=checkpoint_train,
        checkpoint_manager=manager_train,
        checkpoint_test=checkpoint_test,
    )

    c.learn()


def load_pretrained_net(args, rule_predictor, game):
    experiment_name = f"{args.experiment_name}/{args.seed}"
    net = rule_predictor.net
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
