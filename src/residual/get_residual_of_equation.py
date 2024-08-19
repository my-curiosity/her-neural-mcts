import pandas as pd
import tensorflow as tf
from src.syntax_tree.syntax_tree import SyntaxTree
import copy
from src.syntax_tree.state import State
import warnings
import numpy as np
from src.utils.error import NonFiniteError


def get_residual_of_equation(state, function_to_get_current_tree_representation_int, logger):
    # wenn start node linker komplett:
    # rechtes kind berechne residual
    # rechtes kind prefix rest baum
    # neuer baum aus prefix
    if is_residual_calc_possible(state):
        last_child = state.syntax_tree.start_node.list_children[-1]
        sub_tree_prefix = state.syntax_tree.get_subtree_in_prefix_notion(last_child.node_id)
        residual_state = _calculate_residual(
            function_to_get_current_tree_representation_int,
            last_child,
            logger,
            state,
            sub_tree_prefix
        )

        return residual_state
    else:
        return state


def _calculate_residual(function_to_get_current_tree_representation_int, last_child, logger, state, sub_tree_prefix):
    try:

        new_data_frame = create_new_data_frame(last_child=last_child, state=state)
        new_observation = {
            'data_frame': new_data_frame,
            'current_tree_representation_str': sub_tree_prefix,
            'current_tree_representation_int': function_to_get_current_tree_representation_int(
                sub_tree_prefix.split()
            ),
            'id_last_node': state.observation['id_last_node'],
            'last_symbol': state.observation['last_symbol']
        }
        new_syntax_tree = create_sub_syntax_tree(last_child, state, sub_tree_prefix)

        new_state = State(syntax_tree=new_syntax_tree, observation=new_observation, done=state.done, production_action=state.production_action, residual_calculated=True,
                              previous_state=state.previous_state)
        return new_state
    except FloatingPointError:
        logger.debug(f"In the calculation of the residual a FloatingPointError occur"
                     f"the equation is: {state.syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n"
                     f"the dataset is: {state.observation['data_frame']} ")
        return state
    except RuntimeError as e:
        logger.debug(f"In the calculation of the residual a RuntimeError occur."
                     f"The error is {e} "
                     f"The previous state is used")
        return state
    except NotImplementedError as e:
        logger.debug(f"In the calculation of the residual a nan occur.This should never happen"
                     f"the equation is: {state.syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n"
                     f"the dataset is: {state.observation['data_frame']} ")
        return state
    except NonFiniteError as e:
        logger.debug(f"In the calculation of the residual a non or inf occur."
                     f"Old state is returned  happen ")
        return state


def create_new_data_frame(last_child, state):
    df = state.observation['data_frame']
    residual = state.syntax_tree.residual(
        node_id=last_child.node_id,
        dataset=df
    )
    new_df = copy.deepcopy(df)
    new_df['y'] = residual
    return new_df


def create_sub_syntax_tree(last_child, state, sub_tree_prefix):
    new_syntax_tree = SyntaxTree(args=state.syntax_tree.args,
                                 grammar=state.syntax_tree.grammar
                                 )

    new_syntax_tree.start_node.node_id = last_child.node_id
    new_syntax_tree.nodes_to_expand = [last_child.node_id]
    new_syntax_tree.start_node.depth = last_child.depth
    new_syntax_tree.current_depth = last_child.depth
    new_syntax_tree.dict_of_nodes[last_child.node_id] = new_syntax_tree.dict_of_nodes.pop(0)
    new_syntax_tree.constants_in_tree = copy.deepcopy(state.syntax_tree.constants_in_tree)
    new_syntax_tree.prefix_to_syntax_tree(prefix=sub_tree_prefix.split())
    new_syntax_tree.num_constants_in_complete_tree = \
        state.syntax_tree.num_constants_in_complete_tree
    return new_syntax_tree


def is_residual_calc_possible(state):
    if len(state.syntax_tree.start_node.list_children) == 0:
        return False
    if len(state.syntax_tree.nodes_to_expand) == 0:
        return False
    last_child = state.syntax_tree.start_node.list_children[-1]
    if state.syntax_tree.nodes_to_expand[0] < last_child.node_id:
        return False
    if last_child.invertible == False:
        return False
    else:
        return True
