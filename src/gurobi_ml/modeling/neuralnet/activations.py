# Copyright © 2023 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Internal module to make MIP modeling of activation functions."""

import numpy as np
from gurobipy import GRB


class Identity:
    """Class to apply identity activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    """

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MIP model for identity activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        layer.gp_model.addConstr(output == layer.input @ layer.coefs + layer.intercept)


class ReLU:
    """Class to apply the ReLU activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation
    """

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MIP model for ReLU activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "mixing"):
                mixing = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
                layer.mixing = mixing
            layer.gp_model.update()

            layer.gp_model.addConstr(
                layer.mixing == layer.input @ layer.coefs + layer.intercept
            )
        else:
            mixing = layer._input
        for index in np.ndindex(output.shape):
            layer.gp_model.addGenConstrMax(
                output[index],
                [
                    mixing[index],
                ],
                constant=0.0,
                name=layer._indexed_name(index, "relu"),
            )


class SiLU:
    """Class to apply the SiLU activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation
    """

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MINLP model for SiLU activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        output.setAttr("lb", 0.0)  
        output.setAttr("ub", 1.0) 
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "mixing"):
                mixing = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
                layer.mixing = mixing
            layer.gp_model.update()

            layer.gp_model.addConstr(
                layer.mixing == layer.input @ layer.coefs + layer.intercept
            )
        else:
            mixing = layer._input
            
        if not hasattr(layer, "x"):
                x = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("x"),
                )
            
        if not hasattr(layer, "exp_x"):
                exp_x = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("exp_x"),
                )
                
        if not hasattr(layer, "exp_sum"):
                exp_sum = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("exp_sum"),
                )
                
        for index in np.ndindex(output.shape):
            layer.gp_model.addConstr(x[index] == - mixing[index])
            layer.gp_model.addGenConstrExp(x[index], exp_x[index])
            layer.gp_model.addConstr(exp_sum[index] == 1 + exp_x[index])
            layer.gp_model.addConstr(
                output[index] * exp_sum[index] == 1,
                name=layer._indexed_name(index, "silu"),
            )