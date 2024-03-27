#!/usr/bin/env python3
"""Implementation of SchNet embedding.

Done by Côme Cattin, 2024.
"""
import dataclasses
from typing import Callable, Dict, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from fennol.models.misc.encodings import RadialBasis, SpeciesEncoding
from fennol.models.misc.nets import FullyConnectedNet


class SchNetEmbedding(nn.Module):
    """SchNet embedding.

    Continuous filter convolutional neural network for modeling quantum
    interactions.

    References
    ----------
    SCHÜTT, Kristof, KINDERMANS, Pieter-Jan, SAUCEDA FELIX, Huziel Enoc, et al.
    Schnet: A continuous-filter convolutional neural network for
    modeling quantum interactions.
    Advances in neural information processing systems, 2017, vol. 30.
    https://proceedings.neurips.cc/paper_files/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf

    Parameters
    ----------
    dim : int, default=64
        The dimension of the embedding.
    nlayers : int, default=3
        The number of interaction layers.
    graph_key : str, default="graph"
        The key for the graph input.
    embedding_key : str, default="embedding"
        The key for the embedding output.
    radial_basis : dict, default={}
        The radial basis function parameters.
    species_encoding : dict, default={}
        The species encoding parameters.
    activation : Union[Callable, str], default=nn.softplus
        The activation function.
    """

    _graphs_properties: Dict
    dim: int = 64
    nlayers: int = 3
    graph_key: str = "graph"
    embedding_key: str = "embedding"
    radial_basis: dict = dataclasses.field(default_factory=dict)
    species_encoding: dict = dataclasses.field(default_factory=dict)
    activation: Union[Callable, str] = nn.softplus

    FID: str = "SCHNET"

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        onehot = SpeciesEncoding(
            **self.species_encoding,
            name="SpeciesEncoding"
        )(species)

        xi_prev_layer = onehot
        distances = graph["distances"]
        radial_basis = RadialBasis(
            **{
                "end": cutoff,
                **self.radial_basis,
                "name": "RadialBasis",
            }
        )(distances)
        # Interaction layer
        for layer in range(self.nlayers):
            # Atom-wise
            xi = nn.Dense(
                self.dim,
                name=f"atom_wise_1_layer{layer}",
                use_bias=True
            )(xi_prev_layer)

            # cfconv
            w_l = FullyConnectedNet(
                [self.dim, self.dim],
                activation=self.activation,
                name=f"filter_weight_{layer}",
                use_bias=True,
            )(radial_basis)
            xi_j = xi[edge_dst]
            xi = jax.ops.segment_sum(
                w_l * xi_j, edge_src, species.shape[0]
            )

            # Atom-wise
            xi = nn.Dense(
                self.dim,
                name=f"atom_wise_2_{layer}",
                use_bias=True
            )(xi)

            # Activation
            xi = self.activation(xi)

            # Atom-wise
            xi = nn.Dense(
                self.dim,
                name=f"atom_wise_3_{layer}",
                use_bias=True
            )(xi)

            # Residual connection
            if layer > 0:
                xi = xi + xi_prev_layer
            xi_prev_layer = xi

        output = {
            **inputs,
            self.embedding_key: xi,
        }
        return output


if __name__ == "__main__":

    from jax.random import PRNGKey

    from fennol import FENNIX

    # Test
    model_water = FENNIX(
    cutoff=5.0,
    rng_key=PRNGKey(0),
    modules={
            'embedding': {
                'module_name': 'SCHNET',
                'dim': 64,
                'nlayers': 3,
                'graph_key': 'graph',
                'embedding_key': 'embedding',
            },
        }
    )
    coordinates_water = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ]
        ]
    ).reshape(-1, 3)

    species = jnp.array(
        [
            [8, 1, 1]
        ]
    ).reshape(-1)

    natoms = jnp.array([3])
    batch_index = jnp.array([0, 0, 0])

    output_water = model_water(
        species=species,
        coordinates=coordinates_water,
        natoms=natoms,
        batch_index=batch_index
    )

