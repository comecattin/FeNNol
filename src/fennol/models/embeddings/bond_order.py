#!/usr/bin/env python3
"""Embedding using bond order.

Realized by Côme Cattin, 2024.
"""
import dataclasses
from typing import Callable, Dict, Sequence, Union

import flax.linen as nn
import jax

from ...utils.activations import ssp
from ..misc.encodings import RadialBasis, SpeciesEncoding
from ..misc.nets import FullyConnectedNet


class SchNETBondEmbedding(nn.Module):
    """Schnet embedding using bond order."""

    _graphs_properties: Dict
    dim: int = 64
    nlayers: int = 3
    conv_hidden: Sequence[int] = dataclasses.field(
        default_factory=lambda: [64, 64]
    )
    graph_key: str = "graph"
    embedding_key: str = "embedding"
    radial_basis: dict = dataclasses.field(default_factory=dict)
    species_encoding: dict = dataclasses.field(default_factory=dict)
    activation: Union[Callable, str] = ssp
    FID: str = "SCHNET_BOND"

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        species = inputs["species"]
        graph = inputs[self.graph_key]
        switch = graph["switch"][:, None]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        onehot = SpeciesEncoding(
            **self.species_encoding, name="SpeciesEncoding"
        )(species)

        xi_prev_layer = nn.Dense(
            self.dim, name="species_linear", use_bias=False
        )(onehot)

        distances = graph["distances"]
        radial_basis = RadialBasis(
            **{
                "end": cutoff,
                **self.radial_basis,
                "name": "RadialBasis",
            }
        )(distances)

        def atom_wise(xi, i, layer):
            return nn.Dense(
                self.dim, name=f"atom_wise_{i}_{layer}", use_bias=True
            )(xi)

        # Interaction layer
        for layer in range(self.nlayers):
            # Atom-wise
            xi = atom_wise(xi_prev_layer, 1, layer)

            # cfconv
            w_l = FullyConnectedNet(
                [*self.conv_hidden, self.dim],
                activation=self.activation,
                name=f"filter_weight_{layer}",
                use_bias=True,
            )(radial_basis)
            xi_j = xi[edge_dst]
            xi = jax.ops.segment_sum(
                self.activation(w_l) * xi_j * switch, edge_src, species.shape[0]
            )

            # Atom-wise
            xi = atom_wise(xi, 2, layer)

            # Activation
            xi = self.activation(xi)

            # Atom-wise
            xi = atom_wise(xi, 3, layer)

            # Residual connection
            xi = xi + xi_prev_layer
            xi_prev_layer = xi

        output = {
            **inputs,
            self.embedding_key: xi,
        }
        return output


if __name__ == "__main__":
    pass
