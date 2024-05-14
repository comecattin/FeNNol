#!/usr/bin/env python3
"""Allegro style embedding using bond order.

Realized by Côme Cattin, 2024.
"""
import dataclasses
from typing import Callable, Dict, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...utils.activations import ssp
from ..misc.encodings import RadialBasis, SpeciesEncoding
from ..misc.nets import FullyConnectedNet


class AllegroBond(nn.Module):
    """Allegro embedding using bond order."""

    _graphs_properties: Dict
    dim: int = 64
    nlayers: int = 3
    conv_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [64, 64])
    init_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [64, 64])
    n_bonds_type: int = 5
    n_channel: int = 16
    graph_key: str = "graph"
    bond_order_key: str = "bond_order"
    ecfp_key: str = "ECFP"
    ecfp_dim: int = 16
    embedding_key: str = "embedding"
    radial_basis: dict = dataclasses.field(default_factory=dict)
    species_encoding: dict = dataclasses.field(default_factory=dict)
    activation: Union[Callable, str] = ssp
    FID: str = "ALLEGRO_BOND"

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        species = inputs["species"]
        graph = inputs[self.graph_key]
        switch = graph["switch"][:, None]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        
        # Topological information
        if self.is_initializing():
            ecfp = jnp.zeros((species.shape[0], self.ecfp_dim))
        else:
            ecfp = inputs[self.ecfp_key].reshape(-1, self.ecfp_dim)
        bond_order = inputs[self.bond_order_key]
        weights_bond_order = jnp.array([1, 1.5, 2, 0.5, 3, 0.25])
        bond_order = weights_bond_order[bond_order]

        onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
            species
        )

        xi = nn.Dense(self.dim, name="species_linear", use_bias=False)(
            onehot
        )

        


        distances = graph["distances"]
        
        radial_basis = RadialBasis(
            **{
                "end": cutoff,
                **self.radial_basis,
                "name": "RadialBasis",
            }
        )(distances)
        
        x_ij = jnp.concatenate(
            (
                onehot[edge_src], onehot[edge_dst],
                radial_basis,
                bond_order[:, None],
                ecfp[edge_src], ecfp[edge_dst]
            ),
            axis=-1
        )
        x_ij = FullyConnectedNet(
            [*self.init_hidden, self.dim],
            activation=self.activation,
            name="fc_0",
        )(x_ij) * switch

        uij = graph['vec'][:,:]/distances[:,None]
        uij = jnp.concatenate((jnp.ones((uij.shape[0],1)),uij),axis=1)

        v_ij = nn.Dense(
            self.n_channel,
            name="v_ij",
            use_bias=False,
        )(x_ij)[:,None,:] * uij[:,:,None]

        # Interaction layer
        for layer in range(self.nlayers):

            wij = nn.Dense(
                self.n_channel,
                name=f"w_ij_{layer}",
                use_bias=False,
            )(x_ij)[:,None,:] * uij[:,:,None]

            wi = jax.ops.segment_sum(wij, edge_src, species.shape[0])[edge_src]
            
            wi0 = wi[:,0,:]
            wi1 = wi[:,1:,:]
            vij0 = v_ij[:,0,:]
            vij1 = v_ij[:,1:,:]

            scalar1 = wi0 * vij0
            scalar2 = jnp.sum(wi1 * vij1, axis=1)

            vectorial1 = jnp.cross(wi1, vij1, axis=1)
            vectorial2 = wi0[:,None,:] * vij1
            vectorial3 = wi1 * vij0[:,None,:]

            dx_ij = jnp.concatenate(
                (x_ij,scalar1,scalar2),axis=-1
            )

            dx_ij = FullyConnectedNet(
                [*self.conv_hidden, self.dim],
                activation=self.activation,
                name=f"fc_{layer + 1}",
            )(dx_ij) * switch

            x_ij = x_ij + dx_ij

            if layer < self.nlayers - 1:
                vectorial = jnp.concatenate(
                    (vectorial1, vectorial2, vectorial3),
                    axis=-1
                )

                vectorial = nn.Dense(
                    self.n_channel,
                    name=f"vectorial_{layer}",
                    use_bias=False,
                )(vectorial)

                scalar = jnp.concatenate(
                    (scalar1, scalar2),
                    axis=-1
                )
                scalar = nn.Dense(
                    self.n_channel,
                    name=f"scalar_{layer}",
                    use_bias=True,
                )(scalar)

                v_ij = v_ij + jnp.concatenate(
                    (scalar[:,None,:], vectorial),
                    axis=1
                )
            
        output = {
            **inputs,
            self.embedding_key: x_ij,
            "radial_basis": radial_basis,
            "bond_order": bond_order,
            "ECFP": ecfp,
        }
        return output


if __name__ == "__main__":
    pass
