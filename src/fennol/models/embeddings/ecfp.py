#!/usr/bin/env python3
"""Embedding using bond order.

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
from ...utils.bonded_neighbor import get_first_neighbor, bonded_order_sparse_to_dense
from ...utils.ecfp import ecfp


class ECFP(nn.Module):
    """Schnet embedding using bond order."""

    _graphs_properties: Dict
    dim: int = 64
    nlayers: int = 3
    conv_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [64, 64])
    n_bonds_type: int = 5
    sub_dim: int = 16
    ecfp_dim: int = 16
    ecfp_radius: int = 5
    graph_key: str = "graph"
    bond_order_key: str = "bond_order"
    embedding_key: str = "embedding"
    radial_basis: dict = dataclasses.field(default_factory=dict)
    species_encoding: dict = dataclasses.field(default_factory=dict)
    activation: Union[Callable, str] = ssp
    FID: str = "ECFP"

    @nn.compact
    def __call__(self, inputs):
        """Forward pass."""
        species = inputs["species"]
        graph = inputs[self.graph_key]
        switch = graph["switch"][:, None]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        bond_order = inputs[self.bond_order_key].reshape(-1, self.n_bonds_type)

        atom_bonded, bond_matrix = get_first_neighbor(
            len(species),
            edge_src,
            edge_dst
        )

        bond_order_dense = bonded_order_sparse_to_dense(
            bond_order,
            edge_src,
            edge_dst,
            len(species)
        )
        valency_without_hydrogen = jnp.ones(len(species))
        charge = jnp.zeros(len(species))
        attached_hydrogens = jnp.zeros(len(species))
        is_in_ring = jnp.zeros(len(species))

        ecfp_encode = ecfp(
            species=species,
            bond_order=bond_order_dense,
            bond_matrix=bond_matrix,
            atom_bonded=atom_bonded,
            attached_hydrogens=attached_hydrogens,
            valency_without_hydrogen=valency_without_hydrogen,
            charges=charge,
            is_in_ring=is_in_ring,
            radius=self.ecfp_radius,
            dim_encoding=self.ecfp_dim
        )


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

        # Interaction layer
        for layer in range(self.nlayers):

            si, mi = jnp.split(
                nn.Dense(
                    self.sub_dim * 2,
                    name=f"species_linear_{layer}",
                    use_bias=True,
                )(xi),
                [
                    self.sub_dim,
                ],
                axis=-1,
            )

            # cfconv
            xi_shape = radial_basis.shape[1] * mi.shape[1] * bond_order.shape[1]
            xi_j = (
                radial_basis[:, :, None, None]
                * mi[edge_dst, None, :, None]
                * bond_order[:, None, None, :]
            ).reshape(radial_basis.shape[0], xi_shape)

            dxi = jax.ops.segment_sum(xi_j * switch, edge_src, species.shape[0])

            dxi = jnp.concatenate([dxi, si], axis=-1)

            dxi = FullyConnectedNet(
                [*self.conv_hidden, self.dim],
                activation=self.activation,
                name=f"fc_{layer}",
            )(dxi)

            # Residual connection
            xi = xi + dxi
            
        output = {
            **inputs,
            self.embedding_key: xi,
            "radial_basis": radial_basis,
            "bond_order": bond_order,
        }
        return output


if __name__ == "__main__":
    pass
