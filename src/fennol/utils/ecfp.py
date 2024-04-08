#!/usr/bin/env python3
"""ECFP graph embeddings model."""

import numpy as np

#from .periodic_table import ATOMIC_MASSES
from fennol.utils.periodic_table import ATOMIC_MASSES

def get_hash(list):
    """Get hash of a list."""
    return hash(tuple(list))


def invariants(
    immediate_neighbors: int,
    valency_without_hydrogen: int,
    atomic_number: int,
    atomic_mass: float,
    charge: int,
    attached_hydrogens: int,
    is_in_ring: bool,
):
    """Get invariants of the atoms.

    Parameters
    ----------
    immediate_neighbors : int
        Number of immediate neighbors.
    valency_without_hydrogen : int
        Valency without hydrogen.
    atomic_number : int
        Atomic number.
    atomic_mass : float
        Atomic mass.
    charge : int
        Formal charge on the atom.
    attached_hydrogens : int
        Number of hydrogens attached to the atom.
    is_in_ring : bool
        Is the atom in a ring.

    Returns
    -------
    hash : int
        Hash of the invariants.
    """
    return get_hash(
        [
            immediate_neighbors,
            valency_without_hydrogen,
            atomic_number,
            atomic_mass,
            charge,
            attached_hydrogens,
            is_in_ring,
        ]
    )


def ecfp(
    species: np.ndarray,
    bond_order: np.ndarray,
    atom_bonded: np.ndarray,
    bond_matrix: np.ndarray,
    attached_hydrogens: np.ndarray,
    charges: np.ndarray,
    is_in_ring: np.ndarray,
    radius: int = 5,
    dim_encoding: int = 16,
):
    """ECFP graph embeddings.

    Parameters
    ----------
    species : np.ndarray
        Atomic numbers of the atoms.
        The shape of the array is (n_atoms,).
    bond_order : np.ndarray
        Bond orders between the atoms.
        The shape of the array is (n_atoms, n_atoms).
    atom_bonded : np.ndarray
        Number of atoms bonded to the atom.
        The shape of the array is (n_atoms,).
    bond_matrix : np.ndarray
        Adjacency matrix of the atoms.
        The shape of the array is (n_atoms, n_max_bond).
        The padding is done with -1.
    attached_hydrogens : np.ndarray
        Number of hydrogens attached to the atom.
        The shape of the array is (n_atoms,).
    charges : np.ndarray
        Formal charges on the atoms.
        The shape of the array is (n_atoms,).
    is_in_ring : np.ndarray
        Is the atom in a ring.
        The shape of the array is (n_atoms,).
    radius : int, optional
        Number of iterations to perform, by default 5
    dim_encoding : int, optional
        Dimension of the final encoding, by default 16

    Returns
    -------
    identifiers: np.ndarray
        One-hot encoded identifiers.
    identifiers_output: list
        List of the identifiers history.
    """

    identifiers_output = []
    identifiers = []
    # Iteration 0
    for i, atom in enumerate(bond_order):
        identifier_init = invariants(
            atom_bonded[i],
            valency_without_hydrogen[i],
            species[i],
            ATOMIC_MASSES[species[i]],
            charges[i],
            attached_hydrogens[i],
            is_in_ring[i],
        )
        identifiers.append(identifier_init)
    identifiers_output.extend(identifiers)

    # Iteration 1 to radius
    for layer in range(radius):
        new_identifiers = []
        for i, atom in enumerate(bond_matrix):
            atom_new_identifier = []
            atom_new_identifier.append((layer, identifiers[i]))
            for j, neighbor in enumerate(atom):
                atom_new_identifier.append((bond_order[i, j], identifiers[neighbor]))
            new_identifiers.append(get_hash(atom_new_identifier))
        identifiers = new_identifiers
        identifiers_output.extend(identifiers)

    identifiers = one_hot_hash_list(identifiers, dim_encoding)

    return identifiers, identifiers_output


def one_hot_hash_list(hash_list, dim):
    """Encode the hash list to one-hot encoding."""
    one_hot = np.array(hash_list) % dim
    one_hot = np.eye(dim)[one_hot]
    return one_hot


if __name__ == "__main__":
    from fennol.utils.bonded_neighbor import get_first_neighbor, bonded_order_sparse_to_dense

    species = np.array([1, 1, 6, 6, 1, 6, 1, 6, 1, 6, 1, 1, 1])

    edge_src = [2,2,3,3,5,5,7,7,9,9,9,9]
    edge_dst = [0,1,2,4,3,6,5,8,7,10,11,12]

    atom_bonded, bond_matrix = get_first_neighbor(
        len(species),
        edge_src,
        edge_dst
    )

    bond_order_spare = np.array([1,1,1.5,1,1.5,1,1.5,1,1.5,1,1,1])
    bond_order = bonded_order_sparse_to_dense(
        bond_order_spare,
        edge_src,
        edge_dst,
        len(species)
    )

    valency_without_hydrogen = np.ones(len(species))
    charge = np.zeros(len(species))
    attached_hydrogens = np.zeros(len(species))
    is_in_ring = np.zeros(len(species))

    output = ecfp(
        species=species,
        bond_order=bond_order,
        bond_matrix=bond_matrix,
        atom_bonded=atom_bonded,
        attached_hydrogens=attached_hydrogens,
        charges=charge,
        is_in_ring=is_in_ring,
        radius=5,
        dim_encoding=16,
    )
