#!/usr/bin/env python3

import numpy as np

def get_first_neighbor(
        n_atom: int,
        edge_src: list,
        edge_dst: list,
        max_neighbor: int = 4
    ):
    """Get the first neighbor of each atom.

    Parameters
    ----------
    n_atom : int
        Number of atoms in the system.
    edge_src : list
        List of source atoms in the graph.
    edge_dst : list
        List of destination atoms in the graph.
    max_neighbor : int, optional
        Padding of the neighbor list, by default 4

    Returns
    -------
    n_12 : np.ndarray
        Number of neighbors for each atom.
    i_12 : np.ndarray
        List of neighbors for each atom.
    """
    n_12 = np.zeros(n_atom, dtype=int)
    i_12 = - np.ones((n_atom, max_neighbor), dtype=int)
    for i, j in zip(edge_src, edge_dst):
        i_12[i, n_12[i]] = j
        i_12[j, n_12[j]] = i
        n_12[i] += 1
        n_12[j] += 1
    return n_12, i_12

def get_third_neighbor(
        n_atom: int,
        n_12: np.ndarray,
        i_12: np.ndarray,
        max_neighbor: int = 4,
    ):
    """Get the third neighbor of each atom.

    Parameters
    ----------
    n_atom : int
        Number of atoms in the system.
    max_neighbor : int, optional
        Padding of the neighbor list, by default 4
    n_12 : np.ndarray, optional
        Number of first neighbors for each atom, by default None
    i_12 : np.ndarray, optional
        List of first neighbors for each atom, by default None
    
    Returns
    -------
    n_13 : np.ndarray
        Number of third neighbors for each atom.
    i_13 : np.ndarray
        List of third neighbors for each atom.
    """

    n_13 = np.zeros(n_atom, dtype=int)
    i_13 = - np.ones((n_atom, max_neighbor * 3), dtype=int)
    for i in range(n_atom):
        for j in range(n_12[i]):
            jj = i_12[i, j]
            for k in range(n_12[jj]):
                kk = i_12[jj, k]
                if kk == i:
                    continue
                if kk in i_12[i]:
                    continue
                if kk in i_13[i]:
                    continue
                i_13[i, n_13[i]] = kk
                n_13[i] += 1
    return n_13, i_13

def get_fourth_neighbor(
        n_atom: int,
        n_12: np.ndarray,
        n_13: np.ndarray,
        i_12: np.ndarray,
        i_13: np.ndarray,
        max_neighbor: int = 4,
    ):
    """Get the fourth neighbor of each atom.

    Parameters
    ----------
    n_atom : int
        Number of atoms in the system.
    n_12 : np.ndarray
        Number of first neighbors for each atom.
    n_13 : np.ndarray
        Number of third neighbors for each atom.
    i_12 : np.ndarray
        List of first neighbors for each atom.
    i_13 : np.ndarray
        List of third neighbors for each atom.
    max_neighbor : int, optional
        Padding of the neighbor list, by default 4

    Returns
    -------
    n_14 : np.ndarray
        Number of fourth neighbors for each atom.
    i_14 : np.ndarray
        List of fourth neighbors for each atom.
    """

    n_14 = np.zeros(n_atom, dtype=int)
    i_14 = - np.ones((n_atom, max_neighbor * 9), dtype=int)
    for i in range(n_atom):
        for j in range(n_13[i]):
            jj = i_13[i, j]
            for k in range(n_12[jj]):
                kk = i_12[jj, k]
                if kk == i:
                    continue
                if kk in i_12[i]:
                    continue
                if kk in i_13[i]:
                    continue
                if kk in i_14[i]:
                    continue
                i_14[i, n_14[i]] = kk
                n_14[i] += 1
    return n_14, i_14

def bonded_order_sparse_to_dense(bond_order, edge_src, edge_dst, n_atom):
    """Convert the sparse bond order to dense bond order.

    Parameters
    ----------
    bond_order : np.ndarray
        Bond order between atoms.
    edge_src : np.ndarray
        Source atoms in the graph.
    edge_dst : np.ndarray
        Destination atoms in the graph.
    n_atom : int
        Number of atoms in the system.

    Returns
    -------
    np.ndarray
        Dense bond order matrix.
    """
    bond_order_dense = np.zeros((n_atom, n_atom))
    for i, j, bo in zip(edge_src, edge_dst, bond_order):
        bond_order_dense[i, j] = bo
        bond_order_dense[j, i] = bo
    return bond_order_dense

if __name__ == "__main__":
    pass