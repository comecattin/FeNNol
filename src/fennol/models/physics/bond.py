import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any, Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils import AtomicUnits as au
import dataclasses
from ...utils.periodic_table import (
    D3_ELECTRONEGATIVITIES,
    D3_HARDNESSES,
    D3_VDW_RADII,
    D3_COV_RADII,
    D3_KAPPA,
    VDW_RADII,
    VALENCE_ELECTRONS,
    PAULING_ELECTRONEGATIVITY,
)


class CND4(nn.Module):
    """ Coordination number as defined in D4 dispersion correction
    
    FID : CN_D4
    """
    graph_key: str = "graph"
    """ The key for the graph input."""
    output_key: Optional[str] = None
    """ The key for the output."""
    k0: float = 7.5
    k1: float = 4.1
    k2: float = 19.09
    k3: float = 254.56
    electronegativity_factor: bool = False
    """ Whether to include electronegativity factor."""
    trainable: bool = False
    """ Whether the parameters are trainable."""

    FID: ClassVar[str]  = "CN_D4"

    @nn.compact
    def __call__(self, inputs):
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        species = inputs["species"]

        if self.trainable:
            rc = self.param("rc", lambda key: jnp.asarray(D3_COV_RADII))[species]
        else:
            rc = jnp.asarray(D3_COV_RADII)[species]
        rcij = rc[edge_src] + rc[edge_dst]
        rij = graph["distances"] / au.BOHR

        if self.trainable:
            k0 = self.k0 * jnp.abs(self.param("k0", lambda key: jnp.asarray(1.0)))
        else:
            k0 = self.k0

        CNij = (
            0.5 * (1 + jax.scipy.special.erf(-k0 * (rij / rcij - 1.))) * graph["switch"]
        )

        if self.electronegativity_factor:
            k1 = self.k1
            k2 = self.k2
            k3 = self.k3
            if self.trainable:
                k1 = k1 * jnp.abs(self.param("k1", lambda key: jnp.asarray(1.0)))
                k2 = self.param("k2", lambda key: jnp.asarray(1.0))
                k3 = jnp.abs(self.param("k3", lambda key: jnp.asarray(1.0)))
                en = self.param("en", lambda key: jnp.asarray(D3_ELECTRONEGATIVITIES))[
                    species
                ]
            else:
                en = jnp.asarray(PAULING_ELECTRONEGATIVITY)[species]
            en_ij = jnp.abs(en[edge_src] - en[edge_dst])
            dij = k1 * jnp.exp(-((en_ij + k2) ** 2) / k3)
            CNij = CNij * dij
        CNi = jax.ops.segment_sum(CNij, edge_src, species.shape[0])

        output_key = self.name if self.output_key is None else self.output_key
        return {**inputs, output_key: CNi, output_key + "_pair": CNij}


class SumSwitch(nn.Module):
    """Sum (a power of) the switch values for each neighbor.
    
    FID : SUM_SWITCH

    """
    graph_key: str = "graph"
    """ The key for the graph input."""
    output_key: Optional[str] = None
    """ The key for the output."""
    pow: float = 1.0
    """ The power to raise the switch values to."""
    trainable: bool = False
    """ Whether the pow parameter is trainable."""

    FID: ClassVar[str]  = "SUM_SWITCH"

    @nn.compact
    def __call__(self, inputs):
        graph = inputs[self.graph_key]
        edge_src = graph["edge_src"]
        switch = graph["switch"]

        if self.trainable:
            p = jnp.abs(
                self.param("pow", lambda key: jnp.asarray(self.pow))
            )
        else:
            p = self.pow
        shift=(1.e-3)**p

        cn = jax.ops.segment_sum((1.e-3+switch)**p-shift, edge_src, inputs["species"].shape[0])

        output_key = self.name if self.output_key is None else self.output_key
        return {**inputs, output_key: cn}


class CNShift(nn.Module):
    
    cn_key: str
    output_key: Optional[str] = None
    kappa_key: Optional[str] = None
    sqrt_shift: float = 1.0e-6
    ref_value: Union[str, float] = 1.0
    enforce_positive: bool = False
    cn_pow: float = 0.5

    FID: ClassVar[str]  = "CN_SHIFT"


    @nn.compact
    def __call__(self, inputs):
        CNi = inputs[self.cn_key]
        if self.kappa_key is not None:
            kappai = inputs[self.kappa_key]
            assert kappai.shape == CNi.shape
        else:
            species = inputs["species"]
            kappai = self.param("kappa", nn.initializers.zeros, (len(D3_COV_RADII),))[
                species
            ]
        shift = kappai * (CNi + self.sqrt_shift) ** self.cn_pow

        if isinstance(self.ref_value, str):
            ref_value = inputs[self.ref_value]
            assert ref_value.shape == shift.shape
        else:
            ref_value = self.ref_value

        if self.enforce_positive:
            shift = jax.nn.celu(shift, alpha=ref_value)

        out = ref_value + shift

        output_key = self.name if self.output_key is None else self.output_key
        return {**inputs, output_key: out}


class CNStore(nn.Module):
    cn_key: str
    output_key: Optional[str] = None
    store_size: int = 10
    n_gaussians: int = 4
    isolated_value: float = 0.0
    init_scale_cn: float = 5.0
    init_scale_values: float = 1.0
    beta: float = 6.0
    trainable: bool = True
    output_dim: int = 1
    squeeze: bool = True

    FID: ClassVar[str]  = "CN_STORE"

    @nn.compact
    def __call__(self, inputs):
        cn = inputs[self.cn_key]
        species = inputs["species"]

        cn_refs = self.param(
            "cn_refs",
            nn.initializers.uniform(self.init_scale_cn),
            (len(D3_COV_RADII), self.store_size),
        )[species]
        values_refs = self.param(
            "values_refs",
            nn.initializers.uniform(self.init_scale_values),
            (len(D3_COV_RADII), self.store_size, self.output_dim),
        )[species]

        beta = self.beta
        if self.trainable:
            beta = self.param("beta", lambda key: jnp.asarray(self.beta))
        j = jnp.asarray(np.arange(self.n_gaussians)[None, None, :], dtype=jnp.float32)
        delta_cns = jnp.log(
            jnp.sum(
                jnp.exp(-beta * j * ((cn[:, None] - cn_refs) ** 2)[:, :, None]), axis=-1
            )
        )
        w = jax.nn.softmax(delta_cns, axis=-1)

        values = jnp.sum(w[:, :, None] * values_refs, axis=1)
        if self.output_dim == 1 and self.squeeze:
            values = jnp.squeeze(values, axis=-1)

        output_key = self.name if self.output_key is None else self.output_key
        return {**inputs, output_key: values}
