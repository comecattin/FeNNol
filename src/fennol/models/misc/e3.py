import flax.linen as nn
from typing import Any, Sequence, Callable, Union
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from typing import Optional, Tuple
from ...utils.spherical_harmonics import CG_SO3, spherical_to_cartesian_tensor

### e3nn version
try:
    import e3nn_jax as e3nn
    E3NN_AVAILABLE = True
    E3NN_EXCEPTION = None
except Exception as e:
    E3NN_AVAILABLE = False
    E3NN_EXCEPTION = e


class FullTensorProduct(nn.Module):
    """Tensor product of two spherical harmonics"""

    lmax1: int
    lmax2: int
    lmax_out: Optional[int] = None
    ignore_parity: bool = False

    @nn.compact
    def __call__(self, x1, x2) -> None:
        irreps_1 = [(l, (-1) ** l) for l in range(self.lmax1 + 1)]
        irreps_2 = [(l, (-1) ** l) for l in range(self.lmax2 + 1)]
        irreps_out = []
        lsout = []
        psout = []
        i12 = []

        lmax_out = self.lmax_out or self.lmax1 + self.lmax2

        for i1, (l1, p1) in enumerate(irreps_1):
            for i2, (l2, p2) in enumerate(irreps_2):
                for lout in range(abs(l1 - l2), l1 + l2 + 1):
                    if p1 * p2 != (-1) ** lout and not self.ignore_parity:
                        continue
                    if lout > lmax_out:
                        continue
                    lsout.append(lout)
                    psout.append(p1 * p2)
                    i12.append((i1, i2))

        lsout = np.array(lsout)
        psout = np.array(psout)
        idx = np.lexsort((psout, lsout))
        lsout = lsout[idx]
        psout = psout[idx]
        i12 = [i12[i] for i in idx]
        irreps_out = [(l, p) for l, p in zip(lsout, psout)]

        slices_1 = [0]
        for l, p in irreps_1:
            slices_1.append(slices_1[-1] + 2 * l + 1)
        slices_2 = [0]
        for l, p in irreps_2:
            slices_2.append(slices_2[-1] + 2 * l + 1)
        slices_out = [0]
        for l, p in irreps_out:
            slices_out.append(slices_out[-1] + 2 * l + 1)

        assert slices_1[-1] == (self.lmax1 + 1) ** 2
        assert slices_2[-1] == (self.lmax2 + 1) ** 2

        shape = ((self.lmax1 + 1) ** 2, (self.lmax2 + 1) ** 2, slices_out[-1])
        w3js = np.zeros(shape)
        for iout, (lout, pout) in enumerate(irreps_out):
            i1, i2 = i12[iout]
            l1, p1 = irreps_1[i1]
            l2, p2 = irreps_2[i2]
            w3j = CG_SO3(l1, l2, lout)
            scale = (2 * lout + 1) ** 0.5
            w3js[
                slices_1[i1] : slices_1[i1 + 1],
                slices_2[i2] : slices_2[i2 + 1],
                slices_out[iout] : slices_out[iout + 1],
            ] = (
                w3j * scale
            )
        w3j = jnp.asarray(w3js)

        return jnp.einsum("...a,...b,abc->...c", x1, x2, w3j), irreps_out


class FilteredTensorProduct(nn.Module):
    """Tensor product of two spherical harmonics filtered to give back the irreps of the first input"""

    lmax1: int
    lmax2: int
    lmax_out: Optional[int] = None
    ignore_parity: bool = False

    @nn.compact
    def __call__(self, x1, x2) -> None:
        irreps_1 = [(l, (-1) ** l) for l in range(self.lmax1 + 1)]
        irreps_2 = [(l, (-1) ** l) for l in range(self.lmax2 + 1)]
        lmax_out = self.lmax_out if self.lmax_out is not None else self.lmax1
        irreps_out = [(l, (-1) ** l) for l in range(lmax_out + 1)]

        slices_1 = [0]
        for l, p in irreps_1:
            slices_1.append(slices_1[-1] + 2 * l + 1)
        slices_2 = [0]
        for l, p in irreps_2:
            slices_2.append(slices_2[-1] + 2 * l + 1)
        slices_out = [0]
        for l, p in irreps_out:
            slices_out.append(slices_out[-1] + 2 * l + 1)

        shape = (x1.shape[-1], x2.shape[-1], x1.shape[-1])
        w3js = []
        for iout, (lout, pout) in enumerate(irreps_out):
            for i1, (l1, p1) in enumerate(irreps_1):
                for i2, (l2, p2) in enumerate(irreps_2):
                    if pout != p1 * p2 and not self.ignore_parity:
                        continue
                    if lout > l1 + l2 or lout < abs(l1 - l2):
                        continue
                    w3j = CG_SO3(l1, l2, lout)
                    w3j_full = np.zeros(shape)
                    scale = (2 * lout + 1) ** 0.5
                    w3j_full[
                        slices_1[i1] : slices_1[i1 + 1],
                        slices_2[i2] : slices_2[i2 + 1],
                        slices_out[iout] : slices_out[iout + 1],
                    ] = (
                        w3j * scale
                    )
                    w3js.append(w3j_full)
        npath = len(w3js)
        w3j = jnp.asarray(np.stack(w3js))

        weights = self.param("weights", jax.nn.initializers.normal(), (npath,))

        ww3j = jnp.einsum("p,pabc->abc", weights, w3j)
        return jnp.einsum("...a,...b,abc->...c", x1, x2, ww3j)


class ChannelMixing(nn.Module):
    lmax: int
    nchannels: int
    nchannels_out: Optional[int] = None
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False

    FID: str = "CHANNEL_MIXING"

    @nn.compact
    def __call__(self, inputs):
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        ########################################
        nchannels_out = self.nchannels_out or self.nchannels
        weights = self.param(
            "weights",
            jax.nn.initializers.normal(),
            (nchannels_out, self.nchannels),
        )
        out = jnp.einsum("ij,...jk->...ik", weights, x)
        if self.squeeze and nchannels_out == 1:
            out = jnp.squeeze(out, axis=-2)
        ########################################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class ChannelMixingE3(nn.Module):
    lmax: int
    nchannels: int
    nchannels_out: Optional[int] = None
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False

    FID: str = "CHANNEL_MIXING_E3"

    @nn.compact
    def __call__(self, inputs):
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        ########################################
        nrep = np.array([2 * l + 1 for l in range(self.lmax + 1)])
        nchannels_out = self.nchannels_out or self.nchannels
        weights = jnp.repeat(
            self.param(
                "weights",
                jax.nn.initializers.normal(),
                (nchannels_out, self.nchannels, self.lmax + 1),
            ),
            nrep,
            axis=-1,
        )
        out = jnp.einsum("ijk,...jk->...ik", weights, x)
        if self.squeeze and nchannels_out == 1:
            out = jnp.squeeze(out, axis=-2)
        ########################################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class SphericalToCartesian(nn.Module):
    lmax: int
    input_key: Optional[str] = None
    output_key: Optional[str] = None

    FID: str = "SPHERICAL_TO_CARTESIAN"

    @nn.compact
    def __call__(self, inputs) -> Any:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        ########################################
        out = spherical_to_cartesian_tensor(x, self.lmax)
        ########################################

        if self.input_key is not None:
            output_key = self.input_key if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out