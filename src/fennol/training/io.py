import os, io, sys
import numpy as np
from scipy.spatial.transform import Rotation
from collections import defaultdict
import pickle
import glob
from flax import traverse_util
from typing import Dict, List, Tuple, Union, Optional, Callable
from .databases import DBDataset, H5Dataset
from ..models.preprocessing import AtomPadding

import json
import yaml

try:
    import tomlkit
except ImportError:
    tomlkit = None

try:
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError(
        "PyTorch is required for training. Install the CPU version from https://pytorch.org/get-started/locally/"
    )

from ..models import FENNIX


def load_configuration(config_file: str) -> Dict[str, any]:
    if config_file.endswith(".json"):
        parameters = json.load(open(config_file))
    elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
        parameters = yaml.load(open(config_file), Loader=yaml.FullLoader)
    elif tomlkit is not None and config_file.endswith(".toml"):
        parameters = tomlkit.loads(open(config_file).read())
    else:
        supported_formats = [".json", ".yaml", ".yml"]
        if tomlkit is not None:
            supported_formats.append(".toml")
        raise ValueError(
            f"Unknown config file format. Supported formats: {supported_formats}"
        )
    return parameters


def load_dataset(
    dspath: str,
    batch_size: int,
    rename_refs=[],
    infinite_iterator=False,
    atom_padding=False,
    ref_keys=None,
    split_data_inputs=False,
    np_rng: Optional[np.random.Generator] = None,
    train_val_split=True,
    training_parameters={},
    add_flags=["training"],
):
    """
    Load a dataset from a pickle file.

    And return two iterators for training and validation batches.

    Args:
        training_parameters (dict): A dictionary with the following keys:
            - 'dspath': str. Path to the pickle file containing the dataset.
            - 'batch_size': int. Number of samples per batch.
        rename_refs (list, optional): A list of strings with the names
                of the reference properties to rename.
            Default is an empty list.

    Returns
    -------
        tuple: A tuple of two infinite iterators,
            one for training batches and one for validation batches.
            For each element in the batch, we expect a "species" key with
                the atomic numbers of the atoms in the system.
                Arrays are concatenated along the first axis and
                the following keys are added to distinguish between the systems:
                    - 'natoms': np.ndarray.
                        Array with the number of atoms in each system.
                    - 'batch_index': np.ndarray.
                        Array with the index of the system to which each atom
            if the keys "forces", "total_energy", "atomic_energies"
            or any of the elements in rename_refs are present,
            the keys are renamed by prepending "true_" to the key name.
    """

    # rename_refs = set(["forces", "total_energy", "atomic_energies"] + list(rename_refs))
    rename_refs = set(list(rename_refs))
    pbc_training = training_parameters.get("pbc_training", False)
    minimum_image = training_parameters.get("minimum_image", False)

    input_keys = [
        "species",
        "coordinates",
        "natoms",
        "batch_index",
        "total_charge",
        "flags",
    ]
    if pbc_training:
        input_keys += ["cells"]
    if atom_padding:
        input_keys += ["true_atoms", "true_sys"]

    flags = {f: None for f in add_flags}
    if minimum_image and pbc_training:
        flags["minimum_image"] = None

    additional_input_keys = set(training_parameters.get("additional_input_keys", []))
    additional_input_keys_ = set()
    for key in additional_input_keys:
        if key not in input_keys:
            additional_input_keys_.add(key)
    additional_input_keys = additional_input_keys_

    all_inputs = set(input_keys + list(additional_input_keys))

    extract_all_keys = ref_keys is None
    if ref_keys is not None:
        ref_keys = set(ref_keys)
        ref_keys_ = set()
        for key in ref_keys:
            if key not in all_inputs:
                ref_keys_.add(key)

    random_rotation = training_parameters.get("random_rotation", False)
    if random_rotation:
        assert np_rng is not None, "np_rng must be provided for adding noise."
        rotated_keys = set(
            ["coordinates", "cells", "forces", "virial_tensor", "stress_tensor"]
            + list(training_parameters.get("rotated_keys", []))
        )
        print("Applying random Rotations to the following keys:", rotated_keys)

    if pbc_training:
        print("Periodic boundary conditions are active.")
        length_nopbc = training_parameters.get("length_nopbc", 1000.0)

        def collate_fn_(batch):
            output = defaultdict(list)
            atom_shift = 0

            for i, d in enumerate(batch):
                nat = d["species"].shape[0]

                output["natoms"].append(np.asarray([nat]))
                output["batch_index"].append(np.asarray([i] * nat))
                if "total_charge" not in d:
                    total_charge = np.asarray(0.0, dtype=np.float32)
                else:
                    total_charge = np.asarray(d["total_charge"], dtype=np.float32)
                output["total_charge"].append(total_charge)
                if "cell" not in d:
                    cell = np.asarray(
                        [
                            [length_nopbc, 0.0, 0.0],
                            [0.0, length_nopbc, 0.0],
                            [0.0, 0.0, length_nopbc],
                        ]
                    )
                else:
                    cell = np.asarray(d["cell"])
                output["cells"].append(cell.reshape(1, 3, 3))

                if extract_all_keys:
                    for k, v in d.items():
                        if k in ("cell", "total_charge"):
                            continue
                        v_array = np.array(v)
                        # Shift atom number if necessary
                        if k.endswith("_atidx"):
                            v_array = v_array + atom_shift
                        output[k].append(v_array)
                else:
                    output["species"].append(np.asarray(d["species"]))
                    output["coordinates"].append(np.asarray(d["coordinates"]))
                    for k in additional_input_keys:
                        v_array = np.array(d[k])
                        # Shift atom number if necessary
                        if k.endswith("_atidx"):
                            v_array = v_array + atom_shift
                        output[k].append(v_array)
                    for k in ref_keys_:
                        v_array = np.array(d[k])
                        # Shift atom number if necessary
                        if k.endswith("_atidx"):
                            v_array = v_array + atom_shift
                        output[k].append(v_array)
                        if k + "_mask" in d:
                            output[k + "_mask"].append(np.asarray(d[k + "_mask"]))
                atom_shift += nat

            if random_rotation:
                euler_angles = np_rng.uniform(0.0, 2 * np.pi, (len(batch), 3))
                r = [
                    Rotation.from_euler("xyz", euler_angles[i]).as_matrix().T
                    for i in range(len(batch))
                ]
                for k in rotated_keys:
                    if k in output:
                        for i in range(len(batch)):
                            output[k][i] = output[k][i] @ r[i]

            for k, v in output.items():
                if v[0].ndim == 0:
                    output[k] = np.stack(v)
                else:
                    output[k] = np.concatenate(v, axis=0)
            for key in rename_refs:
                if key in output:
                    output["true_" + key] = output.pop(key)

            output["flags"] = flags
            return output

    else:

        def collate_fn_(batch):
            output = defaultdict(list)
            atom_shift = 0
            for i, d in enumerate(batch):
                if "cell" in d:
                    raise ValueError(
                        "Activate pbc_training to use periodic boundary conditions."
                    )
                nat = d["species"].shape[0]
                output["natoms"].append(np.asarray([nat]))
                output["batch_index"].append(np.asarray([i] * nat))
                if "total_charge" not in d:
                    total_charge = np.asarray(0.0, dtype=np.float32)
                else:
                    total_charge = np.asarray(d["total_charge"], dtype=np.float32)
                output["total_charge"].append(total_charge)
                if extract_all_keys:
                    for k, v in d.items():
                        if k == "total_charge":
                            continue
                        v_array = np.array(v)
                        # Shift atom number if necessary
                        if k.endswith("_atidx"):
                            v_array = v_array + atom_shift
                        output[k].append(v_array)
                else:
                    output["species"].append(np.asarray(d["species"]))
                    output["coordinates"].append(np.asarray(d["coordinates"]))
                    for k in additional_input_keys:
                        v_array = np.array(d[k])
                        # Shift atom number if necessary
                        if k.endswith("_atidx"):
                            v_array = v_array + atom_shift
                        output[k].append(v_array)
                    for k in ref_keys_:
                        v_array = np.array(d[k])
                        # Shift atom number if necessary
                        if k.endswith("_atidx"):
                            v_array = v_array + atom_shift
                        output[k].append(v_array)
                        if k + "_mask" in d:
                            output[k + "_mask"].append(np.asarray(d[k + "_mask"]))
                atom_shift += nat

            if random_rotation:
                euler_angles = np_rng.uniform(0.0, 2 * np.pi, (len(batch), 3))
                r = [
                    Rotation.from_euler("xyz", euler_angles[i]).as_matrix().T
                    for i in range(len(batch))
                ]
                for k in rotated_keys:
                    if k in output:
                        for i in range(len(batch)):
                            output[k][i] = output[k][i] @ r[i]

            for k, v in output.items():
                try:
                    if v[0].ndim == 0:
                        output[k] = np.stack(v)
                    else:
                        output[k] = np.concatenate(v, axis=0)
                except Exception as e:
                    raise Exception(f"Error in key {k}: {e}")
            for key in rename_refs:
                if key in output:
                    output["true_" + key] = output.pop(key)

            output["flags"] = flags
            return output

    collate_layers_train = [collate_fn_]
    collate_layers_valid = [collate_fn_]

    ### collate preprocessing
    # add noise to the training data
    noise_sigma = training_parameters.get("noise_sigma", None)
    if noise_sigma is not None:
        assert isinstance(noise_sigma, dict), "noise_sigma should be a dictionary"

        for sigma in noise_sigma.values():
            assert sigma >= 0, "Noise sigma should be a positive number"

        print("Adding noise to the training data:")
        for key, sigma in noise_sigma.items():
            print(f"  - {key} with sigma = {sigma}")

        assert np_rng is not None, "np_rng must be provided for adding noise."

        def collate_with_noise(batch):
            for key, sigma in noise_sigma.items():
                if key in batch and sigma > 0:
                    batch[key] += np_rng.normal(0, sigma, batch[key].shape).astype(
                        batch[key].dtype
                    )
            return batch

        collate_layers_train.append(collate_with_noise)

    if atom_padding:
        padder = AtomPadding()
        padder_state = padder.init()

        def collate_with_padding(batch):
            padder_state_up, output = padder(padder_state, batch)
            padder_state.update(padder_state_up)
            return output

        collate_layers_train.append(collate_with_padding)
        collate_layers_valid.append(collate_with_padding)

    if split_data_inputs:

        # input_keys += additional_input_keys
        # input_keys = set(input_keys)
        print("Input keys:",all_inputs)
        print("Ref keys:",ref_keys)

        def collate_split(batch):
            inputs = {}
            refs = {}
            for k, v in batch.items():
                if k in all_inputs:
                    inputs[k] = v
                if k in ref_keys:
                    refs[k] = v
                if k.endswith("_mask") and k[:-5] in ref_keys:
                    refs[k] = v
            return inputs, refs

        collate_layers_train.append(collate_split)
        collate_layers_valid.append(collate_split)

    ### apply all collate preprocessing
    if len(collate_layers_train) == 1:
        collate_fn_train = collate_layers_train[0]
    else:

        def collate_fn_train(batch):
            for layer in collate_layers_train:
                batch = layer(batch)
            return batch

    if len(collate_layers_valid) == 1:
        collate_fn_valid = collate_layers_valid[0]
    else:

        def collate_fn_valid(batch):
            for layer in collate_layers_valid:
                batch = layer(batch)
            return batch

    # dspath = training_parameters.get("dspath", None)
    print(f"Loading dataset from {dspath}...", end="")
    # print(f"   the following keys will be renamed if present : {rename_refs}")
    sharded_training = False
    if dspath.endswith(".db"):
        dataset = {}
        if train_val_split:
            dataset["training"] = DBDataset(dspath, table="training")
            dataset["validation"] = DBDataset(dspath, table="validation")
        else:
            dataset = DBDataset(dspath)
    elif dspath.endswith(".h5") or dspath.endswith(".hdf5"):
        dataset = {}
        if train_val_split:
            dataset["training"] = H5Dataset(dspath, table="training")
            dataset["validation"] = H5Dataset(dspath, table="validation")
        else:
            dataset = H5Dataset(dspath)
    elif dspath.endswith(".pkl") or dspath.endswith(".pickle"):
        with open(dspath, "rb") as f:
            dataset = pickle.load(f)
        if not train_val_split and isinstance(dataset, dict):
            dataset = dataset["training"]
    elif os.path.isdir(dspath):
        if train_val_split:
            dataset = {}
            with open(dspath + "/validation.pkl", "rb") as f:
                dataset["validation"] = pickle.load(f)
        else:
            dataset = None

        shard_files = sorted(glob.glob(dspath + "/training_*.pkl"))
        nshards = len(shard_files)
        if nshards == 0:
            raise ValueError("No dataset shards found.")
        elif nshards == 1:
            with open(shard_files[0], "rb") as f:
                if train_val_split:
                    dataset["training"] = pickle.load(f)
                else:
                    dataset = pickle.load(f)
        else:
            print(f"Found {nshards} dataset shards.")
            sharded_training = True

    else:
        raise ValueError(
            f"Unknown dataset format. Supported formats: '.db', '.h5', '.pkl', '.pickle'"
        )
    print(" done.")

    ### BUILD DATALOADERS
    # batch_size = training_parameters.get("batch_size", 16)
    shuffle = training_parameters.get("shuffle_dataset", True)
    if train_val_split:
        dataloader_validation = DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_valid,
        )

    if sharded_training:

        def iterate_sharded_dataset():
            indices = np.arange(nshards)
            if shuffle:
                assert np_rng is not None, "np_rng must be provided for shuffling."
                np_rng.shuffle(indices)
            for i in indices:
                filename = shard_files[i]
                print(f"# Loading dataset shard from {filename}...", end="")
                with open(filename, "rb") as f:
                    dataset = pickle.load(f)
                print(" done.")
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=collate_fn_train,
                )
                for batch in dataloader:
                    yield batch

        class DataLoaderSharded:
            def __iter__(self):
                return iterate_sharded_dataset()

        dataloader_training = DataLoaderSharded()
    else:
        dataloader_training = DataLoader(
            dataset["training"] if train_val_split else dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_train,
        )

    if not infinite_iterator:
        if train_val_split:
            return dataloader_training, dataloader_validation
        return dataloader_training

    def next_batch_factory(dataloader):
        while True:
            yield from dataloader

    training_iterator = next_batch_factory(dataloader_training)
    if train_val_split:
        validation_iterator = next_batch_factory(dataloader_validation)
        return training_iterator, validation_iterator
    return training_iterator


def load_model(
    parameters: Dict[str, any],
    model_file: Optional[str] = None,
    rng_key: Optional[str] = None,
) -> FENNIX:
    """
    Load a FENNIX model from a file or create a new one.

    Args:
        parameters (dict): A dictionary of parameters for the model.
        model_file (str, optional): The path to a saved model file to load.

    Returns
    -------
        FENNIX: A FENNIX model object.
    """
    print_model = parameters["training"].get("print_model", False)
    if model_file is None:
        model_file = parameters.get("model_file", None)
    if model_file is not None and os.path.exists(model_file):
        model = FENNIX.load(model_file, use_atom_padding=False)
        if print_model:
            print(model.summarize())
        print(f"Restored model from '{model_file}'.")
    else:
        assert (
            rng_key is not None
        ), "rng_key must be specified if model_file is not provided."
        model_params = parameters["model"]
        if isinstance(model_params, str):
            assert os.path.exists(
                model_params
            ), f"Model file '{model_params}' not found."
            model = FENNIX.load(model_params, use_atom_padding=False)
            print(f"Restored model from '{model_params}'.")
        else:
            model = FENNIX(**model_params, rng_key=rng_key, use_atom_padding=False)
        if print_model:
            print(model.summarize())
    return model


def copy_parameters(variables, variables_ref, params):
    def merge_params(full_path_, v, v_ref):
        full_path = "/".join(full_path_[1:]).lower()
        status = (False, "")
        for path in params:
            if full_path.startswith(path.lower()) and len(path) > len(status[1]):
                status = (True, path)
        return v_ref if status[0] else v

    flat = traverse_util.flatten_dict(variables, keep_empty_nodes=False)
    flat_ref = traverse_util.flatten_dict(variables_ref, keep_empty_nodes=False)
    return traverse_util.unflatten_dict(
        {
            k: merge_params(k, v, flat_ref[k]) if k in flat_ref else v
            for k, v in flat.items()
        }
    )


class TeeLogger(object):
    def __init__(self, name):
        self.file = io.TextIOWrapper(open(name, "wb"), write_through=True)
        self.stdout = None

    def __del__(self):
        self.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()

    def bind_stdout(self):
        if isinstance(sys.stdout, TeeLogger):
            raise ValueError("stdout already bound to a Tee instance.")
        if self.stdout is not None:
            raise ValueError("stdout already bound.")
        self.stdout = sys.stdout
        sys.stdout = self

    def unbind_stdout(self):
        if self.stdout is None:
            raise ValueError("stdout is not bound.")
        sys.stdout = self.stdout
