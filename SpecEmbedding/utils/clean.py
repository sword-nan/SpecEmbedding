from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import random
import numpy as np
from tqdm import tqdm
from matchms import Spectrum
from matchms.Spectrum import Spectrum
from matchms.filtering import (
    normalize_intensities,
    select_by_mz,
    require_minimum_number_of_peaks,
    add_parent_mass,
    default_filters,
    derive_adduct_from_name,
    derive_smiles_from_inchi,
    derive_inchi_from_smiles,
    harmonize_undefined_inchi,
    derive_inchikey_from_inchi,
    harmonize_undefined_smiles,
    repair_inchi_inchikey_smiles,
    harmonize_undefined_inchikey,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_inchi,
    is_valid_smiles,
    is_valid_inchikey
)
from matchms.importing import load_from_mgf, load_from_msp, load_from_mzxml
from sklearn.model_selection import train_test_split

from SpecEmbedding.type import TokenSequence

def count_annotations(spectra: Sequence[Spectrum], desc: str):
    inchi_seq = []
    smile_seq = []
    inchikey_seq = []
    compound_seq = []
    for spec in tqdm(spectra):
        inchi_seq.append(spec.get('inchi'))
        smile_seq.append(spec.get('smiles'))
        inchikey = spec.get('inchikey')
        compound_name = spec.get("compound_name")
        compound_seq.append(compound_name)
        if inchikey is None:
            inchikey = spec.get('inchikey_inchi')
        inchikey_seq.append(inchikey)

    inchi_count = sum([1 for x in inchi_seq if x is not None])
    smiles_count = sum([1 for x in smile_seq if x is not None])
    inchikey_count = sum([1 for x in inchikey_seq if x is not None])
    compound_count = sum([1 for x in compound_seq if x is not None])
    print(desc)
    print(f'compound_name: {compound_count}', '--',
          f'unique: {len(set(compound_seq))}')
    print(f'inchi: {inchi_count}', '--', f'unique: {len(set(inchi_seq))}')
    print(f'smiles: {smiles_count}', '--', f'unique: {len(set(smile_seq))}')
    print(f'inchikey: {inchikey_count}', '--', f'unique: {
          len(set([inchikey[:14] for inchikey in inchikey_seq if inchikey is not None]))}')


def apply_filters(s: Spectrum):
    s = default_filters(s)
    s = add_parent_mass(s)
    s = derive_adduct_from_name(s)
    return s

def read_raw_spectra(path: str):
    suffix = Path(path).suffix
    if suffix == ".mgf":
        spectra = list(load_from_mgf(path))
    elif suffix == ".msp":
        spectra = list(load_from_msp(path))
    elif suffix == ".mzxml":
        spectra = list(load_from_mzxml(path))
    else:
        raise ValueError(f"Not support the {suffix} format")
    
    spectra = [default_filters(s) for s in spectra]
    spectra = [normalize_intensities(s) for s in spectra]
    return spectra


def clean_metadata(s: Spectrum):
    s = harmonize_undefined_inchikey(s)
    s = harmonize_undefined_smiles(s)
    s = harmonize_undefined_inchi(s)
    s = repair_inchi_inchikey_smiles(s)
    return s


def clean_metadata2(s: Spectrum):
    s = derive_inchi_from_smiles(s)
    s = derive_smiles_from_inchi(s)
    s = derive_inchikey_from_inchi(s)
    return s


def seperate_spectra_by_ionmode(spectra: Sequence[Spectrum]):
    positive: Sequence[Spectrum] = []
    negative: Sequence[Spectrum] = []

    for spec in spectra:
        if spec.get('ionmode') == 'positive':
            positive.append(spec)
        elif spec.get('ionmode') == 'negative':
            negative.append(spec)
        else:
            print('Unknown ionmode', spec.get('smiles'), spec.get('ionmode'))
    return positive, negative


def minimal_processing(s: Spectrum):
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=10.0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=5)
    return s


def is_annotated(spectra: Sequence[Spectrum]):
    filtered: list[Spectrum] = []
    for s in tqdm(spectra):
        smiles = s.get('smiles')
        inchi = s.get('inchi')
        inchi_key = s.get('inchikey')
        if is_valid_inchikey(inchi_key) and len(inchi_key) > 13 and is_valid_inchi(inchi) and is_valid_smiles(smiles):
            filtered.append(s)
    return filtered


def filter_by_precursor_mz(spectra: Sequence[Spectrum]):
    filtered: list[Spectrum] = []
    for s in tqdm(spectra):
        prec_mz = s.get("precursor_mz")
        if prec_mz is not None and 10.0 < prec_mz < 1000:
            filtered.append(s)
    return filtered


def seperate_by_instrument(spectra: Sequence[Spectrum], instrument2type: dict):
    orbitrap: list[Spectrum] = []
    qtof: list[Spectrum] = []
    other: list[Spectrum] = []
    # seperated = defaultdict(list)

    for s in tqdm(spectra):
        instrument: str = s.get("instrument_type")
        instrument_type = instrument2type[instrument]
        if instrument_type == 'Orbitrap':
            orbitrap.append(s)
        elif instrument_type == 'qTOF':
            qtof.append(s)
        else:
            other.append(s)
        # seperated[instrument2type[instrument]].append(s)

    return orbitrap, qtof, other


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def get_unique_smiles(spectra):
    smiles_seq = []
    for s in spectra:
        smiles = s.get("smiles")
        smiles_seq.append(smiles)
    return np.unique(smiles_seq)


def get_ref_query(
    spectrums: Sequence[Spectrum],
    targets: Sequence[str]
):
    query = []
    reference = []
    smiles_metadata = defaultdict(list)

    for s in tqdm(spectrums, "seek for target_spectrums"):
        smiles = s.get("smiles")
        if smiles in targets:
            smiles_metadata[smiles].append(s)

    for smiles, metadata in tqdm(smiles_metadata.items(), "split query and reference set"):
        if len(metadata) == 1:
            reference.append(metadata[0])
        else:
            choice = np.random.randint(0, len(metadata))
            query.append(metadata[choice])
            indices = np.arange(len(metadata))
            indices = np.delete(indices, choice)
            reference.extend([
                metadata[index]
                for index in indices
            ])

    query = np.array(query)
    reference = np.array(reference)

    return query, reference


def split_train_test(spectra: Sequence[Spectrum], test_size: float, seed: int):
    unique_smiles = get_unique_smiles(spectra)

    train, test = train_test_split(
        unique_smiles, test_size=test_size, random_state=seed
    )

    return train, test, unique_smiles

def get_labels(
    unique_smiles,
    smiles_seq
):
    labels = []
    for smiles in smiles_seq:
        labels.append(np.where(unique_smiles == smiles)[0][0])
    return np.array(labels)


def get_classified_tokenset(
    unique_smiles: Sequence[str],
    sequences: list[TokenSequence],
    show_progress_bar: bool = True
):
    label2sequences: dict[int, list[TokenSequence]] = defaultdict(list)
    labels = []
    pbar = sequences
    if show_progress_bar:
        pbar = tqdm(sequences, total=len(sequences),
                    desc="classify the dataset")
    for seq in pbar:
        label = np.where(unique_smiles == seq["smiles"])[0][0]
        label2sequences[label].append(seq)
        labels.append(label)
    return dict(label2sequences), np.sort(np.unique(labels))


def get_smiles(sequences: list[TokenSequence], show_progress_bar: bool = True):
    pbar = sequences
    if show_progress_bar:
        pbar = tqdm(sequences, total=len(sequences), desc="get smiles")

    smiles_seq = []
    for sequence in pbar:
        smiles_seq.append(sequence["smiles"])
    return np.array(smiles_seq)