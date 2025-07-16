from collections.abc import Sequence

import numpy as np
from tqdm import tqdm
from matchms import Spectrum

from SpecEmbedding.data.const import SpecialToken
from SpecEmbedding.type import Peak, MetaData, TokenSequence

class Tokenizer:
    def __init__(self, max_len: int, show_progress_bar: bool = True) -> None:
        """
            Tokenization of mass spectrometry data

            Parameters:
            ---
            -   max_len: Maximum number of peaks to extract
            -   show_progress_bar: Whether to display a progress bar
        """
        self.max_len = max_len
        self.show_progress_bar = show_progress_bar

    def tokenize(self, s: Spectrum):
        """
            Tokenization of mass spectrometry data
        """
        metadata = self.get_metadata(s)
        mz = []
        intensity = []
        for peak in metadata["peaks"]:
            mz.append(peak["mz"])
            intensity.append(peak["intensity"])

        mz = np.array(mz)
        intensity = np.array(intensity)
        mask = np.zeros((self.max_len, ), dtype=bool)
        if len(mz) < self.max_len:
            mask[len(mz):] = True
            mz = np.pad(
                mz, (0, self.max_len - len(mz)),
                mode='constant', constant_values=SpecialToken["PAD"]
            )

            intensity = np.pad(
                intensity, (0, self.max_len - len(intensity)),
                mode='constant', constant_values=SpecialToken["PAD"]
            )

        return TokenSequence(
            mz=np.array(mz, np.float32),
            intensity=np.array(intensity, np.float32),
            mask=mask,
            smiles=metadata["smiles"]
        )

    def tokenize_sequence(self, spectra: Sequence[Spectrum]):
        sequences: list[TokenSequence] = []
        pbar = spectra
        if self.show_progress_bar:
            pbar = tqdm(spectra, total=len(spectra), desc="tokenization")
        for s in pbar:
            sequences.append(self.tokenize(s))

        return sequences

    def get_metadata(self, s: Spectrum):
        """
            get the metadata from spectrum

            -   smiles
            -   precursor_mz
            -   peaks
        """
        precursor_mz = s.get("precursor_mz")
        smiles = s.get("smiles")
        peaks = np.array(s.peaks.to_numpy, np.float32)
        intensity = peaks[:, 1]
        argmaxsort_index = np.sort(
            np.argsort(intensity)[::-1][:self.max_len - 1]
        )
        peaks = peaks[argmaxsort_index]
        peaks[:, 1] = peaks[:, 1] / max(peaks[:, 1])
        packaged_peaks: list[Peak] = [
            Peak(
                mz=np.array(precursor_mz, np.float32),
                intensity=2
            )
        ]
        for mz, intensity in peaks:
            packaged_peaks.append(
                Peak(
                    mz=mz,
                    intensity=intensity
                )
            )
        metadata = MetaData(
            smiles=smiles,
            peaks=packaged_peaks
        )
        return metadata