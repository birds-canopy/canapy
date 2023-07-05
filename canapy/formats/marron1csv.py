# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
from typing import ClassVar, Optional, Union

import attr
import pandera
import crowsetta
import numpy as np
import pandas as pd

from pandera.typing import Series
from crowsetta.typing import PathLike


class DataFormatError(Exception):
    pass


class Marron1SeqSchema(pandera.SchemaModel):
    wave: Series[pd.StringDtype] = pandera.Field(coerce=True)
    start: Series[float] = pandera.Field(coerce=True)
    end: Series[float] = pandera.Field(coerce=True)
    syll: Series[pd.StringDtype] = pandera.Field(coerce=True, nullable=True)

    class Config:
        ordered = False
        strict = True


@crowsetta.formats.register_format
@crowsetta.interface.SeqLike.register
@attr.define
class Marron1CSV:
    """Example custom annotation format"""

    name: ClassVar[str] = "marron1csv"
    ext: ClassVar[str] = ".csv"

    starts: np.ndarray = attr.field(eq=attr.cmp_using(eq=np.array_equal))
    ends: np.ndarray = attr.field(eq=attr.cmp_using(eq=np.array_equal))
    sylls: np.ndarray = attr.field(eq=attr.cmp_using(eq=np.array_equal))
    annot_path: pathlib.Path
    notated_path: Optional[pathlib.Path] = attr.field(
        default=None, converter=attr.converters.optional(pathlib.Path)
    )

    @classmethod
    def from_file(
        cls,
        annot_path: PathLike,
        notated_path: Optional[PathLike] = None,
    ) -> "Self":  # noqa: F821
        annot_path = pathlib.Path(annot_path)
        crowsetta.validation.validate_ext(annot_path, extension=cls.ext)

        df = pd.read_csv(annot_path)
        df = Marron1SeqSchema.validate(df)

        wave_files = df["wave"].unique()

        if len(wave_files) > 1:
            raise DataFormatError(
                f"An annotation file is referencing more than one audio file, in "
                f"{annot_path}: found {len(wave_files)} audio file refs - "
                f"{wave_files}"
            )
        else:
            wave_path = wave_files[0]

        if notated_path is None:
            notated_path = wave_path
        else:
            notated_path = notated_path / wave_path

        return cls(
            starts=df["start"].values,
            ends=df["end"].values,
            sylls=df["syll"].values,
            annot_path=annot_path,
            notated_path=notated_path,
        )

    def to_seq(self, round_times: bool = True, decimals: int = 3) -> crowsetta.Sequence:
        """Convert this annotation to a :class:`crowsetta.Sequence`.

        Parameters
        ----------
        round_times : bool
            If True, round ``onsets_s`` and ``offsets_s``.
            Default is True.
        decimals : int
            Number of decimals places to round floating point numbers to.
            Only meaningful if round_times is True.
            Default is 3, so that times are rounded to milliseconds.

        Returns
        -------
        seq : crowsetta.Sequence

        Examples
        --------
        >>> example = crowsetta.data.get('aud-seq')
        >>> audseq = crowsetta.formats.seq.AudSeq.from_file(example.annot_path)
        >>> seq = audseq.to_seq()

        Notes
        -----
        The ``round_times`` and ``decimals`` arguments are provided
        to reduce differences across platforms
        due to floating point error, e.g. when loading annotation files
        and then sending them to a csv file,
        the result should be the same on Windows and Linux.
        """
        if round_times:
            onsets_s = np.around(self.starts, decimals=decimals)
            offsets_s = np.around(self.ends, decimals=decimals)
        else:
            onsets_s = self.starts
            offsets_s = self.ends

        seq = crowsetta.Sequence.from_keyword(
            labels=self.sylls, onsets_s=onsets_s, offsets_s=offsets_s
        )
        return seq

    def to_annot(
        self, round_times: bool = True, decimals: int = 3
    ) -> crowsetta.Annotation:
        """Convert this annotation to a :class:`crowsetta.Annotation`.

        Parameters
        ----------
        round_times : bool
            If True, round onsets_s and offsets_s.
            Default is True.
        decimals : int
            Number of decimals places to round floating point numbers to.
            Only meaningful if round_times is True.
            Default is 3, so that times are rounded to milliseconds.

        Returns
        -------
        annot : crowsetta.Annotation

        Examples
        --------
        >>> example = crowsetta.data.get('aud-seq')
        >>> audseq = crowsetta.formats.seq.AudSeq.from_file(example.annot_path)
        >>> annot = audseq.to_annot()

        Notes
        -----
        The ``round_times`` and ``decimals`` arguments are provided
        to reduce differences across platforms
        due to floating point error, e.g. when loading annotation files
        and then sending them to a csv file,
        the result should be the same on Windows and Linux.
        """
        seq = self.to_seq(round_times, decimals)
        return crowsetta.Annotation(
            annot_path=self.annot_path, notated_path=self.notated_path, seq=seq
        )

    def to_file(self, annot_path: PathLike) -> None:
        """Save this 'aud-seq' annotation to a txt file
        in the standard/default Audacity LabelTrack format.

        Parameters
        ----------
        annot_path : str, pathlib.Path
            Path with filename of txt file that should be saved.
        """
        df = pd.DataFrame.from_records(
            {
                "wave": self.notated_path.name,
                "start": self.starts,
                "end": self.ends,
                "syll": self.sylls,
            }
        )
        df = df[["wave", "start", "end", "syll"]]  # put in correct order
        try:
            df = Marron1SeqSchema.validate(df)
        except pandera.errors.SchemaError as e:
            raise ValueError(
                f"Annotations produced an invalid dataframe, "
                f"cannot convert to "
                f"Marron1-like csv file:\n{df}"
            ) from e

        df.to_csv(annot_path, index=False)


if __name__ == "__main__":
    data = "/home/nathan/Documents/Code/canapy-test/data/Songfile_2022-05-04_16-10-53_CSC1_CSC20_raw_chunk_119_annot.csv"

    scribe = crowsetta.Transcriber(format="marron1csv")

    ext = crowsetta.formats.by_name("marron1csv").ext

    fp = scribe.from_file(data)
    seq = fp.to_seq()
    annots = fp.to_annot()
    print(seq)
    print(annots)

    print(pd.DataFrame(seq.as_dict()))

    for seg in annots.seq.segments:
        print(seg)
