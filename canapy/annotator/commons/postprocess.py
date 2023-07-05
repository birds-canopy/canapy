# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import itertools

from collections import defaultdict

import numpy as np


def maximum_a_posteriori(logits, classes=None):
    logits = np.atleast_2d(logits)

    predictions = np.argmax(logits, axis=1)

    if classes is not None:
        predictions = np.take(classes, predictions)

    return predictions


def group_frames(seq, min_frame_nb=5, exclude=["SIL", "TRASH"]):
    """
    Group all consecutive equivalent labels in a discrete
    time sequence into a single label,
    with number of frames grouped attached. All groups of
    consecutive predictions
    composed of N < min_frame_nb items will be removed.
    """
    # first grouping
    # newseq = [s for s in seq if s not in exclude]
    # newseq = [list(g) for k, g in itertools.groupby(seq)]
    # newseq = [rseq for rseq in newseq
    #           if len(rseq) >= min_frame_nb]
    #
    # newseq = flatten(newseq)
    #
    # grouped_sequence = [(k, len(list(g))) for k, g in itertools.groupby(newseq)]
    #
    # return grouped_sequence

    win_length = min_frame_nb * 2 + 1

    new_seq = [None] * (len(seq) - win_length + 1)
    # size of the sequence minus size of the window plus the center part/frame of it
    if len(new_seq) <= 0:
        # print(seq)
        return []

    else:
        for i in range(len(new_seq)):
            win = np.roll(seq, shift=-i)[:win_length]
            groups = [(k, len(list(g))) for k, g in itertools.groupby(win)]
            # for each window, name of the syllables and number of apparition
            lengths = defaultdict(int)
            for g in groups:  # for each syllable and its number of apparition
                lengths[g[0]] += g[1]  # number of vote for each syllable in a window

            max_keys = [
                key for key, value in lengths.items() if value == max(lengths.values())
            ]
            new_seq[i] = max_keys

        new_seq = (
            [new_seq[0]] * (win_length // 2)
            + new_seq
            + [new_seq[-1]] * (win_length // 2)
        )  # to have the same length
        # of the initial sequence we copy the first and the last items at the start and at the end of the new sequence

        gp_new_seq = [[k, len(list(g))] for k, g in itertools.groupby(new_seq)]
        # first groupby to group juxtaposed frame for the same syllable
        length_new_seq = len(new_seq)

        # then, for each window where  syllables are too short or different syllables are equally present we look to
        # the environment group and choose the syllable which is the longest
        syll_prec = gp_new_seq[0]
        final_seq = gp_new_seq
        for i, frame in enumerate(gp_new_seq):
            if frame[1] < min_frame_nb and i + 1 <= len(new_seq):
                syll_suiv = gp_new_seq[i + 1]

                if syll_prec[1] > syll_suiv[1]:
                    if isinstance(syll_prec[0], str) == True:
                        final_seq[i][0] = syll_prec[0]
                    else:
                        final_seq[i][0] = syll_prec[0][0]

                else:
                    if isinstance(syll_suiv[0], str) == True:
                        final_seq[i][0] = syll_suiv[0]
                    else:
                        final_seq[i][0] = syll_suiv[0][0]

            else:
                final_seq[i][0] = final_seq[i][0][0]
                final_seq[i][1] = gp_new_seq[i][1]
                syll_prec = gp_new_seq[i]

        # print(final_seq)

        # here is a handmade groupby to group the old 'hesitant' window to the initial ones
        agg_final = []
        prec = None
        compteur = -1
        for syll, length in final_seq:
            if syll == prec:
                agg_final[compteur][1] += length
            else:
                agg_final.append([syll, length])
                prec = syll
                compteur += 1

        return agg_final
