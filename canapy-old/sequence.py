import math
import itertools

from collections import defaultdict

import numpy as np


def remove(seq, labels):
    """
    Remove some labels in a sequence.
    """
    filtered = seq.copy()
    for lbl in labels:
        filtered = filtered[filtered != lbl]

    return filtered


def group(seq, min_frame_nb=5, exclude=["SIL", "TRASH"]):
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


def lev_dist(s, t):
    """
    Levenshtein distance between two sequences S and T.
    Returns the minimum number of operations (insertions, substitutions, deletions)
    to perform to perfectly match S and T.
    """

    v0 = np.arange(len(t) + 1)
    v1 = np.zeros(len(t) + 1)

    for i in range(0, len(s)):
        v1[0] = i + 1

        for j in range(0, len(t)):
            delcost = v0[j + 1] + 1  # deletions
            incost = v1[j] + 1  # insertions
            if s[i] == t[j]:
                subcost = v0[j]  # substitutions
            else:
                subcost = v0[j] + 1

            v1[j + 1] = min([delcost, incost, subcost])

        v0, v1 = v1, v0

    return v0[len(t)]


def lev_sim(s, t):
    """
    Similarity score based on minimum Levenshtein distance between two
    sequences S and T.
    Returns:
    .. math::
        \frac{|s| + |t| - levenshtein(s, t)}{|s| + |t|}

    """
    return ((len(s) + len(t)) - lev_dist(s, t)) / (len(s) + len(t))


def lcs(s, t):
    """
    Longest Common Subsequence between two sequences. R
    Returns the maximum length matching subsequence between two sequences.
    Similar to the Levenshtein distance but substitutions are not allowed
    to build the matching subsequences.
    """
    c = np.zeros((len(s) + 1, len(t) + 1))

    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            if s[i - 1] == t[j - 1]:
                c[i, j] = c[i - 1, j - 1] + 1
            else:
                c[i, j] = max(c[i, j - 1], c[i - 1, j])
    return c[len(s), len(t)]


def lcs_ratio(s, t):
    """
    Similarity score based on LCS between two sequences S and T.
    Returns:
    .. math::
        \frac{lcs(s, t)}{max(|s), |t|)}
    """
    return lcs(s, t) / max(len(s), len(t))


def flatten(lists):
    """
    Flatten a list of lists into a simple list.
    """
    flat = []
    for ls in lists:
        for e in ls:
            flat.append(e)
    return flat


def ngrams(seq, n=1):
    """
    Generate ngrams from a sequence.
    """
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i : i + n])


def ngram_occurences(doc, n=1):
    """
    Frequencies of ngrams in set of sequences.
    """

    occurences = defaultdict(lambda: defaultdict(int))
    for seq in doc:
        for ng in ngrams(seq, n=n):
            occurences[ng[: n - 2]][ng[n - 1]] += 1

    for ngram in occurences:
        total = float(sum(occurences[ngram].values()))
        for s in occurences[ngram]:
            occurences[ngram][s] /= total

    return occurences


def to_seconds(grouped_annots, config):

    new_annots = {}
    # for m, model_annots in grouped_annots.items():
    #     new_annots[m] = {}
    #     for s, song_annots in model_annots.items():
    #         new_annots[m][s] = []
    #         for i, a in enumerate(song_annots):
    #             t = config.to_duration(a[1])
    #             new_a = (a[0], t)
    #             new_annots[m][s].append(new_a)
    for s, song_annots in grouped_annots.items():
        new_annots[s] = []
        for i, a in enumerate(song_annots):
            t = config.to_duration(a[1])
            new_a = (a[0], t)
            new_annots[s].append(new_a)
    return new_annots
