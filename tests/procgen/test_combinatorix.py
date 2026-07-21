"""
Tests for ``jaxgmg.procgen.combinatorix`` — the counting/enumeration helpers
underlying the keys-and-chests combinatorial oracle.

The keys oracle enumerates the orders in which an optimal agent could collect
keys and open chests; correctness of that oracle rests on these primitives
producing exactly the right set of sequences (no duplicates, no omissions, all
valid). We check counts against the closed-form formulas and the enumerated
sets against independent ``itertools`` brute force.
"""

import math
import itertools

import numpy as np

from jaxgmg.procgen import combinatorix


# --- counting helpers match the closed forms ------------------------------ #

def test_num_combinations_matches_math():
    for n in range(0, 8):
        for r in range(0, n + 1):
            assert combinatorix.num_combinations(n, r) == math.comb(n, r)


def test_num_permutations_matches_math():
    for n in range(0, 8):
        for r in range(0, n + 1):
            assert combinatorix.num_permutations(n, r) == math.perm(n, r)


def test_num_associations_matches_catalan():
    # Catalan number C(n) = comb(2n, n) / (n + 1).
    catalan = [1, 1, 2, 5, 14, 42, 132, 429]
    for n, expected in enumerate(catalan):
        assert combinatorix.num_associations(n) == expected


# --- combinations ---------------------------------------------------------- #

def test_combinations_match_itertools():
    for n in range(1, 7):
        for r in range(0, n + 1):
            got = np.asarray(combinatorix.combinations(n, r))
            expected = list(itertools.combinations(range(n), r))
            assert got.shape == (math.comb(n, r), r)
            # same set of rows (order is explicitly not guaranteed lexicographic)
            got_set = {tuple(int(x) for x in row) for row in got}
            assert got_set == set(expected)


def test_combinations_rows_are_strictly_increasing_subsets():
    for n in range(1, 7):
        for r in range(1, n + 1):
            got = np.asarray(combinatorix.combinations(n, r))
            for row in got:
                assert list(row) == sorted(row)         # increasing
                assert len(set(int(x) for x in row)) == r   # distinct
                assert all(0 <= int(x) < n for x in row)    # in range


# --- permutations ---------------------------------------------------------- #

def test_permutations_match_itertools():
    for n in range(1, 7):
        for r in range(0, n + 1):
            got = np.asarray(combinatorix.permutations(n, r))
            expected = list(itertools.permutations(range(n), r))
            assert got.shape == (math.perm(n, r), r)
            got_set = {tuple(int(x) for x in row) for row in got}
            assert got_set == set(expected)
            assert len(got_set) == len(expected)       # no duplicates


def test_permutations_rows_are_distinct_within_range():
    for n in range(1, 7):
        for r in range(1, n + 1):
            got = np.asarray(combinatorix.permutations(n, r))
            for row in got:
                assert len(set(int(x) for x in row)) == r   # distinct entries
                assert all(0 <= int(x) < n for x in row)


# --- associations (Dyck words) -------------------------------------------- #

def _is_balanced(seq):
    """No prefix has more 1s ('close') than 0s ('open'); ends balanced."""
    balance = 0
    for x in seq:
        balance += 1 if x == 0 else -1
        if balance < 0:
            return False
    return balance == 0


def test_associations_count_is_catalan():
    for n in range(0, 7):
        got = np.asarray(combinatorix.associations(n))
        assert got.shape == (combinatorix.num_associations(n), 2 * n)


def test_associations_are_all_balanced_with_n_zeros_and_n_ones():
    for n in range(1, 7):
        got = np.asarray(combinatorix.associations(n))
        for row in got:
            row = [int(x) for x in row]
            assert sum(1 for x in row if x == 0) == n
            assert sum(1 for x in row if x == 1) == n
            assert _is_balanced(row), row


def test_associations_are_deduplicated():
    for n in range(1, 7):
        got = np.asarray(combinatorix.associations(n))
        rows = [tuple(int(x) for x in row) for row in got]
        assert len(set(rows)) == len(rows)


def test_associations_equal_brute_force_dyck_set():
    # the enumerated set must be exactly the set of all balanced 0/1 words.
    for n in range(0, 6):
        got = np.asarray(combinatorix.associations(n))
        got_set = {tuple(int(x) for x in row) for row in got}
        brute = {
            seq for seq in itertools.product((0, 1), repeat=2 * n)
            if seq.count(0) == n and _is_balanced(seq)
        }
        assert got_set == brute


def test_associations_first_and_last_rows_documented_form():
    # the docstring promises: first == (0,1)*n, last == (0,)*n + (1,)*n.
    for n in range(1, 7):
        rows = combinatorix.enumerate_associations(n)
        assert rows[0] == (0, 1) * n
        assert rows[-1] == (0,) * n + (1,) * n
