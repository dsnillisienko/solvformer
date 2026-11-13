from solvformer.vote import vote


def test_the_largest_group_wins_and_gives_its_most_likely_program():
    cands = [('greedy', -1.0), ('a', -5.0), ('b', -2.0), ('c', -3.0)]
    assert vote(cands, [10.0, 7.0, 7.0 + 1e-9, 7.0]) == ('b', 7.0)


def test_ties_go_to_the_group_with_the_larger_logprob_sum():
    cands = [('a', -4.0), ('b', -1.0), ('c', -1.5), ('d', -4.0)]
    assert vote(cands, [1.0, 2.0, 2.0, 1.0]) == ('b', 2.0)


def test_failed_programs_do_not_vote():
    cands = [('greedy', -0.1), ('a', -9.0), ('b', -0.2)]
    assert vote(cands, [None, 3.0, None]) == ('a', 3.0)


def test_greedy_is_the_fallback_when_nothing_runs():
    assert vote([('greedy', -3.0), ('a', -1.0)], [None, None]) == ('greedy', None)
