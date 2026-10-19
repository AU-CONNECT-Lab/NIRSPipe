"""Subjects run on threads under --nprocs and query one layout, whose database session is not thread-safe."""

from __future__ import annotations

import threading

from nirspipe.io.bids import channel_status_bads, get_layout, get_nirs_files


def test_threads_querying_one_layout_at_once_all_get_their_files(mini_bids):
    layout = get_layout(mini_bids, validate=False)
    subjects = ["01", "02"] * 4
    expected = {s: get_nirs_files(get_layout(mini_bids, validate=False), subject=s)
                for s in set(subjects)}
    barrier = threading.Barrier(len(subjects), timeout=30)
    results, errors = {}, []

    def query(i, subject):
        try:
            barrier.wait()
            files = get_nirs_files(layout, subject=subject)
            for f in files:
                layout.parse_file_entities(str(f))
                channel_status_bads(layout, f)
                layout.get_metadata(str(f))
            results[i] = files
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=query, args=(i, s)) for i, s in enumerate(subjects)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(results[i] == expected[s] and results[i] for i, s in enumerate(subjects))


def test_the_layout_still_answers_as_a_layout(mini_bids):
    layout = get_layout(mini_bids, validate=False)
    assert sorted(layout.get_subjects()) == ["01", "02"]
    assert str(layout.root).endswith("bids")
