import json
import unittest
from collections import Counter
from unittest.mock import patch

from simpletuner.helpers.data_backend.cursor import DatasetCursor
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted


def make_buckets(**sizes):
    return {
        bucket[1:].replace("_", "."): [f"{bucket}/img{index:03d}.png" for index in range(size)]
        for bucket, size in sizes.items()
    }


def make_cursor(buckets, **overrides):
    options = dict(
        batch_size=2,
        repeats=0,
        seed=7,
        dp_size=1,
        dp_rank=0,
        gradient_accumulation_steps=1,
        allow_oversubscription=False,
        shared_by_all_ranks=False,
    )
    options.update(overrides)
    return DatasetCursor("ds", buckets, **options)


def drain(cursor):
    batches = []
    while True:
        try:
            batches.append(cursor.next_batch())
        except MultiDatasetExhausted:
            return batches


def samples(batches):
    return [ref for _, batch in batches for ref in batch]


def share_pairs(cursor):
    return [(bucket, ref) for bucket, refs in cursor.rank_share().items() for ref in refs]


def per_bucket(batches):
    ordered = {}
    for bucket, batch in batches:
        ordered.setdefault(bucket, []).extend(batch)
    return ordered


class TestEpoch(unittest.TestCase):
    def test_every_sample_appears_once_per_pass(self):
        buckets = make_buckets(b1_0=8, b1_5=4)
        for repeats in (0, 2):
            cursor = make_cursor(buckets, repeats=repeats)
            batches = drain(cursor)
            self.assertEqual(len(batches), len(cursor))
            counts = Counter(samples(batches))
            self.assertEqual(set(counts), set(buckets["1.0"] + buckets["1.5"]))
            self.assertEqual(set(counts.values()), {repeats + 1})

    def test_a_batch_holds_one_bucket(self):
        buckets = make_buckets(b1_0=8, b1_5=4)
        for bucket, batch in drain(make_cursor(buckets)):
            self.assertEqual(len(batch), 2)
            self.assertTrue(set(batch) <= set(buckets[bucket]))

    def test_short_final_batch_is_padded_from_the_head_of_the_bucket_order(self):
        cursor = make_cursor(make_buckets(b1_0=5), batch_size=4)
        order = samples(drain(cursor))
        self.assertEqual(len(order), 8)
        self.assertEqual(len(set(order[:5])), 5)
        self.assertEqual(order[5:], order[:3])

    def test_same_seed_gives_the_same_batches(self):
        buckets = make_buckets(b1_0=9, b1_5=6)
        self.assertEqual(drain(make_cursor(buckets)), drain(make_cursor(buckets)))
        self.assertNotEqual(drain(make_cursor(buckets)), drain(make_cursor(buckets, seed=8)))

    def test_bucket_list_order_does_not_change_the_batches(self):
        buckets = make_buckets(b1_0=9, b1_5=6)
        reordered = {bucket: list(reversed(refs)) for bucket, refs in reversed(buckets.items())}
        self.assertEqual(drain(make_cursor(buckets)), drain(make_cursor(reordered)))

    def test_exhaustion_starts_the_next_epoch_with_a_new_order(self):
        cursor = make_cursor(make_buckets(b1_0=16))
        first = drain(cursor)
        self.assertEqual(cursor.epoch, 2)
        second = drain(cursor)
        self.assertEqual(cursor.epoch, 3)
        self.assertEqual(sorted(samples(first)), sorted(samples(second)))
        self.assertNotEqual(first, second)

    def test_finish_epoch_drops_the_rest(self):
        cursor = make_cursor(make_buckets(b1_0=16))
        cursor.next_batch()
        cursor.finish_epoch()
        self.assertEqual(cursor.epoch, 2)
        self.assertEqual(len(drain(cursor)), len(cursor))

    def test_bucket_lists_are_left_alone_and_later_edits_do_not_reach_the_cursor(self):
        buckets = make_buckets(b1_0=8)
        original = list(buckets["1.0"])
        cursor = make_cursor(buckets)
        self.assertEqual(buckets["1.0"], original)
        buckets["1.0"].pop()
        self.assertEqual(sorted(samples(drain(cursor))), original)

    def test_empty_buckets_are_skipped(self):
        buckets = make_buckets(b1_0=4)
        buckets["2.0"] = []
        cursor = make_cursor(buckets)
        self.assertEqual({bucket for bucket, _ in drain(cursor)}, {"1.0"})
        self.assertEqual(set(cursor.rank_share()), {"1.0"})


class TestRankShare(unittest.TestCase):
    def test_shares_are_disjoint_and_cover_each_bucket(self):
        buckets = make_buckets(b1_0=37, b1_5=16)
        shares = [share_pairs(make_cursor(buckets, dp_size=4, dp_rank=rank)) for rank in range(4)]
        combined = [entry for share in shares for entry in share]
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(sorted(combined), sorted((bucket, ref) for bucket, refs in buckets.items() for ref in refs))

    def test_share_does_not_change_between_epochs(self):
        cursor = make_cursor(make_buckets(b1_0=37), dp_size=4, dp_rank=1)
        share = cursor.rank_share()
        first = drain(cursor)
        self.assertEqual(cursor.rank_share(), share)
        self.assertEqual(set(samples(first)), set(share["1.0"]))
        self.assertEqual(set(samples(drain(cursor))), set(share["1.0"]))

    def test_ranks_have_equal_lengths_and_the_same_bucket_schedule(self):
        buckets = make_buckets(b1_0=37, b1_5=18, b0_5=9)
        cursors = [make_cursor(buckets, dp_size=4, dp_rank=rank) for rank in range(4)]
        self.assertEqual(len({len(cursor) for cursor in cursors}), 1)
        schedules = [[bucket for bucket, _ in drain(cursor)] for cursor in cursors]
        self.assertEqual(len(schedules[0]), len(cursors[0]))
        for schedule in schedules[1:]:
            self.assertEqual(schedule, schedules[0])

    def test_a_shorter_share_wraps_to_its_own_head(self):
        buckets = make_buckets(b1_0=9)
        long_rank = make_cursor(buckets, batch_size=1, dp_size=4, dp_rank=0)
        short_rank = make_cursor(buckets, batch_size=1, dp_size=4, dp_rank=3)
        self.assertEqual(len(long_rank.rank_share()["1.0"]), 3)
        self.assertEqual(len(short_rank.rank_share()["1.0"]), 2)
        order = samples(drain(short_rank))
        self.assertEqual(len(order), 3)
        self.assertEqual(order[2], order[0])

    def test_padding_cycles_through_a_shorter_share(self):
        short_rank = make_cursor(make_buckets(b1_0=9), batch_size=2, dp_size=4, dp_rank=3)
        order = samples(drain(short_rank))
        self.assertEqual(len(order), 4)
        self.assertEqual(order[2:], order[:2])

    def test_a_rank_without_a_sample_of_its_own_borrows_one(self):
        buckets = make_buckets(b1_0=3)
        cursors = [make_cursor(buckets, batch_size=1, repeats=7, dp_size=8, dp_rank=rank) for rank in range(8)]
        self.assertEqual({len(cursor) for cursor in cursors}, {8})
        owners = Counter(ref for cursor in cursors for ref in cursor.rank_share()["1.0"])
        self.assertEqual(sorted(owners.values()), [2, 3, 3])

    def test_shared_dataset_is_whole_on_every_rank(self):
        buckets = make_buckets(b1_0=3)
        for rank in range(4):
            cursor = make_cursor(
                buckets, batch_size=4, dp_size=4, dp_rank=rank, gradient_accumulation_steps=4, shared_by_all_ranks=True
            )
            self.assertEqual(sorted(cursor.rank_share()["1.0"]), buckets["1.0"])


class TestBucketSizeCheck(unittest.TestCase):
    def test_undersized_bucket_is_a_configuration_error(self):
        with self.assertRaisesRegex(ValueError, "zero usable batches") as raised:
            make_cursor(make_buckets(b1_0=64, b1_5=3), batch_size=2, dp_size=4)
        self.assertIn("Bucket 1.5: 3 samples", str(raised.exception))
        self.assertIn("Effective batch size: 8", str(raised.exception))
        self.assertNotIn("Bucket 1.0", str(raised.exception))

    def test_gradient_accumulation_counts_towards_the_effective_batch_size(self):
        buckets = make_buckets(b1_0=6)
        make_cursor(buckets, batch_size=2, gradient_accumulation_steps=3)
        with self.assertRaisesRegex(ValueError, "Effective batch size: 8"):
            make_cursor(buckets, batch_size=2, gradient_accumulation_steps=4)

    def test_repeats_count_towards_the_bucket_size(self):
        make_cursor(make_buckets(b1_0=3), batch_size=2, dp_size=4, repeats=2)

    def test_oversubscription_pads_only_the_undersized_bucket(self):
        buckets = make_buckets(b1_0=64, b1_5=3)
        cursors = [
            make_cursor(buckets, batch_size=2, dp_size=4, dp_rank=rank, allow_oversubscription=True) for rank in range(4)
        ]
        self.assertEqual({len(cursor) for cursor in cursors}, {8 + 2})
        drawn = Counter(ref for cursor in cursors for ref in samples(drain(cursor)))
        self.assertEqual({drawn[ref] for ref in buckets["1.0"]}, {1})
        self.assertEqual(sorted(drawn[ref] for ref in buckets["1.5"]), [5, 5, 6])

    def test_oversubscription_does_not_override_manual_repeats(self):
        with self.assertRaisesRegex(ValueError, "manually set repeats=1"):
            make_cursor(make_buckets(b1_5=3), batch_size=2, dp_size=4, repeats=1, allow_oversubscription=True)

    def test_an_empty_bucket_is_not_an_undersized_bucket(self):
        buckets = make_buckets(b1_0=2)
        buckets["1.5"] = []
        with self.assertRaisesRegex(ValueError, "zero usable batches") as raised:
            make_cursor(buckets, batch_size=1, dp_size=8)
        self.assertNotIn("Bucket 1.5", str(raised.exception))
        cursors = [
            make_cursor(buckets, batch_size=1, dp_size=8, dp_rank=rank, allow_oversubscription=True) for rank in range(8)
        ]
        self.assertEqual([len(cursor) for cursor in cursors], [1] * 8)
        self.assertEqual(Counter(ref for cursor in cursors for ref in samples(drain(cursor))), Counter(buckets["1.0"] * 4))

    def test_shared_dataset_is_not_checked_against_the_training_batch_size(self):
        make_cursor(make_buckets(b1_0=1), batch_size=4, dp_size=4, gradient_accumulation_steps=4, shared_by_all_ranks=True)


class TestState(unittest.TestCase):
    buckets = make_buckets(b1_0=13, b1_5=6, b0_5=4)

    def _resumed(self, state, **overrides):
        allow = overrides.pop("allow_batch_size_change", False)
        cursor = make_cursor(self.buckets, **{"repeats": 1, **overrides})
        cursor.load_state(json.loads(json.dumps(state)), allow_batch_size_change=allow)
        return cursor

    def test_resume_at_every_position_yields_the_rest_of_the_epoch(self):
        # Without repeats the 13-sample bucket ends on a padded batch.
        for repeats in (0, 1):
            expected = per_bucket(drain(make_cursor(self.buckets, repeats=repeats)))
            total = len(make_cursor(self.buckets, repeats=repeats))
            for position in range(total + 1):
                cursor = make_cursor(self.buckets, repeats=repeats)
                before = [cursor.next_batch() for _ in range(position)]
                resumed = self._resumed(cursor.state(), repeats=repeats)
                after = drain(resumed)
                self.assertEqual(len(before) + len(after), total)
                self.assertEqual(per_bucket(before + after), expected, f"repeats={repeats} position={position}")
                self.assertEqual(resumed.epoch, 2)

    def test_resume_keeps_the_epoch_and_the_saved_seed(self):
        cursor = make_cursor(self.buckets, repeats=1, dp_size=2, dp_rank=1)
        drain(cursor)
        cursor.next_batch()
        resumed = self._resumed(cursor.state(), seed=99, dp_size=2, dp_rank=1)
        self.assertEqual(resumed.epoch, 2)
        self.assertEqual(resumed.rank_share(), cursor.rank_share())
        self.assertEqual(per_bucket(drain(resumed)), per_bucket(drain(cursor)))

    def test_old_format_state_is_refused(self):
        cursor = make_cursor(self.buckets)
        for state in ({}, {"current_epoch": 3, "aspect_ratio_bucket_indices": {}, "seen_images": {}}):
            with self.assertRaisesRegex(ValueError, "older release"):
                cursor.load_state(state, allow_batch_size_change=True)

    def test_changed_layout_is_refused(self):
        state = make_cursor(self.buckets, repeats=1, dp_size=2, dp_rank=0).state()
        for overrides in ({"dp_size": 4}, {"dp_size": 2, "dp_rank": 1}, {"dp_size": 2, "repeats": 2}):
            options = {"repeats": 1, "dp_rank": 0, **overrides}
            with self.assertRaisesRegex(ValueError, "different dataset layout"):
                make_cursor(self.buckets, **options).load_state(state, allow_batch_size_change=True)

    def test_changed_dataset_starts_the_epoch_over(self):
        cursor = make_cursor(self.buckets, repeats=1)
        drain(cursor)
        cursor.next_batch()
        grown = {**self.buckets, "1.0": self.buckets["1.0"] + ["b1_0/new.png"]}
        resumed = make_cursor(grown, repeats=1, seed=99)
        with patch("simpletuner.helpers.data_backend.cursor.logger") as logger:
            resumed.load_state(json.loads(json.dumps(cursor.state())), allow_batch_size_change=False)
        logger.warning.assert_called_once()
        self.assertIn("23 samples then, 24 now", logger.warning.call_args.args[0])
        self.assertEqual(resumed.epoch, 2)
        self.assertEqual(resumed.state()["seed"], 7)
        counts = Counter(samples(drain(resumed)))
        self.assertEqual(len(counts), 24)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(resumed.epoch, 3)

    def test_changed_batch_size_needs_the_override(self):
        state = make_cursor(self.buckets, repeats=1).state()
        with self.assertRaisesRegex(ValueError, "SIMPLETUNER_ALLOW_MODIFYING_BSZ"):
            self._resumed(state, batch_size=3)

    def test_changed_batch_size_continues_with_the_unconsumed_samples(self):
        cursor = make_cursor(self.buckets, repeats=1)
        expected = per_bucket(drain(make_cursor(self.buckets, repeats=1)))
        before = per_bucket(cursor.next_batch() for _ in range(9))
        resumed = self._resumed(cursor.state(), batch_size=3, allow_batch_size_change=True)
        after = per_bucket(drain(resumed))
        for bucket, refs in expected.items():
            consumed = before.get(bucket, [])
            natural = len(self.buckets[bucket]) * 2
            self.assertEqual((consumed + after.get(bucket, []))[:natural], refs[:natural], bucket)
        self.assertEqual(resumed.epoch, 2)


if __name__ == "__main__":
    unittest.main()
