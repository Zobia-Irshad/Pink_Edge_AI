#!/usr/bin/env python3
"""
Shared dataset-organisation helpers for Tools/organize_*_data.py.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Both maternal and TB had to solve the same two problems in their raw sources, and getting either
subtly wrong produces a model whose accuracy looks good and means nothing. The logic lives here
once rather than being copied per modality:

  split_groups()      splits GROUP KEYS, never images, so every augmented copy of one scan (and
                      every scan of one patient) lands wholly on one side of the train/validate
                      boundary. Both shipped COCO splits in this project leak almost completely
                      without this -- maternal: 104 of valid's 104 base scans also in train;
                      TB: 699 of valid's 703.

  dedupe_and_merge()  normalises every image once, hashes the RESULT, and uses that to drop exact
                      duplicates and MERGE any two groups that share an image. Merging matters:
                      two groups holding one identical image are the same underlying scan, so
                      dropping the duplicate alone would leave their other near-identical copies
                      straddling the split.
"""

import hashlib
import os
import re
from collections import defaultdict


def split_groups(group_keys, fraction, rng):
    """Split group KEYS into (train, validate). Shuffled with `rng`, so a seeded caller is
    reproducible. At least one group is always held out when there is anything to hold out."""
    keys = sorted(group_keys)
    rng.shuffle(keys)
    n_validate = max(1, round(len(keys) * fraction)) if keys else 0
    return keys[n_validate:], keys[:n_validate]


def dedupe_and_merge(groups, normalise, on_warning=None):
    """`groups` maps group key -> [source paths]; `normalise` turns a path into the encoded bytes
    that will be written. Returns (merged_groups, blobs, stats).

    Union-find merges groups that share an identical normalised image. Hashing the normalised
    result rather than the source bytes is deliberate: it also catches two source files that differ
    only in a way the pipeline discards (resolution, encoding, colour mode)."""
    blobs, owner, parent = {}, {}, {}

    def find(g):
        while parent.setdefault(g, g) != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_dupe, n_merged = 0, 0
    for group in sorted(groups):
        find(group)
        for path in sorted(groups[group]):
            try:
                data = normalise(path)
            except Exception as e:
                if on_warning:
                    on_warning(f"unreadable, skipped {os.path.basename(path)} ({e})")
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in owner:
                n_dupe += 1
                if find(owner[digest]) != find(group):
                    union(owner[digest], group)
                    n_merged += 1
                continue
            owner[digest] = group
            blobs[path] = data

    merged = defaultdict(list)
    for group, paths in groups.items():
        for path in paths:
            if path in blobs:
                merged[find(group)].append(path)

    stats = {"duplicates_dropped": n_dupe, "group_pairs_merged": n_merged,
             "groups_before": len(groups), "groups_after": len(merged)}
    return merged, blobs, stats


def clear_generated(path, prefixes):
    """Empty a destination folder of files this tooling generated (named `<prefix>_...`). Anything
    else is KEPT and returned, so a manually added image is never silently destroyed by a rebuild."""
    kept = []
    if not os.path.isdir(path):
        return kept
    pattern = re.compile(r"^(" + "|".join(re.escape(p) for p in prefixes) + r")_")
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            if pattern.match(name):
                os.remove(full)
            else:
                kept.append(name)
    return kept


def cap_per_group(items, max_per_group, rng):
    """Keep at most `max_per_group` images from each group. `items` is [(path, group)].

    Why this is needed on top of balance_group_wise(): that function drops WHOLE groups to reach an
    image-count target, which is right for keeping the split clean but can quietly destroy scan
    diversity when one class is far more heavily augmented than the other. Mammography is exactly
    that case -- its negatives are 348 scans with one image each, its positives 167 scans with ~7.7
    augmented copies each -- so balancing on image count alone filled the positive class with 261
    images drawn from just 35 distinct scans, and its held-out set with 87 images from only 15.

    An accuracy quoted over 87 images that are really 15 patients at ~6 correlated copies apiece
    overstates the evidence: the effective sample size is the number of scans, not of files. Capping
    first spreads the budget across every available scan instead. For a held-out split the right cap
    is 1 -- one vote per scan, no augmented copy of an image voting twice on the same estimate."""
    if not max_per_group:
        return list(items)
    by_group = defaultdict(list)
    for pair in items:
        by_group[pair[1]].append(pair)
    kept = []
    for key in sorted(by_group):
        group_items = sorted(by_group[key])
        rng.shuffle(group_items)
        kept.extend(group_items[:max_per_group])
    return kept


def balance_group_wise(items_by_class, rng, target=None):
    """Cap every class to `target` (default: the smallest class), dropping WHOLE groups rather than
    slicing one in half. `items_by_class` maps class -> [(path, group)]. An unbalanced training set
    lets a model score well by leaning on the majority class."""
    if target is None:
        target = min((len(v) for v in items_by_class.values()), default=0)
    out = {}
    for cls, items in items_by_class.items():
        if len(items) <= target:
            out[cls] = list(items)
            continue
        by_group = defaultdict(list)
        for pair in items:
            by_group[pair[1]].append(pair)
        keys = sorted(by_group)
        rng.shuffle(keys)
        kept = []
        for k in keys:
            if len(kept) >= target:
                break
            kept.extend(by_group[k])
        out[cls] = kept[:target]
    return out
