#!/usr/bin/env python
"""Build the Crosson low-cloud Landsat list for the austral summers of 2019-2023.

The cloud record comes from the cloudFreeImages FFT classifier, which scored every
Landsat 8/9 overpass that covers the Crosson ROI polygon with the sun up
(``notebooks/results/crosson/summary.csv``, 667 overpasses, 2013-2026).  ``clear_frac``
is the fraction of the 480 m classifier cells inside the polygon called clear.
Scene IDs come from the inventory that the same pipeline built.

Output is one row per overpass (a pass may be 2-3 row-adjacent scenes that get
mosaicked), with the S3 prefix of each scene so the images can be pulled straight from
``s3://usgs-landsat`` without a STAC search -- LandsatLook STAC has dropped at least one
of these overpasses (LC09 2021-12-23 p007), and going direct also avoids the ERS login.

Usage:
    python scripts/crosson_landsat_list.py [--cfi <cloudFreeImages checkout>] \
        [--out lists/crosson_landsat_2019_2023.csv]
"""

import argparse
import os

import pandas as pd

# Jan 1 of each listed year +- 2 months.  Austral summers are labelled by their January.
SUMMERS = (2019, 2020, 2021, 2022, 2023)
WINDOW_MONTHS = 2

# clear_frac bands.  A is what the surface-RMS work used for single-image products;
# B and C are still worth having for a per-pixel check, which screens cloud itself and
# pools several images per DEM.
TIERS = (("A", 0.90), ("B", 0.70), ("C", 0.50))

S3_BUCKET = "usgs-landsat"
S3_ROOT = "collection02/level-1/standard/oli-tirs"


def window_bounds(summer, months=WINDOW_MONTHS):
    """Jan 1 of `summer` +- `months`, as a half-open [start, end) pair."""
    centre = pd.Timestamp(year=summer, month=1, day=1)
    return centre - pd.DateOffset(months=months), centre + pd.DateOffset(months=months)


def tier_of(clear_frac):
    for name, floor in TIERS:
        if clear_frac >= floor:
            return name
    return "-"


def scene_prefix(product_id):
    """s3://usgs-landsat prefix holding one scene's bands, from its L1 product ID.

    LC08_L1GT_010112_20131027_20201016_02_T2 -> .../oli-tirs/2013/010/112/<id>/
    """
    parts = product_id.split("_")
    path_row, acquired = parts[2], parts[3]
    path, row = path_row[:3], path_row[3:]
    return f"s3://{S3_BUCKET}/{S3_ROOT}/{acquired[:4]}/{path}/{row}/{product_id}/"


def build(cfi_root):
    crosson = os.path.join(cfi_root, "notebooks", "results", "crosson")
    summary = pd.read_csv(os.path.join(crosson, "summary.csv"), parse_dates=["date"])
    inventory = pd.read_csv(os.path.join(crosson, "inventory.csv"))

    # summary.csv carries the cloud scores; inventory.csv carries the scene IDs and rows.
    df = summary.merge(
        inventory[["key", "rows", "n_scenes", "ids", "roi_cover"]], on="key", how="left"
    )

    keep = []
    for summer in SUMMERS:
        start, end = window_bounds(summer)
        block = df[(df.date >= start) & (df.date < end)].copy()
        block["summer"] = summer
        block["window"] = f"{start.date()}..{end.date()}"
        keep.append(block)
    df = pd.concat(keep, ignore_index=True)

    df["tier"] = df.clear_frac.map(tier_of)
    df["download"] = df.tier != "-"
    df["scene_prefixes"] = df.ids.map(
        lambda ids: ";".join(scene_prefix(i) for i in str(ids).split(";"))
    )

    cols = [
        "summer", "window", "key", "sensor", "date", "path", "rows", "family",
        "sun_el", "sun_az", "clear_frac", "clear_rift_zone", "data_frac",
        "catalog_cc", "tier", "download", "n_scenes", "ids", "scene_prefixes",
    ]
    df = df[cols].sort_values(["summer", "date"]).reset_index(drop=True)
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    return df


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfi", default=os.path.expanduser("~/git_repos/cloudFreeImages"),
                   help="cloudFreeImages checkout holding notebooks/results/crosson")
    p.add_argument("--out", default=os.path.join(here, "lists",
                                                 "crosson_landsat_2019_2023.csv"))
    args = p.parse_args()

    df = build(args.cfi)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    picked = df[df.download]
    print(f"{len(df)} overpasses in the five summer windows, {len(picked)} to download "
          f"({picked.n_scenes.sum()} scenes)")
    print()
    print(pd.crosstab([picked.summer], [picked.family, picked.tier]).to_string())
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
