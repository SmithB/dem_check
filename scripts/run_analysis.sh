#! /usr/bin/env bash
# Summaries of the test runs: per-DEM table + AUCs, filter-vs-class, fit residuals by class.
# Reads the test sets from the repository and writes to $LDCHECK_WORK/results.
REPO=$(cd $(dirname $0)/.. && pwd)
WORK=${LDCHECK_WORK:-/home/ben/DEM_regions/SPK/landsat_DEM_check}
PY=${LDCHECK_PYTHON:-python}
QA=${LDCHECK_QA:-/Volumes/ice1/ben/REMA/landsat_DEM_check/qa}
cd $WORK
$PY $REPO/scripts/summarize_tests.py $REPO/test_sets/test_set.csv $QA results/test
$PY $REPO/scripts/compare_filter_class.py results/test_summary.csv $QA results/test_filter_vs_class.csv
$PY $REPO/scripts/fit_residuals_by_class.py results/fit_points_test_set.h5 results/test_summary.csv $QA results/test_fit_residuals_by_class.csv
