#! /usr/bin/env bash
#
# get_pgc_2res.sh plus the Landsat shape check.
#
# Download a PGC 2-m strip DEM (with its matchtag and bitmask), filter it to
# coarser resolutions with dem_filter.py (exactly as get_pgc_2res.sh does), then
#   1. check the shape of the 2-m DEM against Landsat 8/9 pan images
#      (python -m ldcheck.check_dem; outputs in $LDCHECK_QA/<strip>/)
#   2. for each resolution R, combine the standard filtered DEM with a relaxed
#      one (looser slope/roughness limits, 2-pixel matchtag erosion), keeping
#      relaxed samples only where the images confirm the DEM's shape and
#      removing standard samples where they contradict it:
#      ${R}m_ldcheck/<name>_dem_filt.tif  (same format as the dem_filter output)
#   3. if >= 20 % of the standard filtered DEM is inconsistent with the images, list
#      it in ${R}m_ldcheck/problem_DEMs_ldcheck.txt
#      (read_DEM_data reads problem_DEMs*.txt from the GeoIndex directory)
#
# usage: get_pgc_2res_ldcheck.sh URL [--res R1 [R2 ...]]
#   URL    the strip's .json metadata URL
#   --res  the output resolutions, in m (default: 32 16).  The dem_filter.py
#          arguments are read from ../filter_defaults_${R}m.txt.
#
# The standard ${R}m/ outputs do not depend on the check.  If the check fails
# (no images, no USGS login, ...), the ${R}m_ldcheck/ output is not made and the
# raw 2-m files are kept, so that the check can be rerun.
# The ldcheck package (the dem_check repository) must be installed in the python used.
# Environment: LDCHECK_QA (check outputs), LDCHECK_PYTHON (python to use, default: python).

prog=$(basename $0)
LDCHECK_QA=${LDCHECK_QA:-/Volumes/ice1/ben/REMA/landsat_DEM_check/qa}
LDCHECK_PYTHON=${LDCHECK_PYTHON:-python}

usage() {
    echo "usage: $prog URL [--res R1 [R2 ...]]   (resolutions in m, default: 32 16)" >&2
    exit 2
}

url=""
res_list=()
while [ $# -gt 0 ]; do
    case $1 in
        --res)
            shift
            n_res=0
            while [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+m?$ ]]; do
                res_list+=(${1%m})
                n_res=$((n_res+1))
                shift
            done
            [ $n_res -gt 0 ] || { echo "$prog: --res needs at least one resolution" >&2; usage; }
            ;;
        -h|--help) usage ;;
        -*) echo "$prog: unknown option $1" >&2; usage ;;
        *)
            [ -z "$url" ] || { echo "$prog: more than one URL ($url, $1)" >&2; usage; }
            url=$1
            shift
            ;;
    esac
done
[ -n "$url" ] || usage
[ ${#res_list[@]} -gt 0 ] || res_list=(32 16)

missing=0
for res in "${res_list[@]}"; do
    [ -f ../filter_defaults_${res}m.txt ] || { echo "$prog: ../filter_defaults_${res}m.txt not found" >&2; missing=1; }
done
[ $missing -eq 0 ] || exit 1

dem_url=$(echo $url | sed 's/\.json$/_dem.tif/')
match_url=$(echo $url | sed 's/\.json$/_matchtag.tif/')
mask_url=$(echo $url | sed 's/\.json$/_bitmask.tif/')
dem_file=$(basename $dem_url)
match_file=$(basename $match_url)
mask_file=$(basename $mask_url)
base=${dem_file%_dem.tif}

filt_file_for() { echo ${2:-${1}m}/$(echo $dem_file | sed s/_2m_/_${1}m_/ | sed s/_dem.tif/_dem_filt.tif/); }

todo=()
for res in "${res_list[@]}"; do
    [ -f $(filt_file_for $res) ] && [ -f $(filt_file_for $res ${res}m_ldcheck) ] || todo+=($res)
done
[ ${#todo[@]} -gt 0 ] || { rm -f $dem_file $match_file $mask_file; exit 0; }

# download
for this_url in $url $dem_url $match_url $mask_url; do
    file=$(basename $this_url)
    [ -f $file ] && continue
    wget -O $file.part $this_url 2>&1 | grep -o "[0-9.]\+ [KM]*B/s"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "$prog: could not download $this_url" >&2
        rm -f $file.part
        exit 1
    fi
    mv $file.part $file
done

# 1. the Landsat check (once for all resolutions)
check_json=$LDCHECK_QA/$base/${base}_ldcheck.json
check_tif=$LDCHECK_QA/$base/${base}_ldcheck.tif
if [ ! -f $check_tif ]; then
    $LDCHECK_PYTHON -m ldcheck.check_dem $PWD/$dem_file --out_dir $LDCHECK_QA
fi

for res in "${todo[@]}"; do
    filt_file=$(filt_file_for $res)
    mkdir -p ${res}m ${res}m_ldcheck
    if [ ! -f $filt_file ]; then
        if ! dem_filter.py $dem_file $filt_file @../filter_defaults_${res}m.txt || [ ! -f $filt_file ]; then
            echo "$prog: the ${res}m filter step failed for $dem_file" >&2
            rm -f $filt_file
            continue
        fi
    fi
    # 2./3. combine with the check
    [ -f $check_tif ] || { echo "$prog: no Landsat check for $base; skipping ${res}m_ldcheck" >&2; continue; }
    ls_file=$(filt_file_for $res ${res}m_ldcheck)
    $LDCHECK_PYTHON -m ldcheck.edit $PWD/$dem_file $check_tif $PWD/$ls_file \
        --res $res --standard_file $PWD/$filt_file || rm -f $ls_file
    verdict=$($LDCHECK_PYTHON -m ldcheck.flag $check_json --filtered $PWD/$filt_file)
    if [ "$verdict" == "bad" ]; then
        echo "$(basename $filt_file)  # ldcheck: $($LDCHECK_PYTHON -m ldcheck.flag $check_json --filtered $PWD/$filt_file --why)" \
            >> ${res}m_ldcheck/problem_DEMs_ldcheck.txt
    fi
done

for res in "${res_list[@]}"; do
    if [ ! -f $(filt_file_for $res) ] || [ ! -f $(filt_file_for $res ${res}m_ldcheck) ]; then
        echo "$prog: ${res}m outputs for $dem_file are incomplete, keeping the raw files" >&2
        exit 1
    fi
done
rm -f $dem_file $match_file $mask_file
