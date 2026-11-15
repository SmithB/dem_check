"""
Whole-DEM verdict from an ldcheck summary.

usage: python -m ldcheck.flag file_ldcheck.json [--filtered dem_filt.tif] [--why]
prints 'bad', 'ok' or 'unknown' (or the reason, with --why)

With --filtered (the standard dem_filter output for the strip), the verdict uses the
cells that the filter kept, i.e. the data that would go into altimetryFit:
    bad if >= KEPT_INCONS (20 %) of the kept cells are in Landsat class 4 (inconsistent).
On the SPK test sets (2026-11-14, STATUS.md) this flagged every DEM from the SPK
problem lists (lowest 25 %) and none of 25 good DEMs (highest 17 %).
Without --filtered, the whole 2-m DEM is used, which also counts areas (clouds,
unmatched blobs) that dem_filter already removes, so the rule is looser to read:
    bad if >= 30 % of the area with a shading signal is inconsistent and the signal
    covers >= 2 % of the DEM.
'unknown': the check did not finish, or there is too little clear imagery or signal.
"""
import sys
import json

KEPT_INCONS = 0.2
RULES = dict(min_signal_frac=0.02, max_incons_of_signal=0.3, min_clear=0.2)


def verdict(s, kept=None, rules=RULES):
    if s.get('status') != 'done':
        return 'unknown', f"check status {s.get('status')}"
    clear = s.get('frac_with_clear_image') or 0.
    if kept is not None:
        if kept['n_kept'] == 0:
            return 'unknown', 'the filtered DEM is empty'
        fk = kept['kept_frac_incons']
        if fk >= KEPT_INCONS:
            return 'bad', f'{100*fk:.0f}% of the filtered DEM is inconsistent with the Landsat images'
        if clear < rules['min_clear'] or kept['kept_frac_signal'] < 0.01:
            return 'unknown', f'too little clear imagery or signal (clear {100*clear:.0f}%)'
        return 'ok', f'{100*fk:.1f}% of the filtered DEM inconsistent'
    fs = s.get('frac_signal') or 0.
    fi = s.get('frac_inconsistent_of_signal') or 0.
    if fs >= rules['min_signal_frac'] and fi >= rules['max_incons_of_signal']:
        return 'bad', f'{100*fi:.0f}% of the area with shading signal is inconsistent with the images'
    if fs < rules['min_signal_frac'] or clear < rules['min_clear']:
        return 'unknown', 'too little clear imagery or signal'
    return 'ok', f'{100*fi:.0f}% of the signal area inconsistent'


def main():
    args = sys.argv[1:]
    with open(args[0]) as fh:
        s = json.load(fh)
    kept = None
    if '--filtered' in args:
        from .compare import kept_area_stats
        kept = kept_area_stats(s['out_tif'], args[args.index('--filtered') + 1])
    v, why = verdict(s, kept)
    print(why if '--why' in args else v)


if __name__ == '__main__':
    main()
