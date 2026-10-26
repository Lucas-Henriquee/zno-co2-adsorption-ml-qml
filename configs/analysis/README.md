# Analysis Configuration

`zno_current.yaml` connects one simulation run (`results/simulations/<run-id>`)
and one model run (`results/models/<run-id>`). Analysis output is always
written to a new directory below `results/analyses/`.

Input roots can be overridden without editing YAML:

```bash
./run_analysis.sh all \
  --scientific-results results/simulations/zno_run_01 \
  --model-run results/models/another_local_run \
  --run-id another_analysis
```

The screening step ranks only successful DFT rows (`dft_status == ok`, carbonate
excluded) by adsorption energy. Band gaps are PBE Kohn-Sham gaps
(`*_band_gap_gamma_ev`) and the TDDFT values describe the isolated CO2 molecule;
both are reported for information and are not compared with photocatalysis
targets.

The scientific and model analyses remain separate. A missing model run does not
prevent a science-only execution, and model analysis does not modify simulation
outputs.
