# Upstream — `parameter/msis_estimator/`

This directory is a vendored snapshot of the MSIS / MLWE security-estimate
scripts originally distributed with Kyber and Dilithium by the PQ-CRYSTALS
team.

## Source

- Project: *PQ-CRYSTALS Security Estimates* (Kyber / Dilithium)
- Repository: <https://github.com/pq-crystals/security-estimates>
- Default branch: `master`
- Upstream head at the time of this snapshot:
  `75c26949a902ca297b181375bfb7cfaf22cce784` (most recent upstream commit
  date 2021-03-16).

The upstream repository provides the Python scripts used in the Kyber and
Dilithium NIST PQC submissions to estimate the concrete cost of attacks
against MLWE and MSIS.

## License

The upstream repository does **not** ship a `LICENSE` file (the GitHub
license metadata field is `null`).  We therefore do not claim a specific
open-source license for the imported code beyond what the original authors
have indicated through the public availability of the repository for
research use.

Users who wish to redistribute or modify these scripts should contact the
PQ-CRYSTALS team for clarification.  The vendored copy here is included for
reproducibility of the concrete parameter analysis only.

## Citation

If you use this code, please cite the relevant PQ-CRYSTALS papers:

> Léo Ducas, Eike Kiltz, Tancrède Lepoint, Vadim Lyubashevsky, Peter
> Schwabe, Gregor Seiler, and Damien Stehlé.  *CRYSTALS-Dilithium:
> A Lattice-Based Digital Signature Scheme.*  IACR Transactions on
> Cryptographic Hardware and Embedded Systems, 2018(1):238–268.

> Joppe Bos, Léo Ducas, Eike Kiltz, Tancrède Lepoint, Vadim Lyubashevsky,
> John M. Schanck, Peter Schwabe, Gregor Seiler, and Damien Stehlé.
> *CRYSTALS — Kyber: A CCA-secure module-lattice-based KEM.*  IEEE European
> Symposium on Security and Privacy (EuroS&P), 2018.

The BKZ cost model in `model_BKZ.py` follows the line of work culminating in
Becker–Ducas–Gama–Laarhoven (SODA 2016) and follow-ups; see the comments in
`model_BKZ.py` for in-line references.

## Files

The vendored snapshot consists of:

- `MSIS_security.py`        — MSIS hardness estimate (used by the Lemur
  parameter scripts).
- `MSIS_security_optimized.py` — variant with a tighter search loop.  This
  file is **not present** in the upstream `master` branch and originates
  from a downstream optimisation; treat it as a local addition.
- `model_BKZ.py`            — concrete BKZ cost model.
- `proba_util.py`           — small probability/Gaussian helpers.

## Local modifications

The Lemur parameter pipeline imports `MSIS_security` from this directory and
uses it as a black-box hardness oracle from `parameter/lemur_param.sage` and
`parameter/chipmunk_param.sage`.  Minor constants such as `STEPS_b` and
`STEPS_m` may differ from upstream defaults; these are search-loop tuning
parameters, not behavioural changes to the cost model itself.
