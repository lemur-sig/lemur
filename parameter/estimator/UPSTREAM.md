# Upstream — `parameter/estimator/`

This directory is a vendored snapshot of the **Lattice Estimator** by Martin
R. Albrecht and contributors.

## Source

- Project: *Lattice Estimator*
- Repository: <https://github.com/malb/lattice-estimator>
- Default branch: `main`
- Vendored at or near commit `6019056011d10d7e9c30a0d5da2d2f729fbc2eec`
  (2026-04-28); see `git log` of that repository for the authoritative
  history.

The upstream repository covers concrete cost estimates for LWE, NTRU, and
SIS, and is the de facto standard tool for evaluating the security of
lattice-based schemes.

## License

Per the upstream `README.rst`, the Lattice Estimator is distributed under the
**GNU Lesser General Public License v3 or later (LGPLv3+)**.  The full
license text is available at
<https://www.gnu.org/licenses/lgpl-3.0.txt>.

The upstream repository does not ship a top-level `LICENSE`/`COPYING` file;
the LGPLv3+ statement is the authoritative source.  This NOTICE preserves
that statement for the vendored copy.

## Citation

If you use this code, please cite the original paper:

> Martin R. Albrecht, Rachel Player, and Sam Scott.  *On the concrete
> hardness of Learning with Errors.*  Journal of Mathematical Cryptology,
> 9(3):169–203, 2015.  Pre-print: IACR ePrint Report 2015/046,
> <https://eprint.iacr.org/2015/046>.

The Lattice Estimator additionally implements cost models from a number of
follow-up works; see the comments in `cost.py`, `reduction.py`, and
`simulator.py` for the references attached to each model.

## Local modifications

This vendored copy is used unchanged for parameter selection.  No deliberate
behavioural changes have been made to the imported modules; if any local
edits are present they are limited to import-path adjustments needed to run
under SageMath from `parameter/lemur_param.sage`.
