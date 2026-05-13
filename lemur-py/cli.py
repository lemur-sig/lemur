#!/usr/bin/env python3
"""
cli.py — Command-line interface for the Lemur multi-signature scheme.

Each user maintains their own raw seed secret key, mutable signer state,
and public key files. The pp.bin is shared by all participants. Signatures
are stored as .sig files.

Commands
--------
  setup         Generate shared public parameters (pp.bin).
  keygen        Generate raw seed key, mutable signer state, and public key.
  sign          Sign a message at a given time slot.
  verify        Individually verify a signature (iVrfy).
  aggregate     Combine N individual signatures into one (aVrfy-ready).
  batch-verify  Verify an aggregated signature against N public keys.
  sizes         Print serialised sizes for all object types.
  vectors       Generate deterministic test vectors (JSON).

Deterministic usage
-------------------
  # Fully pinned setup:
  python cli.py setup --kots-seed 000102...1f --hvc-seed 202122...3f --out pp.bin

  # Fully pinned keygen:
  python cli.py keygen --pp pp.bin --seed 010101...01 --sk alice.sk \
      --stateful-sk alice.state --pk alice.pk

  # Test vectors (self-contained, writes JSON):
  python cli.py vectors --out vectors.json
  python cli.py vectors --signers 4 --slot 1 --msg "vote" --out vectors.json
"""

import argparse
import json
import sys
from pathlib import Path

from kots import KOTS, inf_norm
from hvc import HVC
from lemur import LEMUR
from codec import LemurCodec, make_codec, pp_peek_tau
from profiles import DEFAULT, LemurProfile

# Module-level scheme instances for the fixed parameter set.
_profile: LemurProfile = DEFAULT
_kots:    KOTS
_hvc:     HVC
_scheme:  LEMUR
_codec:   LemurCodec


def _set_profile(profile: LemurProfile) -> None:
    """(Re)build module-level scheme singletons for a given profile."""
    global _profile, _kots, _hvc, _scheme, _codec
    _profile = profile
    _kots    = KOTS(profile=profile)
    _hvc     = HVC(_kots)
    _scheme  = LEMUR(_kots, _hvc)
    _codec   = LemurCodec(_scheme)


_set_profile(DEFAULT)


def _load_pp(path: str) -> tuple:
    """Read a pp file and return `(pp, codec, scheme)`.

    Only τ is read from pp, and the codec is rebuilt at that depth if it
    differs from the module's current τ.
    """
    data = _read(path)
    tau = pp_peek_tau(data)

    if tau == _scheme.hvc.tau:
        codec = _codec
    else:
        codec = make_codec(tau, profile=_profile)

    pp, _, _ = codec.pp_decode(data)
    return pp, codec, codec.scheme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def _write(path: str, data: bytes) -> None:
    Path(path).write_bytes(data)
    print(f"  wrote {path}  ({len(data):,} bytes)")


def _fmt(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.2f} MB"


def _parse_seed(hex_str: str, name: str) -> bytes:
    """Decode a hex seed and validate it is exactly 32 bytes."""
    try:
        b = bytes.fromhex(hex_str)
    except ValueError:
        sys.exit(f"{name}: not valid hex")
    if len(b) != 32:
        sys.exit(f"{name}: must be 32 bytes (got {len(b)})")
    return b


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def cmd_setup(args):
    kots_seed = _parse_seed(args.kots_seed, "--kots-seed") if args.kots_seed else None
    hvc_seed  = _parse_seed(args.hvc_seed,  "--hvc-seed")  if args.hvc_seed  else None
    if args.tau is not None:
        if not (0 < args.tau <= 32):
            sys.exit(f"--tau={args.tau} out of range [1, 32]")
        codec = make_codec(args.tau, profile=_profile)
    else:
        codec = _codec
    pp, kots_seed, hvc_seed = codec.setup(kots_seed, hvc_seed)
    data = codec.pp_encode(kots_seed, hvc_seed)
    _write(args.out, data)
    print(f"  KOTS seed: {kots_seed.hex()}")
    print(f"  HVC  seed: {hvc_seed.hex()}")
    print(f"  tau: {codec.scheme.hvc.tau}")


# ---------------------------------------------------------------------------
# keygen
# ---------------------------------------------------------------------------

def cmd_keygen(args):
    pp, codec, scheme = _load_pp(args.pp)
    master_seed = _parse_seed(args.seed, "--seed") if args.seed else None
    sk_seed, sk_state, pk = codec.keygen(pp, master_seed)
    _write(args.sk, codec.sk_encode(sk_seed))
    _write(args.stateful_sk, codec.sk_state_encode(sk_state))
    _write(args.pk, codec.pk_encode(pk))
    print(f"  master seed: {sk_seed.hex()}")
    print(f"  slots: 0..{scheme.hvc.n_slots - 1}  (tau={scheme.hvc.tau})")


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------

def cmd_sign(args):
    pp, codec, scheme = _load_pp(args.pp)
    m  = args.msg.encode()
    if args.sk is not None:
        if args.slot is None:
            sys.exit("--slot is required when signing with --sk")
        sk_seed = codec.sk_decode(_read(args.sk))
        t = args.slot
        if not (0 <= t < scheme.hvc.n_slots):
            sys.exit(f"slot {t} out of range [0, {scheme.hvc.n_slots - 1}]")
        sigma = codec.sign_seed(pp, sk_seed, t, m)
    else:
        sk_state = codec.sk_state_decode(_read(args.stateful_sk))
        try:
            sigma, next_state, t = codec.sign_stateful(
                pp, sk_state, m, args.slot
            )
        except ValueError as exc:
            sys.exit(str(exc))
        _write(args.stateful_sk, codec.sk_state_encode(next_state))
    _write(args.out, codec.sig_encode(sigma))
    print(f"  slot={t}  ||Z||_inf={inf_norm(sigma[0])}")


# ---------------------------------------------------------------------------
# verify  (individual iVrfy)
# ---------------------------------------------------------------------------

def cmd_verify(args):
    pp, codec, scheme = _load_pp(args.pp)
    m     = args.msg.encode()
    t     = args.slot
    try:
        pk    = codec.pk_decode(_read(args.pk))
        sigma = codec.sig_decode(_read(args.sig), pp, t)
        ok = scheme.ivrfy(pp, pk, t, m, sigma)
    except Exception:
        ok = False
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def cmd_aggregate(args):
    pp, codec, scheme = _load_pp(args.pp)
    pks  = [codec.pk_decode(_read(p)) for p in args.pks]
    m    = args.msg.encode()
    t    = args.slot
    sigs = [codec.sig_decode(_read(s), pp, t) for s in args.sigs]

    if len(pks) != len(sigs):
        sys.exit("--pks and --sigs must have the same length")

    print(f"  aggregating {len(sigs)} signatures at slot {t} ...")
    sigma_agg = scheme.aggregate(pp, pks, t, m, sigs)
    if sigma_agg is None:
        sys.exit("Aggregation failed (all gamma attempts exhausted or a signature is invalid)")

    _, _, attempt = sigma_agg
    print(f"  success on attempt {attempt}  ||Z_agg||_inf={inf_norm(sigma_agg[0])}")
    _write(args.out, codec.agg_sig_encode(sigma_agg, len(pks)))


# ---------------------------------------------------------------------------
# batch-verify  (aVrfy)
# ---------------------------------------------------------------------------

def cmd_batch_verify(args):
    pp, codec, scheme = _load_pp(args.pp)
    m     = args.msg.encode()
    t     = args.slot
    try:
        pks       = [codec.pk_decode(_read(p)) for p in args.pks]
        sigma_agg = codec.agg_sig_decode(_read(args.sig), pp, t, len(pks))
        ok = scheme.avrfy(pp, pks, t, m, sigma_agg)
    except Exception:
        ok = False
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------

def cmd_sizes(_args):
    kots = _scheme.kots
    hvc = _scheme.hvc

    print(
        f"Lemur serialised sizes  (d={kots.d}, tau={hvc.tau}, "
        f"n_slots={hvc.n_slots}, alpha_w={_scheme.alpha_w}, gamma={_scheme.gamma})"
    )
    print(
        f"  omega={hvc.omega}, kappa={hvc.kappa}, kappa'={hvc.kappa_prime}, "
        f"rho={hvc.rho}, nu={hvc.nu}, eta={hvc.eta}"
    )
    print(f"  m={kots.m}, k={kots.k}, n={kots.n}")
    print(
        f"  beta_z={kots.beta_z:,}  beta_sigma={kots.beta_sigma:,}  "
        f"beta_agg={hvc.beta_agg:,}"
    )
    print()

    from codec import compute_agg_encoding
    for n_signers in [2, 128, 1024, 8192]:
        s = _codec.sizes(n_signers)
        print(f"--- N={n_signers} ---")
        col = max(len(k) for k in s) + 2
        for name, val in s.items():
            indent = "  " if name.startswith("  ") else ""
            label = name.lstrip()
            if isinstance(val, int):
                print(f"  {indent}{label:<{col - len(indent)}} "
                      f"{val:>10,}  ({_fmt(val)})")
            else:
                print(f"  {indent}{label:<{col - len(indent)}} {val}")
        print()


# ---------------------------------------------------------------------------
# vectors — deterministic test vector generation
# ---------------------------------------------------------------------------

# Fixed seeds used by the vectors command.  Signer i uses bytes([i+1]) * 32
# (signer 0 → 0x01 * 32, signer 1 → 0x02 * 32, …).
_VEC_KOTS_SEED = bytes(range(32))        # 00 01 02 … 1f
_VEC_HVC_SEED  = bytes(range(32, 64))    # 20 21 22 … 3f


def _signer_seed(i: int) -> bytes:
    return bytes([i + 1]) * 32


def cmd_vectors(args):
    """Run a fully deterministic sign/aggregate/verify workflow and emit JSON."""
    if args.tau is not None:
        kots = KOTS(profile=_profile)
        hvc  = HVC(kots, tau=args.tau)
        scheme = LEMUR(kots, hvc)
        codec  = LemurCodec(scheme)
    else:
        kots = _scheme.kots
        hvc  = _scheme.hvc
        scheme = _scheme
        codec  = _codec
    n    = args.signers
    t    = args.slot
    msg  = args.msg.encode()

    if not (0 <= t < hvc.n_slots):
        sys.exit(f"slot {t} out of range [0, {hvc.n_slots - 1}]")
    if n < 1:
        sys.exit("--signers must be >= 1")

    print(f"  generating vectors: {n} signer(s), slot={t}, tau={hvc.tau}, msg={msg!r}",
          file=sys.stderr)

    # ---- setup ----
    pp, _, _ = codec.setup(_VEC_KOTS_SEED, _VEC_HVC_SEED)

    # ---- keygen ----
    keys = []
    for i in range(n):
        seed = _signer_seed(i)
        sk_seed, sk_state, pk = codec.keygen(pp, seed)
        keys.append({
            "seed": seed,
            "sk_seed": sk_seed,
            "sk_state": sk_state,
            "pk": pk,
        })
        print(f"  keygen {i} done", file=sys.stderr)

    pks = [k["pk"] for k in keys]

    # ---- sign ----
    sigs_raw = []
    sigs_enc = []
    ivrfy    = []
    for i, k in enumerate(keys):
        sigma = codec.sign_seed(pp, k["sk_seed"], t, msg)
        ok    = scheme.ivrfy(pp, k["pk"], t, msg, sigma)
        sigs_raw.append(sigma)
        sigs_enc.append(codec.sig_encode(sigma))
        ivrfy.append(ok)
        print(f"  sign {i}: ivrfy={'OK' if ok else 'FAIL'}", file=sys.stderr)

    # ---- aggregate ----
    sigma_agg = scheme.aggregate(pp, pks, t, msg, sigs_raw)
    if sigma_agg is None:
        sys.exit("Aggregation failed during vector generation")
    _, _, attempt = sigma_agg
    agg_enc  = codec.agg_sig_encode(sigma_agg, n)
    avrfy    = scheme.avrfy(pp, pks, t, msg, sigma_agg)
    print(f"  aggregate: attempt={attempt}, avrfy={'OK' if avrfy else 'FAIL'}",
          file=sys.stderr)

    # ---- build JSON ----
    vectors = {
        "implementation": "lemur-py reference",
        "parameters": {
            "d":            kots.d,
            "q_kots":       kots.q,
            "q_hvc":        hvc.q,
            "k":            kots.k,
            "m":            kots.m,
            "n":            kots.n,
            "alpha":        kots.alpha,
            "alpha_h":      kots.alpha_h,
            "beta_z":       kots.beta_z,
            "beta_sigma":   kots.beta_sigma,
            "omega":        hvc.omega,
            "eta":          hvc.eta,
            "tau":          hvc.tau,
            "kappa":        hvc.kappa,
            "kappa_prime":  hvc.kappa_prime,
            "alpha_w":      scheme.alpha_w,
            "gamma":        scheme.gamma,
        },
        "seeds": {
            "kots": _VEC_KOTS_SEED.hex(),
            "hvc":  _VEC_HVC_SEED.hex(),
        },
        "pp": codec.pp_encode(_VEC_KOTS_SEED, _VEC_HVC_SEED).hex(),
        "slot":    t,
        "message": msg.hex(),
        "signers": [
            {
                "index": i,
                "seed":  _signer_seed(i).hex(),
                "pk":    codec.pk_encode(k["pk"]).hex(),
            }
            for i, k in enumerate(keys)
        ],
        "signatures": [s.hex() for s in sigs_enc],
        "ivrfy":     ivrfy,
        "aggregate": agg_enc.hex(),
        "agg_attempt": attempt,
        "avrfy":     avrfy,
    }

    body = json.dumps(vectors, indent=2)
    if args.out:
        Path(args.out).write_text(body)
        print(f"  wrote {args.out}  ({len(body):,} chars)", file=sys.stderr)
    else:
        print(body)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Lemur multi-signature CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="Generate public parameters")
    s.add_argument("--out",       default="pp.bin", metavar="FILE")
    s.add_argument("--kots-seed", default=None, metavar="HEX",
                   help="32-byte KOTS seed (hex); random if omitted")
    s.add_argument("--hvc-seed",  default=None, metavar="HEX",
                   help="32-byte HVC seed (hex); random if omitted")
    s.add_argument("--tau", type=int, default=None, metavar="TAU",
                   help="tree depth (default: 20; use 3 for fast testing)")

    s = sub.add_parser("keygen", help="Generate seed key, stateful key, and public key")
    s.add_argument("--pp",   required=True, metavar="FILE")
    s.add_argument("--sk",   required=True, metavar="FILE")
    s.add_argument("--stateful-sk", required=True, metavar="FILE",
                   help="output mutable signer state")
    s.add_argument("--pk",   required=True, metavar="FILE")
    s.add_argument("--seed", default=None, metavar="HEX",
                   help="32-byte master seed (hex); random if omitted")

    s = sub.add_parser("sign", help="Sign with either raw seed key or stateful signer key")
    s.add_argument("--pp",   required=True, metavar="FILE")
    group = s.add_mutually_exclusive_group(required=True)
    group.add_argument("--sk", metavar="FILE",
                       help="raw 32-byte master seed secret key")
    group.add_argument("--stateful-sk", metavar="FILE",
                       help="mutable signer state; updated in place")
    s.add_argument("--slot", type=int, default=None, metavar="T",
                   help="time slot; optional for --stateful-sk")
    s.add_argument("--msg",  required=True, metavar="TEXT")
    s.add_argument("--out",  required=True, metavar="FILE")

    s = sub.add_parser("verify", help="Verify an individual signature")
    s.add_argument("--pp",   required=True, metavar="FILE")
    s.add_argument("--pk",   required=True, metavar="FILE")
    s.add_argument("--slot", required=True, type=int, metavar="T")
    s.add_argument("--msg",  required=True, metavar="TEXT")
    s.add_argument("--sig",  required=True, metavar="FILE")

    s = sub.add_parser("aggregate", help="Aggregate N signatures")
    s.add_argument("--pp",   required=True, metavar="FILE")
    s.add_argument("--slot", required=True, type=int, metavar="T")
    s.add_argument("--msg",  required=True, metavar="TEXT")
    s.add_argument("--pks",  required=True, nargs="+", metavar="FILE")
    s.add_argument("--sigs", required=True, nargs="+", metavar="FILE")
    s.add_argument("--out",  required=True, metavar="FILE")

    s = sub.add_parser("batch-verify", help="Verify an aggregated signature")
    s.add_argument("--pp",   required=True, metavar="FILE")
    s.add_argument("--slot", required=True, type=int, metavar="T")
    s.add_argument("--msg",  required=True, metavar="TEXT")
    s.add_argument("--pks",  required=True, nargs="+", metavar="FILE")
    s.add_argument("--sig",  required=True, metavar="FILE")

    sub.add_parser("sizes", help="Print serialised sizes")

    s = sub.add_parser("vectors", help="Generate deterministic test vectors (JSON)")
    s.add_argument("--signers", type=int, default=2, metavar="N",
                   help="number of signers (default: 2)")
    s.add_argument("--slot", type=int, default=0, metavar="T",
                   help="signing slot (default: 0)")
    s.add_argument("--msg",  default="test vector", metavar="TEXT",
                   help="message string (default: 'test vector')")
    s.add_argument("--out",  default=None, metavar="FILE",
                   help="write JSON to FILE (default: stdout)")
    s.add_argument("--tau",  type=int, default=None, metavar="TAU",
                   help="tree depth (default: 20; use 3 for fast testing)")

    return p


_COMMANDS = {
    "setup":        cmd_setup,
    "keygen":       cmd_keygen,
    "sign":         cmd_sign,
    "verify":       cmd_verify,
    "aggregate":    cmd_aggregate,
    "batch-verify": cmd_batch_verify,
    "sizes":        cmd_sizes,
    "vectors":      cmd_vectors,
}

if __name__ == "__main__":
    args = build_parser().parse_args()
    _COMMANDS[args.command](args)
