//! Lemur CLI — matches the Python cli.py interface.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process;

use lemur_rs::codec::{
    agg_sig_decode, agg_sig_encode, pk_decode_with_profile, pk_encode_with_profile, pp_decode,
    pp_encode, sig_decode, sig_encode, sizes, sk_decode, sk_encode, sk_state_decode_with_profile,
    sk_state_encode_with_profile,
};
use lemur_rs::kots::inf_norm;
use lemur_rs::lemur::{
    lemur_aggregate, lemur_avrfy, lemur_ivrfy, lemur_keygen, lemur_setup_with_profile_and_tau,
    lemur_sign_seed, lemur_sign_stateful_mut, LemurSk,
};
use lemur_rs::profile::{Profile, D256_K4};

#[derive(Parser)]
#[command(name = "lemur", about = "Lemur multi-signature CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate shared public parameters
    Setup {
        #[arg(long, default_value = "pp.bin")]
        out: PathBuf,
        #[arg(long, help = "32-byte KOTS seed (hex); random if omitted")]
        kots_seed: Option<String>,
        #[arg(long, help = "32-byte HVC seed (hex); random if omitted")]
        hvc_seed: Option<String>,
        #[arg(long, help = "tree depth (default: 20; use 3 for fast testing)")]
        tau: Option<usize>,
    },
    /// Generate seed key, stateful key, and public key
    Keygen {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        sk: PathBuf,
        #[arg(long = "stateful-sk")]
        stateful_sk: PathBuf,
        #[arg(long)]
        pk: PathBuf,
        #[arg(long, help = "32-byte master seed (hex); random if omitted")]
        seed: Option<String>,
    },
    /// Sign with either raw seed key or stateful signer key
    Sign {
        #[arg(long)]
        pp: PathBuf,
        #[arg(
            long,
            conflicts_with = "stateful_sk",
            required_unless_present = "stateful_sk",
            help = "raw 32-byte master seed secret key"
        )]
        sk: Option<PathBuf>,
        #[arg(
            long = "stateful-sk",
            conflicts_with = "sk",
            required_unless_present = "sk",
            help = "mutable signer state; updated in place"
        )]
        stateful_sk: Option<PathBuf>,
        #[arg(long, help = "time slot; optional for --stateful-sk")]
        slot: Option<usize>,
        #[arg(long)]
        msg: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify an individual signature
    Verify {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        pk: PathBuf,
        #[arg(long)]
        slot: usize,
        #[arg(long)]
        msg: String,
        #[arg(long)]
        sig: PathBuf,
    },
    /// Aggregate N signatures
    Aggregate {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        slot: usize,
        #[arg(long)]
        msg: String,
        #[arg(long, num_args = 1..)]
        pks: Vec<PathBuf>,
        #[arg(long, num_args = 1..)]
        sigs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify an aggregated signature
    #[command(name = "batch-verify")]
    BatchVerify {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        slot: usize,
        #[arg(long)]
        msg: String,
        #[arg(long, num_args = 1..)]
        pks: Vec<PathBuf>,
        #[arg(long)]
        sig: PathBuf,
    },
    /// Print serialized sizes
    Sizes {
        #[arg(
            long,
            default_value = "1024",
            help = "number of signers for aggregate encoding under the shipped profile"
        )]
        n: usize,
    },
    /// Generate deterministic test vectors (JSON)
    Vectors {
        #[arg(long, default_value = "2")]
        signers: usize,
        #[arg(long, default_value = "0")]
        slot: usize,
        #[arg(long, default_value = "test vector")]
        msg: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, help = "tree depth (default: 20; use 3 for fast testing)")]
        tau: Option<usize>,
    },
}

fn read_file(path: &PathBuf) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("error reading {}: {e}", path.display());
        process::exit(1);
    })
}

fn write_file(path: &PathBuf, data: &[u8]) {
    std::fs::write(path, data).unwrap_or_else(|e| {
        eprintln!("error writing {}: {e}", path.display());
        process::exit(1);
    });
    println!("  wrote {}  ({} bytes)", path.display(), data.len());
}

fn parse_seed(hex_str: &str, name: &str) -> [u8; 32] {
    let b = hex::decode(hex_str).unwrap_or_else(|_| {
        eprintln!("{name}: not valid hex");
        process::exit(1);
    });
    if b.len() != 32 {
        eprintln!("{name}: must be 32 bytes (got {})", b.len());
        process::exit(1);
    }
    b.try_into().unwrap()
}

fn random_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    let r = std::fs::read("/dev/urandom").unwrap_or_default();
    for (i, &b) in r.iter().take(32).enumerate() {
        seed[i] = b;
    }
    seed
}

fn fmt_bytes(n: usize) -> String {
    if n < 1024 {
        format!("{n} B")
    } else {
        format!("{:.1} KB", n as f64 / 1024.0)
    }
}

// Fixed seeds for test vectors
const VEC_KOTS_SEED: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];
const VEC_HVC_SEED: [u8; 32] = [
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
];

fn signer_seed(i: usize) -> [u8; 32] {
    [(i + 1) as u8; 32]
}

fn main() {
    let cli = Cli::parse();
    let profile: &'static Profile = &D256_K4;
    match cli.command {
        Commands::Setup {
            out,
            kots_seed,
            hvc_seed,
            tau,
        } => {
            let ks = kots_seed
                .as_deref()
                .map(|s| parse_seed(s, "--kots-seed"))
                .unwrap_or_else(random_seed);
            let hs = hvc_seed
                .as_deref()
                .map(|s| parse_seed(s, "--hvc-seed"))
                .unwrap_or_else(random_seed);
            let tau = tau.unwrap_or(profile.tau);
            if tau == 0 || tau > 32 {
                eprintln!("--tau={tau} out of range [1, 32]");
                process::exit(1);
            }
            let data = pp_encode(&ks, &hs, tau);
            write_file(&out, &data);
            println!("  profile: {}", profile.name);
            println!("  KOTS seed: {}", hex::encode(ks));
            println!("  HVC  seed: {}", hex::encode(hs));
            println!("  tau: {tau}");
        }
        Commands::Keygen {
            pp,
            sk: sk_path,
            stateful_sk: stateful_sk_path,
            pk: pk_path,
            seed,
        } => {
            let pp_data = read_file(&pp);
            let (ks, hs, tau) = pp_decode(&pp_data).unwrap_or_else(|e| {
                eprintln!("pp_decode: {e}");
                process::exit(1);
            });
            let master_seed = seed
                .as_deref()
                .map(|s| parse_seed(s, "--seed"))
                .unwrap_or_else(random_seed);
            let pp_obj = lemur_setup_with_profile_and_tau(&ks, &hs, profile, tau);
            let (sk, sk_state, pk) = lemur_keygen(&pp_obj, &master_seed);
            write_file(&sk_path, &sk_encode(&sk.master_seed));
            let sk_state_bytes =
                sk_state_encode_with_profile(&sk_state, profile).unwrap_or_else(|e| {
                    eprintln!("sk_state_encode: {e}");
                    process::exit(1);
                });
            write_file(&stateful_sk_path, &sk_state_bytes);
            write_file(&pk_path, &pk_encode_with_profile(&pk, profile));
            println!("  master seed: {}", hex::encode(master_seed));
            println!(
                "  slots: 0..{}  (tau={})",
                (1usize << pp_obj.hvc_pp.tau) - 1,
                pp_obj.hvc_pp.tau
            );
        }
        Commands::Sign {
            pp,
            sk: sk_path,
            stateful_sk: stateful_sk_path,
            slot,
            msg,
            out,
        } => {
            let pp_data = read_file(&pp);
            let (ks, hs, tau) = pp_decode(&pp_data).unwrap_or_else(|e| {
                eprintln!("pp_decode: {e}");
                process::exit(1);
            });
            let pp_obj = lemur_setup_with_profile_and_tau(&ks, &hs, profile, tau);
            let (sig, slot_used) = if let Some(sk_path) = sk_path {
                let slot = slot.unwrap_or_else(|| {
                    eprintln!("--slot is required when signing with --sk");
                    process::exit(1);
                });
                let sk_data = read_file(&sk_path);
                let master_seed = sk_decode(&sk_data).unwrap_or_else(|e| {
                    eprintln!("sk_decode: {e}");
                    process::exit(1);
                });
                let sk = LemurSk { master_seed };
                (lemur_sign_seed(&pp_obj, &sk, slot, msg.as_bytes()), slot)
            } else {
                let sk_path = stateful_sk_path.as_ref().unwrap();
                let sk_data = read_file(sk_path);
                let mut sk_state =
                    sk_state_decode_with_profile(&sk_data, profile).unwrap_or_else(|e| {
                        eprintln!("sk_state_decode: {e}");
                        process::exit(1);
                    });
                let (sig, slot_used) =
                    lemur_sign_stateful_mut(&pp_obj, &mut sk_state, msg.as_bytes(), slot)
                        .unwrap_or_else(|e| {
                            eprintln!("{e}");
                            process::exit(1);
                        });
                let next_bytes =
                    sk_state_encode_with_profile(&sk_state, profile).unwrap_or_else(|e| {
                        eprintln!("sk_state_encode: {e}");
                        process::exit(1);
                    });
                write_file(sk_path, &next_bytes);
                (sig, slot_used)
            };
            let norm = inf_norm(&sig.z.0);
            write_file(&out, &sig_encode(&sig, &pp_obj.hvc_pp));
            println!("  slot={slot_used}  ||Z||_inf={norm}");
        }
        Commands::Verify {
            pp,
            pk: pk_path,
            slot,
            msg,
            sig: sig_path,
        } => {
            let pp_data = read_file(&pp);
            let (ks, hs, tau) = pp_decode(&pp_data).unwrap_or_else(|e| {
                eprintln!("pp_decode: {e}");
                process::exit(1);
            });
            let pk = pk_decode_with_profile(&read_file(&pk_path), profile).unwrap_or_else(|e| {
                eprintln!("pk_decode: {e}");
                process::exit(1);
            });
            let pp_obj = lemur_setup_with_profile_and_tau(&ks, &hs, profile, tau);
            let sig = match sig_decode(&read_file(&sig_path), &pp_obj, slot) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("sig_decode: {e}");
                    println!("FAIL");
                    process::exit(1);
                }
            };
            match lemur_ivrfy(&pp_obj, &pk, slot, msg.as_bytes(), &sig) {
                Ok(()) => println!("OK"),
                Err(_) => {
                    println!("FAIL");
                    process::exit(1);
                }
            }
        }
        Commands::Aggregate {
            pp,
            slot,
            msg,
            pks,
            sigs,
            out,
        } => {
            let pp_data = read_file(&pp);
            let (ks, hs, tau) = pp_decode(&pp_data).unwrap_or_else(|e| {
                eprintln!("pp_decode: {e}");
                process::exit(1);
            });
            let pp_obj = lemur_setup_with_profile_and_tau(&ks, &hs, profile, tau);
            let pk_list: Vec<_> = pks
                .iter()
                .map(|p| {
                    pk_decode_with_profile(&read_file(p), profile).unwrap_or_else(|e| {
                        eprintln!("pk_decode: {e}");
                        process::exit(1);
                    })
                })
                .collect();
            let sig_list: Vec<_> = sigs
                .iter()
                .map(|s| {
                    sig_decode(&read_file(s), &pp_obj, slot).unwrap_or_else(|e| {
                        eprintln!("sig_decode: {e}");
                        process::exit(1);
                    })
                })
                .collect();
            println!(
                "  aggregating {} signatures at slot {slot} ...",
                sig_list.len()
            );
            match lemur_aggregate(&pp_obj, &pk_list, slot, msg.as_bytes(), &sig_list) {
                Ok(agg) => {
                    let norm = inf_norm(&agg.z_agg.0);
                    println!("  success on attempt {}  ||Z_agg||_inf={norm}", agg.attempt);
                    let n_signers = pk_list.len();
                    write_file(&out, &agg_sig_encode(&agg, n_signers, &pp_obj));
                }
                Err(e) => {
                    eprintln!("Aggregation failed: {e}");
                    process::exit(1);
                }
            }
        }
        Commands::BatchVerify {
            pp,
            slot,
            msg,
            pks,
            sig: sig_path,
        } => {
            let pp_data = read_file(&pp);
            let (ks, hs, tau) = pp_decode(&pp_data).unwrap_or_else(|e| {
                eprintln!("pp_decode: {e}");
                process::exit(1);
            });
            let pk_list: Vec<_> = pks
                .iter()
                .map(|p| {
                    pk_decode_with_profile(&read_file(p), profile).unwrap_or_else(|e| {
                        eprintln!("pk_decode: {e}");
                        process::exit(1);
                    })
                })
                .collect();
            let pp_obj = lemur_setup_with_profile_and_tau(&ks, &hs, profile, tau);
            let n_signers = pk_list.len();
            let agg = match agg_sig_decode(&read_file(&sig_path), &pp_obj, slot, n_signers) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("agg_sig_decode: {e}");
                    println!("FAIL");
                    process::exit(1);
                }
            };
            match lemur_avrfy(&pp_obj, &pk_list, slot, msg.as_bytes(), &agg) {
                Ok(()) => println!("OK"),
                Err(_) => {
                    println!("FAIL");
                    process::exit(1);
                }
            }
        }
        Commands::Sizes { n } => {
            cmd_sizes(n, profile);
        }
        Commands::Vectors {
            signers,
            slot,
            msg,
            out,
            tau,
        } => {
            run_vectors(signers, slot, &msg, out.as_ref(), tau, profile);
        }
    }
}

fn cmd_sizes(n_signers: usize, profile: &'static Profile) {
    if n_signers != profile.n_signers {
        eprintln!(
            "warning: --n changes only the aggregate encoding estimate under \
             profile={} (sized for N={}); it does not switch to another \
             parameter-estimator row",
            profile.name, profile.n_signers
        );
    }
    let n_slots = 1usize << profile.tau;
    println!(
        "Lemur serialised sizes  (profile={}, d={}, tau={}, \
         n_slots={n_slots}, alpha_w={}, gamma={})",
        profile.name, profile.d, profile.tau, profile.alpha_w, profile.gamma,
    );
    println!(
        "  omega={}, kappa={}, kappa'={}, rho={}, nu={}, eta={}",
        profile.omega, profile.kappa, profile.kappa_prime, profile.k, profile.n, profile.eta,
    );
    println!(
        "  m={}, k={}, n={}",
        profile.m, profile.k, profile.n
    );
    println!(
        "  beta_z={}  beta_sigma={}",
        profile.beta_z, profile.beta_sigma
    );
    println!();

    let pp = lemur_setup_with_profile_and_tau(&[0u8; 32], &[0u8; 32], profile, profile.tau);
    for (name, nbytes) in sizes(&pp, n_signers) {
        let indent = if name.starts_with("  ") { "  " } else { "" };
        let label = name.trim_start();
        println!(
            "  {indent}{label:<40} {nbytes:>10}  ({})",
            fmt_bytes(nbytes)
        );
    }
}

fn run_vectors(
    n: usize,
    t: usize,
    msg: &str,
    out: Option<&PathBuf>,
    tau: Option<usize>,
    profile: &'static Profile,
) {
    let tau = tau.unwrap_or(profile.tau);
    let n_slots = 1usize << tau;

    if t >= n_slots {
        eprintln!("slot {t} out of range [0, {}]", n_slots - 1);
        process::exit(1);
    }
    if n < 1 {
        eprintln!("--signers must be >= 1");
        process::exit(1);
    }

    eprintln!(
        "  generating vectors: {n} signer(s), slot={t}, \
         tau={tau}, msg={msg:?}"
    );

    let msg_bytes = msg.as_bytes();

    // Setup
    let pp = lemur_setup_with_profile_and_tau(&VEC_KOTS_SEED, &VEC_HVC_SEED, profile, tau);

    // KeyGen
    let mut keys: Vec<(
        lemur_rs::lemur::LemurSk,
        lemur_rs::lemur::LemurStateSk,
        lemur_rs::lemur::LemurPk,
    )> = Vec::new();
    for i in 0..n {
        let seed = signer_seed(i);
        let (sk, sk_state, pk) = lemur_keygen(&pp, &seed);
        keys.push((sk, sk_state, pk));
        eprintln!("  keygen {i} done");
    }
    let pks: Vec<_> = keys.iter().map(|(_, _, pk)| pk.clone()).collect();

    // Sign
    let mut sigs_raw = Vec::new();
    let mut sigs_enc = Vec::new();
    let mut ivrfy = Vec::new();
    for (i, (sk, _, pk)) in keys.iter().enumerate() {
        let sig = lemur_sign_seed(&pp, sk, t, msg_bytes);
        let ok = lemur_ivrfy(&pp, pk, t, msg_bytes, &sig).is_ok();
        let encoded = sig_encode(&sig, &pp.hvc_pp);
        sigs_raw.push(sig);
        sigs_enc.push(encoded);
        ivrfy.push(ok);
        eprintln!("  sign {i}: ivrfy={}", if ok { "OK" } else { "FAIL" });
    }

    // Aggregate
    let sigma_agg = lemur_aggregate(&pp, &pks, t, msg_bytes, &sigs_raw).unwrap_or_else(|e| {
        eprintln!("Aggregation failed during vector generation: {e}");
        process::exit(1);
    });
    let attempt = sigma_agg.attempt;
    let agg_enc = agg_sig_encode(&sigma_agg, n, &pp);
    let avrfy = lemur_avrfy(&pp, &pks, t, msg_bytes, &sigma_agg).is_ok();
    eprintln!(
        "  aggregate: attempt={attempt}, avrfy={}",
        if avrfy { "OK" } else { "FAIL" }
    );

    // Build JSON
    let signers_json: Vec<_> = (0..n)
        .map(|i| {
            serde_json::json!({
                "index": i,
                "seed": hex::encode(signer_seed(i)),
                "pk": hex::encode(pk_encode_with_profile(&pks[i], profile)),
            })
        })
        .collect();

    let vectors = serde_json::json!({
        "implementation": "lemur-rs",
        "parameters": {
            "d": profile.d,
            "q_kots": profile.q_kots(),
            "q_hvc": profile.q_hvc(),
            "k": profile.k,
            "m": profile.m,
            "n": profile.n,
            "alpha": profile.alpha,
            "alpha_h": profile.alpha_h,
            "beta_z": profile.beta_z,
            "beta_sigma": profile.beta_sigma,
            "omega": profile.omega,
            "eta": profile.eta,
            "tau": tau,
            "kappa": profile.kappa,
            "kappa_prime": profile.kappa_prime,
            "alpha_w": profile.alpha_w,
            "gamma": profile.gamma,
        },
        "seeds": {
            "kots": hex::encode(VEC_KOTS_SEED),
            "hvc": hex::encode(VEC_HVC_SEED),
        },
        "pp": hex::encode(pp_encode(&VEC_KOTS_SEED, &VEC_HVC_SEED, tau)),
        "slot": t,
        "message": hex::encode(msg_bytes),
        "signers": signers_json,
        "signatures": sigs_enc.iter().map(hex::encode).collect::<Vec<_>>(),
        "ivrfy": ivrfy,
        "aggregate": hex::encode(&agg_enc),
        "agg_attempt": attempt,
        "avrfy": avrfy,
    });

    let body = serde_json::to_string_pretty(&vectors).expect("json serialization");
    if let Some(path) = out {
        std::fs::write(path, &body).unwrap_or_else(|e| {
            eprintln!("write error: {e}");
            process::exit(1);
        });
        eprintln!("  wrote {}  ({} chars)", path.display(), body.len());
    } else {
        println!("{body}");
    }
}
