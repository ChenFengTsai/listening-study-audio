import argparse
import csv
import json
import random
import shlex
from pathlib import Path

def randomized_pair(prompt, sass_url, other_url, other_model_name, rng):
    if rng.random() < 0.5:
        return {
            "prompt": prompt, "audio_url_a": sass_url, "audio_url_b": other_url,
            "model_a": "SASS", "model_b": other_model_name, "is_attention_check": "FALSE",
        }
    else:
        return {
            "prompt": prompt, "audio_url_a": other_url, "audio_url_b": sass_url,
            "model_a": other_model_name, "model_b": "SASS", "is_attention_check": "FALSE",
        }

def public_url(github_base_url: str, blob_path: str) -> str:
    """Generates the GitHub Pages URL for the audio file."""
    return f"{github_base_url.rstrip('/')}/{blob_path}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-jsonl", default="/storage/ssd1/richtsai1103/MusicBench/MusicBench_test_A.json")
    
    # Model directories
    ap.add_argument("--tf-dir", default="/storage/ssd3/richtsai1103/MusicBench/MusicGen_TF/generation_10s_split/seg2")
    ap.add_argument("--sass-dir", default="/storage/ssd3/richtsai1103/MusicBench/SSM_TTM/wav_sample_10s")
    
    # Attention checks
    ap.add_argument("--attn-static-dir", default="/storage/ssd3/richtsai1103/MusicBench/attention/static")
    ap.add_argument("--attn-clean-dir", default="/storage/ssd3/richtsai1103/MusicBench/attention/clean")
    ap.add_argument("--attn-count", type=int, default=2)
    ap.add_argument("--segment-index", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=None)

    # GitHub specific arguments
    ap.add_argument("--github-url", required=True, help="e.g., https://YOUR_USERNAME.github.io/listening-study-audio")
    ap.add_argument("--repo-dir", required=True, help="Local path to your cloned git repository")
    ap.add_argument("--repo-prefix", default="clips", help="Subfolder inside the repo to store audio")

    # Outputs default locally to avoid permission errors
    ap.add_argument("--out-csv", default="./outputs/hits.csv")
    ap.add_argument("--out-copy-script", default="./outputs/copy_to_repo.sh")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Load prompts
    with open(args.prompts_jsonl, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    prompts_by_stem = {Path(s["location"]).stem: s["main_caption"] for s in samples}

    # Flexibly find files and their stems (handles both _seg2.wav and .wav)
    def get_files_and_stems(d: str):
        if not Path(d).exists(): return {}
        file_map = {}
        for p in Path(d).glob("*.wav"):
            stem = p.name.replace(f"_seg{args.segment_index}.wav", "").replace(".wav", "")
            file_map[stem] = p
        return file_map

    sass_map = get_files_and_stems(args.sass_dir)
    tf_map   = get_files_and_stems(args.tf_dir)
    
    sass_stems = set(sass_map.keys())
    tf_stems   = set(tf_map.keys())
    available  = sorted(set(prompts_by_stem) & sass_stems & tf_stems)

    if args.max_pairs: available = available[: args.max_pairs]

    print(f"SASS: {len(sass_stems)} | TF: {len(tf_stems)} | Intersect-with-prompts: {len(available)}")

    if len(available) == 0:
        print("\n[!] Error: No overlapping files found between SASS, TF-MusicGen, and the JSON prompts.")
        return

    # Collect files to copy
    uploads: dict[Path, str] = {}

    def register(local: Path, model_tag: str) -> str:
        blob_path = f"{args.repo_prefix}/{model_tag}/{local.name}"
        uploads[local] = blob_path
        return public_url(args.github_url, blob_path)

    rows = []
    for stem in available:
        prompt   = prompts_by_stem[stem]
        sass_p   = sass_map[stem]
        tf_p     = tf_map[stem]
        
        sass_url = register(sass_p, "SASS")
        tf_url   = register(tf_p,   "TF_MusicGen")
        rows.append(randomized_pair(prompt, sass_url, tf_url, "TF-MusicGen", rng))

    # Attention checks
    static_clips = sorted(Path(args.attn_static_dir).glob("*.wav")) if Path(args.attn_static_dir).exists() else []
    clean_clips  = sorted(Path(args.attn_clean_dir).glob("*.wav")) if Path(args.attn_clean_dir).exists() else []
    n_attn = min(args.attn_count, len(static_clips), len(clean_clips))

    attn_rows = []
    for i in range(n_attn):
        s_url = register(static_clips[i], "attention_static")
        c_url = register(clean_clips[i],  "attention_clean")
        if rng.random() < 0.5:
            a_url, b_url, a_m, b_m = s_url, c_url, "AttentionStatic", "AttentionClean"
        else:
            a_url, b_url, a_m, b_m = c_url, s_url, "AttentionClean", "AttentionStatic"
        attn_rows.append({
            "prompt": "[Attention Check] Please select the clearer, more musical clip.",
            "audio_url_a": a_url, "audio_url_b": b_url,
            "model_a": a_m, "model_b": b_m, "is_attention_check": "TRUE",
        })

    rng.shuffle(rows)
    final = list(rows)
    if attn_rows:
        step = max(1, len(final) // (len(attn_rows) + 1))
        for i, a in enumerate(attn_rows, start=1):
            final.insert(min(i * step, len(final)), a)

    # Write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "audio_url_a", "audio_url_b", "model_a", "model_b", "is_attention_check"])
        writer.writeheader()
        writer.writerows(final)

    # Write copy script
    Path(args.out_copy_script).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_copy_script, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"REPO_DIR={shlex.quote(args.repo_dir)}\n\n")
        
        for local, blob_path in uploads.items():
            local_q = shlex.quote(str(local))
            dest_q = f"\"$REPO_DIR/{blob_path}\""
            dest_dir = f"\"$REPO_DIR/{blob_path.rsplit('/', 1)[0]}\""
            f.write(f"mkdir -p {dest_dir}\n")
            f.write(f"cp {local_q} {dest_q}\n")

    Path(args.out_copy_script).chmod(0o755)
    print(f"\nSuccess! Generated {len(final)} pairs.")
    print(f"Run this to copy files into your git repo: bash {args.out_copy_script}")

if __name__ == "__main__":
    main()