"""Export every atlas entry or replay identified candidates; never rerun search."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_illuminate import export_gallery, replay_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--candidate_ids', nargs='+', type=int, help='reconstruct these saved candidate IDs and compare images/embeddings')
    parser.add_argument('--clip_model_path', type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.candidate_ids:
        print(replay_archive(args.run_dir, args.candidate_ids, args.clip_model_path, args.offline))
    else:
        print(export_gallery(args.run_dir)['counts'])


if __name__ == '__main__':
    main()
