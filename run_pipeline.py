import argparse
from pathlib import Path

from pipeline.parse_script import parse_script_file
from pipeline.models import EnrichedLine, save_lines_json
from pipeline.emotions import default_pause_after
from pipeline.generate_qwen import generate_all
from pipeline.stitch import stitch_episode


def main():
    parser = argparse.ArgumentParser(description="Run Qwen voice pipeline")
    parser.add_argument("script_file", help="Path to script text file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-mp3", default="output/final.mp3")
    args = parser.parse_args()

    print(f"Parsing script: {args.script_file}...")
    turns = parse_script_file(args.script_file)
    if not turns:
        print("No dialogue lines found. Use 'Speaker 1: ...' or '[M] ...'")
        return

    enriched = []
    for turn in turns:
        emotion = "neutral"
        if "?" in turn.text:
            emotion = "curious"
        elif "!" in turn.text:
            emotion = "excited"
        elif "haha" in turn.text.lower() or "lol" in turn.text.lower():
            emotion = "playful"

        enriched.append(
            EnrichedLine(
                index=turn.index,
                speaker=turn.speaker,
                text=turn.text,
                emotion=emotion,
                pause_before_ms=0,
                pause_after_ms=default_pause_after(turn.text, emotion),
            )
        )

    lines_path = "output/lines.json"
    save_lines_json(lines_path, enriched)
    print(f"Saved {len(enriched)} lines to {lines_path}")

    clips_dir = "output/clips"
    print("Generating audio clips...")
    generate_all(
        enriched,
        config_path="config/voices.yaml",
        clips_dir=clips_dir,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"Stitching -> {args.output_mp3}...")
        stitch_episode(lines_path, clips_dir, args.output_mp3)
        print("Done!")
    else:
        print("Dry run complete.")


if __name__ == "__main__":
    main()