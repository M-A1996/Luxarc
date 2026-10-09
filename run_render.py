"""
Run the Architectural Render Agent from the command line.

  python run_render.py <image_path> "make it dusk, Scandinavian, add trees"

Optional crop (keep only the building, drop surroundings):
  python run_render.py <image_path> "..." --crop 100 80 900 720   # left top right bottom

Output lands in ./output ; intermediate artifacts (redacted image + structure map,
i.e. exactly what leaves in cloud mode) land in ./work so you can inspect them.
"""
import sys

from render_agent.graph import run_agent


def main():
    args = [a for a in sys.argv[1:]]
    if len(args) < 2:
        print(__doc__)
        return
    crop = None
    if "--crop" in args:
        i = args.index("--crop")
        try:
            crop = tuple(int(x) for x in args[i + 1:i + 5])
            args = args[:i]
        except Exception:
            print("Bad --crop; expected 4 integers: left top right bottom")
            return
    image_path, request = args[0], " ".join(args[1:])
    print(f"\nImage:   {image_path}\nRequest: {request}\n" + "=" * 60)
    result = run_agent(image_path, request, crop_box=crop)
    print("=" * 60)
    print(f"Final render: {result['output_path']}")
    print(f"Critic score: {result['score']}/10  ·  renders tried: {result['attempts']}")


if __name__ == "__main__":
    main()
