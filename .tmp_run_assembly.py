import json

from benchmarks.swebench.build_base_images import (
    assemble_all_agent_images,
    builder_image_tag,
)
from benchmarks.utils.build_utils import default_build_output_dir


def main() -> int:
    build_dir = default_build_output_dir("princeton-nlp/SWE-bench_Lite", "test")
    manifest = build_dir / "base-manifest.jsonl"
    base_images = []
    with manifest.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("error") or not obj.get("tags"):
                continue
            base_images.append(obj["base_image"])

    print(f"successful base images: {len(base_images)}")
    return assemble_all_agent_images(
        base_images=base_images,
        builder_tag=builder_image_tag(),
        build_dir=build_dir,
        target_image="ghcr.io/openhands/eval-agent-server",
        target="source-minimal",
        push=False,
        max_workers=12,
        max_retries=2,
        force_build=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
