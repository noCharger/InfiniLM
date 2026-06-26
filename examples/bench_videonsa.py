import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from infinilm.llm.llm import LLM
from infinilm.llm.sampling_params import SamplingParams
from infinilm.processors import AutoInfinilmProcessor


def parse_int_list(values):
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    return [int(value) for value in values]


def probe_video_metadata(video_path):
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            try:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                cap.release()
            duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
            return {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
                "source": "cv2",
            }

    try:
        from torchvision.io import read_video_timestamps

        pts, fps = read_video_timestamps(video_path, pts_unit="sec")
    except Exception:
        return {}

    frame_count = len(pts)
    duration = float(pts[-1]) if frame_count > 0 else 0.0
    return {
        "frame_count": frame_count,
        "fps": float(fps or 0.0),
        "width": 0,
        "height": 0,
        "duration": duration,
        "source": "torchvision_timestamps",
    }


def apply_video_auto_args(args):
    if not args.video:
        return {}
    meta = probe_video_metadata(args.video)
    frame_count = meta.get("frame_count", 0)
    duration = meta.get("duration", 0.0)
    width = meta.get("width", 0)
    height = meta.get("height", 0)

    if args.video_num_frames is None:
        if duration > 0:
            inferred = int(round(duration * args.video_auto_sample_fps))
        elif frame_count > 0:
            inferred = frame_count
        else:
            inferred = args.video_auto_min_frames
        if frame_count > 0:
            inferred = min(inferred, frame_count)
        args.video_num_frames = max(
            args.video_auto_min_frames,
            min(args.video_auto_max_frames, inferred),
        )

    if args.video_max_pixels is None:
        source_pixels = (
            width * height
            if width > 0 and height > 0
            else args.video_auto_max_pixels_cap
        )
        args.video_max_pixels = min(source_pixels, args.video_auto_max_pixels_cap)

    return meta


def decode_video_frames(video_path, num_frames):
    if not video_path:
        return None
    try:
        from decord import VideoReader, cpu
        from PIL import Image
    except Exception:
        return video_path

    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    if total == 0:
        return video_path
    num_frames = max(1, min(num_frames or total, total))
    if num_frames == 1:
        indices = [0]
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
    batch = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in batch]


def make_prompt(tokenizer, target_len, prompt_seed):
    seed = prompt_seed.rstrip() + " "
    seed_ids = tokenizer.encode(seed)
    if not seed_ids:
        raise RuntimeError("Tokenizer returned no tokens for the benchmark seed prompt")
    repeat = (target_len + len(seed_ids) - 1) // len(seed_ids)
    return tokenizer.decode((seed_ids * repeat)[:target_len], skip_special_tokens=True)


def make_messages(prompt, image_path, video_path, video_payload, batch_size):
    content = [{"type": "text", "text": prompt}]
    if video_path:
        content = [
            {"type": "video_url", "video_url": {"url": video_payload or video_path}}
        ] + content
    elif image_path:
        content = [{"type": "image_url", "image_url": {"url": image_path}}] + content
    return [[{"role": "user", "content": content}] for _ in range(batch_size)]


def run_case(model, tokenizer, args, batch_size, input_len, output_len):
    prompt = make_prompt(tokenizer, input_len, args.prompt)
    messages = make_messages(
        prompt, args.image, args.video, args.video_payload, batch_size
    )
    sampling = SamplingParams(
        max_tokens=output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    if args.warmup:
        model.chat(messages=messages, sampling_params=sampling, use_tqdm=False)

    start = time.perf_counter()
    outputs = model.chat(messages=messages, sampling_params=sampling, use_tqdm=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    total_new_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(
        "case "
        f"batch_size={batch_size} input_len={input_len} output_len={output_len} "
        f"actual_output_tokens={total_new_tokens} "
        f"elapsed_ms={elapsed_ms:.2f} "
        f"output_tok_per_s={total_new_tokens / (elapsed_ms / 1000):.2f}"
    )
    if outputs and not args.no_print_output:
        print("=== sample prompt ===")
        print(outputs[0].prompt)
        print("=== sample output ===")
        print(outputs[0].outputs[0].text)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VideoNSA multimodal inference with InfiniLM"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="/data-aisoft/pepe/images/bus.jpg")
    parser.add_argument("--image-max-pixels", type=int, default=50176)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-num-frames", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-auto-min-frames", type=int, default=4)
    parser.add_argument("--video-auto-max-frames", type=int, default=8)
    parser.add_argument("--video-auto-sample-fps", type=float, default=1.0)
    parser.add_argument("--video-auto-max-pixels-cap", type=int, default=50176)
    parser.add_argument(
        "--prompt",
        default="describe the image",
        help="image/text prompt seed repeated to target input length",
    )
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        nargs="+",
        default=["4"],
        help="space or comma separated batch sizes",
    )
    parser.add_argument(
        "--input-len",
        nargs="+",
        default=["256", "2048", "8192"],
        help="space or comma separated text token lengths",
    )
    parser.add_argument(
        "--output-len",
        nargs="+",
        default=["128"],
        help="space or comma separated generation lengths",
    )
    parser.add_argument("--enable-paged-attn", action="store_true")
    parser.add_argument("--attn", default="default")
    parser.add_argument("--enable-graph", action="store_true")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--weight-load", default="sync", choices=["sync", "async"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-print-output",
        action="store_true",
        help="only print timing, not sample prompt/output",
    )
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_size)
    input_lens = parse_int_list(args.input_len)
    output_lens = parse_int_list(args.output_len)
    max_batch_size = max(batch_sizes)
    max_cache_len = max(input_lens) + max(output_lens) + 4096
    cache_type = "paged" if args.enable_paged_attn else "static"
    attn = (
        "paged-attn" if args.enable_paged_attn and args.attn == "default" else args.attn
    )

    video_meta = apply_video_auto_args(args)
    args.video_payload = decode_video_frames(args.video, args.video_num_frames)
    if args.image and not args.video and args.image_max_pixels is not None:
        os.environ["INFINILM_VIDEONSA_IMAGE_MAX_PIXELS"] = str(args.image_max_pixels)
    if args.image and not args.video and args.image_min_pixels is not None:
        os.environ["INFINILM_VIDEONSA_IMAGE_MIN_PIXELS"] = str(args.image_min_pixels)
    if args.video_num_frames is not None:
        os.environ["INFINILM_VIDEONSA_VIDEO_NUM_FRAMES"] = str(args.video_num_frames)
    if args.video_max_pixels is not None:
        os.environ["INFINILM_VIDEONSA_VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
    if args.video_min_pixels is not None:
        os.environ["INFINILM_VIDEONSA_VIDEO_MIN_PIXELS"] = str(args.video_min_pixels)

    print(
        f"bench_config model={args.model} image={args.image} video={args.video} prompt={args.prompt!r} "
        f"device={args.device} paged={args.enable_paged_attn} "
        f"videonsa_nsa=always_on "
        f"image_max_pixels={args.image_max_pixels} "
        f"video_num_frames={args.video_num_frames} video_max_pixels={args.video_max_pixels} "
        f"video_predecoded={isinstance(args.video_payload, list)} "
        f"video_meta={video_meta}"
    )
    processor = AutoInfinilmProcessor.from_pretrained(args.model)
    tokenizer = processor.get_tokenizer()
    device = "cuda" if args.device == "nvidia" else args.device
    model = LLM(
        model_path=args.model,
        device=device,
        tensor_parallel_size=args.tp,
        cache_type=cache_type,
        max_batch_size=max_batch_size,
        max_tokens=max(output_lens),
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_cache_len=max_cache_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        attn_backend=attn,
        enable_graph=args.enable_graph,
        weight_load_mode=args.weight_load,
    )

    for batch_size in batch_sizes:
        for input_len in input_lens:
            for output_len in output_lens:
                run_case(model, tokenizer, args, batch_size, input_len, output_len)


if __name__ == "__main__":
    main()
