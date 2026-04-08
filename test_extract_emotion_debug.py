import json
import os
import traceback


def print_header(title):
    print(f"\n=== {title} ===")


def main():
    print_header("Env")
    print("CWD =", os.getcwd())
    print("PYTHONPATH =", os.environ.get("PYTHONPATH"))
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    print_header("Torch")
    try:
        import torch

        print("torch.__version__ =", torch.__version__)
        print("torch.cuda.is_available() =", torch.cuda.is_available())
        print("torch.cuda.device_count() =", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Torch check failed:")
        print(type(e).__name__, e)
        traceback.print_exc()

    print_header("FunASR Import")
    try:
        import funasr
        from funasr import AutoModel

        print("funasr.__version__ =", getattr(funasr, "__version__", "unknown"))
    except Exception as e:
        print("FunASR import failed:")
        print(type(e).__name__, e)
        traceback.print_exc()
        return

    audio_path = "datasets/HateMM/audios/hate_video_143.wav"
    print_header("Input Audio")
    print("audio_path =", audio_path)
    print("exists =", os.path.exists(audio_path))
    if not os.path.exists(audio_path):
        return

    for device in ["cuda:0", "cpu"]:
        print_header(f"AutoModel Build [{device}]")
        try:
            model = AutoModel(
                model="FunAudioLLM/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                hub="hf",
                disable_update=True,
            )
            print("build_status = success")
        except Exception as e:
            print("build_status = failed")
            print(type(e).__name__, e)
            traceback.print_exc()
            continue

        print_header(f"Generate [{device}]")
        try:
            res = model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            print("generate_status = success")
            print("raw_result =")
            print(json.dumps(res, ensure_ascii=False, indent=2))
            if res and isinstance(res, list) and "text" in res[0]:
                print("raw_text_repr =", repr(res[0]["text"]))
        except Exception as e:
            print("generate_status = failed")
            print(type(e).__name__, e)
            traceback.print_exc()


if __name__ == "__main__":
    main()
