import argparse
import os
from datasets import load_from_disk
from transformers import Wav2Vec2Processor
from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import torchaudio.sox_effects as sox_effects
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/data/huggingface/metrics"
os.environ["HF_HOME"] = "/data/huggingface"
# os.environ["TMPDIR"] = "/data2_from_58175/tmp"
# python separate_aug.py --datasets_path=hf_datasets/train --wav_root_dir="/data/MGTV/train_" --processor_path=processor
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make dataset for w2v2_finetuning")
    parser.add_argument("-p_datasets", "--datasets_path", type=str, default="", help="The path to the dataset file,e.g. hf_datasets/train")
    parser.add_argument("-rd_wav", "--wav_root_dir", type=str, default="", help="The dir of the wav files,e.g. /data/MGTV/train")
    parser.add_argument("-p_processor", "--processor_path", type=str, default="", help="The path to processor")

    args = parser.parse_args()
    print(args)
    datasets_path = args.datasets_path
    wav_root_dir = args.wav_root_dir
    processor_path = args.processor_path

    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    cuda_device = "cuda:0"
    denoiser = pretrained.get_model('mdx_extra_q')
    denoiser.to(cuda_device)

    datasets = load_from_disk(datasets_path)
    def separate(batch):
        file = batch["file"].replace("/data/MGTV/train", wav_root_dir).replace("/data/MGTV/Data_B", wav_root_dir)
        device = "cuda:0"
        # 需先将音频进行上采样
        x = sox_effects.apply_effects_file(file, effects=[['rate', str(44100)]])[0]
        # 复制成双声道音频
        x = x.unsqueeze(1)
        x = x.expand(x.shape[0], 2, x.shape[2]).to(device)
        # 送入人声分离的模型
        out = apply_model(denoiser, x)[0]
        vocal, out_rate = sox_effects.apply_effects_tensor(out[3].mean(0, keepdim=True).cpu(), sample_rate=44100,
                                                           effects=[['rate', str(16000)]])
        # 保存在原音频路径下，后缀"_vocal"
        torchaudio.save(file.replace(".wav", "_vocal.wav"), vocal.cpu(), out_rate, bits_per_sample=16)

    datasets.map(separate)
    print("~"*20, "Done", "~"*20)
