import argparse
import os
from datasets import load_from_disk
from transformers import Wav2Vec2Processor
import torchaudio.sox_effects as sox_effects
import torch
from CzcWav2vec2 import (
    Wav2Vec2ForSequenceClassification_czc,
)
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/data/huggingface/metrics"
os.environ["HF_HOME"] = "/data/huggingface"
# os.environ["TMPDIR"] = "/data2_from_58175/tmp"
# python separate_aug.py --datasets_path=hf_datasets/train --wav_root_dir="/data/MGTV/train_" --processor_path=processor
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make dataset for w2v2_finetuning")
    parser.add_argument("-p_datasets", "--datasets_path", type=str, default="", help="The path to the dataset file,e.g. hf_datasets/train")
    parser.add_argument("-p_model", "--model_path", type=str, default="", help="The dir of the wav files,e.g. /data/MGTV/train")
    parser.add_argument("-p_processor", "--processor_path", type=str, default="", help="The path to processor")
    parser.add_argument("-fn_output", "--output_file_name", type=str, default="submission.csv", help="The name of output file")
    parser.add_argument("-rd_wav", "--wav_root_dir", type=str, default="", help="The dir of the wav files,e.g. /data/MGTV/train")
    args = parser.parse_args()
    print(args)
    cuda_device = "cuda:0"
    datasets_path = args.datasets_path
    wav_root_dir = args.wav_root_dir
    processor_path = args.processor_path
    model_path = args.model_path
    output_file_name =args.output_file_name
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    model = Wav2Vec2ForSequenceClassification_czc.from_pretrained(model_path)


    model.to(cuda_device)
    class_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    dataset = load_from_disk(datasets_path)

    def map_to_pred(batch):
        # 原音频路径
        file = batch["file"][0].replace("/data/MGTV/train", wav_root_dir).replace("/data/MGTV/Data_B", wav_root_dir)
        # 分离后的人声音频路径
        file_vocal = batch["file"][0].replace("/data/MGTV/train", wav_root_dir).replace("/data/MGTV/Data_B", wav_root_dir).replace(".wav", "_vocal.wav")
        speech = sox_effects.apply_effects_file(path=file, effects=[['rate', str(16000)]])[0][0]
        speech_vocal = sox_effects.apply_effects_file(path=file_vocal, effects=[['rate', str(16000)]])[0][0]
        # 归一化
        input_values = processor(speech, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(
            model.device)
        input_values_vocal = processor(speech_vocal, return_tensors="pt", padding="longest",
                                       sampling_rate=16000).input_values.to(model.device)
        with torch.no_grad():
            logits = model(input_values).logits
        logits_vocal = model(input_values_vocal).logits
        # 人声音频输出的后验概率的权重设为0.5
        predicted_ids = torch.argmax(logits + 0.5 * logits_vocal, dim=-1).tolist()
        batch["class"] = [class_list[predicted_ids[0]]]
        return batch
    dataset = dataset.map(map_to_pred, batched=True, batch_size=1)
    dataset = dataset.remove_columns(["file", "label", "length"])
    dataset = dataset.rename_column("id", "file")
    dataset.to_csv(output_file_name, index=None)
    print("~"*20, "Done", "~"*20)
