#!/usr/bin/env python3

import argparse
import torch
import torchaudio
from tqdm import tqdm
import onnxruntime
import torchaudio.compliance.kaldi as kaldi


def main(args):
    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 16
    providers = ["CPUExecutionProvider"]
    #providers = ["CUDAExecutionProvider"]  # 设置提供者为CUDA
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    utt2embedding, spk2embedding = {}, {}
    for utt in tqdm(utt2wav.keys()):
        try:
            audio, sample_rate = torchaudio.load(utt2wav[utt])
        except:
            print("Bad wav path: ", utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        feat = kaldi.fbank(audio,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        if not torch.isnan(torch.tensor(embedding)).any():
            utt2embedding[utt] = embedding
        else:
            utt2embedding[utt] = []
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        if len(embedding) != 0:
            spk2embedding[spk].append(embedding)
        
    #for k, v in spk2embedding.items():
    #    tensor_v = torch.tensor(v)
    #    spk2embedding[k] = torch.tensor(tensor_v).mean(dim=0).tolist()
    
    for k, embeddings_list in spk2embedding.items():
        filtered_embeddings = [emb for emb in embeddings_list if not torch.isnan(torch.tensor(emb)).any()]
        if filtered_embeddings:
            spk2embedding[k] = torch.tensor(filtered_embeddings).mean(dim=0).tolist()
        else:
            #raise ValueError(f"The embeddings for speaker '{k}' are all NaN or the list is empty.")
            spk2embedding[k] = []


    torch.save(utt2embedding, '{}/utt2embedding.pt'.format(args.dir))
    torch.save(spk2embedding, '{}/spk2embedding.pt'.format(args.dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--onnx_path',
                        type=str)
    args = parser.parse_args()
    main(args)
    
    
