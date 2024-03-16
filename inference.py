import argparse
import json
import os

import numpy as np
import torch
import torchaudio as T
from model import UNet
from sampler import SDESampling_batch
from scipy.io.wavfile import write
from sde import VpSdeCos
from utils import (adjust_audio_length, get_event_cond, high_pass_filter,
                   normalize, pooling, resample_audio)

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model

def generate_samples(target_events, class_idx, sampler, cond_scale, device, N, audio_length):
    print(f"Generate {N} samples of class \'{LABELS[class_idx]}\'...")
    noise = torch.randn(N, audio_length, device=device)
    classes = torch.tensor([class_idx]*N, device=device)
    sampler.batch_size = N
    samples = sampler.predict(noise, 100, classes, target_events, cond_scale=cond_scale)
    return samples

def save_samples(samples, output_dir, sr, class_name):
    for j in range(samples.shape[0]):
        sample = samples[j].cpu()
        sample = high_pass_filter(sample)
        write(f"{output_dir}/{class_name}_{str(j+1).zfill(3)}.wav", sr, sample)

def measure_el1_distance(sample, target, event_type):
    sample = normalize(sample)
    target = normalize(target)

    sample_event = get_event_cond(sample, event_type)
    target_event = get_event_cond(target, event_type)
    
    sample_event = pooling(sample_event)
    target_event = pooling(target_event.unsqueeze(0))
    
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(sample_event, target_event)
    return loss.cpu().item()
         

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Set model and sampler
    T.set_audio_backend('sox_io')
    device=torch.device('cuda')
    
    with open(args.param_path) as f:
        params = json.load(f)
    sample_rate = params['sample_rate']
    audio_length = sample_rate * 4
    model = UNet(len(LABELS), params).to(device)
    model = load_ema_weights(model, args.model_path)

    sde = VpSdeCos()
    sampler = SDESampling_batch(model, sde, batch_size=args.N, device=device)
    
    # Prepare target audio if exist
    if args.target_audio is not None:
        target_audio, sr = T.load("TARGET_AUDIO_PATH")
        if sr != sample_rate:
            target_audio = resample_audio(target_audio, sr, sample_rate)
        target_audio = adjust_audio_length(target_audio, audio_length)
        target_event = get_event_cond(target_audio, params['event_type'])
        target_event = target_event.repeat(args.N, 1).to(device)
    else:
        target_event = None
    
    # Generate N samples
    class_idx = LABELS.index(args.class_name)
    generated = generate_samples(target_event, class_idx, sampler, args.cond_scale, device, args.N, audio_length)
    save_samples(generated, args.output_dir, sample_rate, args.class_name)
    print('Done!')
        
    # Measure E-L1 distance if target audio is given
    if args.target_audio is not None:
        dists = []
        for sample in generated:
            dist = measure_el1_distance(sample, target_audio, params['event_type'])
            dists.append(dist)
        print(f"E-L1 distance: {np.mean(dists)}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./pretrained/block-49_epoch-500.pt')
    parser.add_argument('--param_path', type=str, default='./pretrained/params.json')
    parser.add_argument('--target_audio', type=str, help='Path to the target audio file.', default=None)
    # TODO: 여기 인자 이름 안 받고 그냥 인풋처럼 받을 수 없는지?
    parser.add_argument('--class_name', type=str, required=True, help='Class name to generate samples.')
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--cond_scale', type=int, default=3)
    parser.add_argument('--N', type=int, default=3)
    args = parser.parse_args()
    
    main(args)