import json
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from dataset import from_path as dataset_from_path
from model import UNet
from sampler import SDESampling
from sde import SubVpSdeCos
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import get_event_cond, high_pass_filter, normalize, plot_env

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']

def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)

# --- Learner ---
class Learner:
    def __init__(
        self, model_dir, model, train_set, test_set, params, distributed
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.ema_weights = [param.clone().detach()
                            for param in self.model.parameters()]
        self.lr = params['lr']
        self.epoch = 0
        self.step = 0
        self.is_master = True
        self.distributed = distributed
        self.restore_from_checkpoint(params['checkpoint_id'])
        
        self.sde = SubVpSdeCos()
        self.ema_rate = params['ema_rate']
        self.train_set = train_set
        self.test_set = test_set
        self.params = params
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=params['scheduler_factor'], 
            patience=params['scheduler_patience_epoch']*len(self.train_set)//params['num_steps_to_test'], 
            threshold=params['scheduler_threshold']
        )
        self.params['total_params_num'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.loss_fn = nn.MSELoss()
        self.v_loss = nn.MSELoss(reduction="none")
        self.summary_writer = None
        self.n_bins = params['n_bins']
        self.num_elems_in_bins_train = np.zeros(self.n_bins)
        self.sum_loss_in_bins_train = np.zeros(self.n_bins)
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)
        self.cum_grad_norms = 0
        
        
    # Train
    def train(self):
        device = next(self.model.parameters()).device
        while True:
            if self.distributed: self.train_set.sampler.set_epoch(self.epoch)
            for features in (
                tqdm(self.train_set,
                     desc=f"Epoch {self.epoch}")
                if self.is_master
                else self.train_set
            ):
                self.model.train()
                features = _nested_map(
                    features,
                    lambda x: x.to(device) if isinstance(
                        x, torch.Tensor) else x,
                )
                loss = self.train_step(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f"Detected NaN loss at step {self.step}.")
                    
                # Logging by steps
                if self.is_master:
                    if self.step % 250 == 249:
                        self._write_summary(self.step)
                        
                    if self.step % self.params['num_steps_to_test'] == 0:
                        self.test_set_evaluation()
                        self.scheduler.step(sum(self.sum_loss_in_bins_test)/sum(self.num_elems_in_bins_test))
                        self._write_test_summary(self.step)
                self.step += 1
            
            # Logging by epochs
            if self.is_master:
                if self.epoch % self.params['num_epochs_to_save'] == 0:
                    self._write_inference_summary(self.step, device)
                    self.save_to_checkpoint(filename=f'epoch-{self.epoch}')
                self.epoch += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features["audio"]
        classes = features["class"]
        events = features["event"]

        N, T = audio.shape

        t = torch.rand(N, 1, device=audio.device)
        t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
        noise = torch.randn_like(audio)
        noisy_audio = self.sde.perturb(audio, t, noise)
        sigma = self.sde.sigma(t)
        predicted = self.model(noisy_audio, sigma, classes, events)
        loss = self.loss_fn(noise, predicted)

        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.is_master:
            self.update_ema_weights()

        t_detach = t.clone().detach().cpu().numpy()
        t_detach = np.reshape(t_detach, -1)

        vectorial_loss = self.v_loss(noise, predicted).detach()
        vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
        vectorial_loss = np.reshape(vectorial_loss, -1)

        self.update_conditioned_loss(vectorial_loss, t_detach, True)
        self.cum_grad_norms += self.grad_norm
        return loss
    
    
    # Test
    def test_set_evaluation(self):
        with torch.no_grad():
            self.model.eval()
            for features in self.test_set:
                audio = features["audio"].cuda()
                classes = features["class"].cuda()
                events = features["event"].cuda()

                N, T = audio.shape

                t = torch.rand(N, 1, device=audio.device)
                t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
                noise = torch.randn_like(audio)
                noisy_audio = self.sde.perturb(audio, t, noise)
                sigma = self.sde.sigma(t)
                predicted = self.model(noisy_audio, sigma, classes, events)

                vectorial_loss = self.v_loss(noise, predicted).detach()

                vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
                vectorial_loss = np.reshape(vectorial_loss, -1)
                t = t.cpu().numpy()
                t = np.reshape(t, -1)
                self.update_conditioned_loss(
                    vectorial_loss, t, False)
                
    
    # Update loss & ema weights
    def update_conditioned_loss(self, vectorial_loss, continuous_array, isTrain):
        continuous_array = np.trunc(self.n_bins * continuous_array)
        continuous_array = continuous_array.astype(int)
        if isTrain:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_train[continuous_array[k]] += 1
                self.sum_loss_in_bins_train[continuous_array[k]
                                            ] += vectorial_loss[k]
        else:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_test[continuous_array[k]] += 1
                self.sum_loss_in_bins_test[continuous_array[k]
                                           ] += vectorial_loss[k]

    def update_ema_weights(self):
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param -= (1 - self.ema_rate) * (ema_param - param.detach())
    
    # Logging stuff  
    def _write_summary(self, step):
        loss_in_bins_train = np.divide(
            self.sum_loss_in_bins_train, self.num_elems_in_bins_train
        )
        dic_loss_train = {}
        for k in range(self.n_bins):
            dic_loss_train["loss_bin_" + str(k)] = loss_in_bins_train[k]

        sum_loss_n_steps = np.sum(self.sum_loss_in_bins_train)
        mean_grad_norms = self.cum_grad_norms / self.num_elems_in_bins_train.sum() * \
            self.params['batch_size']
        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)

        writer.add_scalar('train/sum_loss_on_n_steps',
                          sum_loss_n_steps, step)
        writer.add_scalar("train/mean_grad_norm", mean_grad_norms, step)
        writer.add_scalars("train/conditioned_loss", dic_loss_train, step)
        writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_train = np.zeros(self.n_bins)
        self.sum_loss_in_bins_train = np.zeros(self.n_bins)
        self.cum_grad_norms = 0

    def _write_test_summary(self, step):
        loss_in_bins_test = np.divide(
            self.sum_loss_in_bins_test, self.num_elems_in_bins_test
        )
        dic_loss_test = {}
        for k in range(self.n_bins):
            dic_loss_test["loss_bin_" + str(k)] = loss_in_bins_test[k]

        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)
        writer.add_scalars("test/conditioned_loss", dic_loss_test, step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)
        
    def _write_inference_summary(self, step, device, cond_scale=3.):
        sde = SubVpSdeCos()
        sampler = SDESampling(self.model, sde)
        
        test_feature = self.get_random_test_feature()
        test_event = test_feature["event"].unsqueeze(0).to(device)
        
        event_loss = []
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio(f"test_sample/audio", test_feature["audio"], step, sample_rate=22050)
        writer.add_image(f"test_sample/envelope", plot_env(test_feature["audio"]), step, dataformats='HWC')
        
        for class_idx in range(len(LABELS)):
            noise = torch.randn(1, self.params['audio_length'], device=device)
            classes = torch.tensor([class_idx], device=device)
            
            sample = sampler.predict(noise, 100, classes, test_event, cond_scale=cond_scale)
            sample = sample.flatten().cpu()
            
            sample = normalize(sample)
            sample = high_pass_filter(sample, sr=22050)
            
            event_loss.append(self.loss_fn(test_event.squeeze(0).cpu(), get_event_cond(sample, self.params['event_type'])))
            writer.add_audio(f"{LABELS[class_idx]}/audio", sample, step, sample_rate=22050)
            writer.add_image(f"{LABELS[class_idx]}/envelope", plot_env(sample), step, dataformats='HWC')
        
        event_loss = sum(event_loss) / len(event_loss)
        writer.add_scalar(f"test/event_loss", event_loss, step)
        writer.flush()
    
    # Utils    
    def get_random_test_feature(self):
        return self.test_set.dataset[random.choice(range(len(self.test_set.dataset)))]
        
    def log_params(self):
        with open(os.path.join(self.model_dir, 'params.json'), 'w') as fp:
            json.dump(self.params, fp, indent=4)
        fp.close()

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "epoch": self.epoch,
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "ema_weights": self.ema_weights,
            "lr": self.lr,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.ema_weights = state_dict["ema_weights"]
        self.lr = state_dict["lr"]

    def restore_from_checkpoint(self, checkpoint_id=None):
        try:
            if checkpoint_id is None:
                # find latest checkpoint_id
                list_weights = glob(f'{self.model_dir}/epoch-*.pt')
                list_ids = [int(os.path.basename(weight_path).split('-')[-1].rstrip('.pt')) for weight_path in list_weights]
                checkpoint_id = list_ids.index(max(list_ids))

            checkpoint = torch.load(list_weights[checkpoint_id])
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, ValueError):
            return False
        
    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}_step-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)


# --- Training functions ---
def _train_impl(replica_id, model, train_set, test_set, params, distributed=False):
    torch.backends.cudnn.benchmark = True
    learner = Learner(
        params['model_dir'], model, train_set, test_set, params, distributed=distributed
    )
    learner.is_master = replica_id == 0
    learner.log_params()
    learner.train()


def train(params):
    model = UNet(num_classes=len(LABELS), params=params).cuda()
    train_set = dataset_from_path(params['train_dirs'], params, LABELS)
    test_set = dataset_from_path(params['test_dirs'], params, LABELS)

    _train_impl(0, model, train_set, test_set, params)


def train_distributed(replica_id, replica_count, port, params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        "nccl", rank=replica_id, world_size=replica_count
    )
    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)

    model = UNet(num_classes=len(LABELS), params=params).cuda()
    train_set = dataset_from_path(params['train_dirs'], params, LABELS, num_workers=os.cpu_count()//2, distributed=True)
    test_set = dataset_from_path(params['test_dirs'], params, LABELS, num_workers=os.cpu_count()//2)
    model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)

    _train_impl(replica_id, model, train_set, test_set, params, distributed=True)
