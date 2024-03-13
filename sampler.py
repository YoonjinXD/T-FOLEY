import torch
from scipy.integrate import solve_ivp


class SDESampling:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule
    
    def conditional_inference(self, audio, sigma, classes, events, cond_scale=1):
        cond_drop_prob = [0.0, 1.0] if events == None else [0.0, 0.0]
        cond_score = self.model(audio, sigma, classes, events, cond_drop_prob=cond_drop_prob)
        if cond_scale != 1: 
            uncond_score = self.model(audio, sigma, classes, events, cond_drop_prob=[1.0, 1.0])
            cond_score = uncond_score + (cond_score - uncond_score) * cond_scale
        
        return cond_score
    
    def predict(
        self,
        audio,
        nb_steps,
        classes,
        amplitude,
        cond_scale = 3.
    ):

        with torch.no_grad():

            sigma, m = self.create_schedules(nb_steps)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)
                
                cond_score = self.conditional_inference(audio, sigma[n], classes, amplitude, cond_scale)
                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * cond_score

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            cond_score = self.conditional_inference(audio, sigma[0], classes, amplitude, cond_scale)
            audio = (audio - sigma[0] * cond_score) / m[0]

        return audio
    
class SDESampling_batch:
    def __init__(self, model, sde, batch_size, device):
        self.model = model
        self.sde = sde
        self.batch_size = batch_size
        self.device = device

    def create_schedules(self, nb_steps, batch):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        t_schedule = t_schedule.expand(batch, -1)
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule
    
    def conditional_inference(self, audio, sigma, classes, events, cond_scale=1):
        cond_drop_prob = [0.0, 1.0] if events == None else [0.0, 0.0]
        cond_score = self.model(audio, sigma, classes, events, cond_drop_prob=cond_drop_prob)
        if cond_scale != 1: 
            uncond_score = self.model(audio, sigma, classes, events, cond_drop_prob=[1.0, 1.0])
            cond_score = uncond_score + (cond_score - uncond_score) * cond_scale
        
        return cond_score
    
    def predict(
        self,
        audio,
        nb_steps,
        classes,
        amplitude,
        cond_scale = 3.
    ):

        with torch.no_grad():

            sigma, m = self.create_schedules(nb_steps, self.batch_size)
            sigma = sigma.permute((1,0)).unsqueeze(2).to(self.device)
            m = m.permute((1,0)).unsqueeze(2).to(self.device)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)
                
                cond_score = self.conditional_inference(audio, sigma[n], classes, amplitude, cond_scale)
                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * cond_score

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            cond_score = self.conditional_inference(audio, sigma[0], classes, amplitude, cond_scale)
            audio = (audio - sigma[0] * cond_score) / m[0]

        return audio
