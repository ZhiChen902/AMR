import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from geotransformer.utils.diffusion_util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.cat([torch.Tensor([1.0]).cuda(), self.alpha_hat[:-1]])

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # smaller t return a small noise
    def noise_images(self, x, t):
        # t = 0
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn_like(x)
        # return torch.clamp(sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, min = 0), Ɛ
        return torch.clamp(sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, min = 1e-6), Ɛ, sqrt_alpha_hat, sqrt_one_minus_alpha_hat

    def Transformation_to_euler(self, Transformation):
        if isinstance(Transformation, torch.Tensor):
            rotation = Transformation[:3, :3].cpu().numpy()
            euler = Rotation.from_matrix(rotation).as_euler('zxy').astype(np.float32)
            euler= torch.from_numpy(euler).cuda()
        else:
            euler = Rotation.from_matrix(Transformation[:3, :3]).as_euler('zxy').astype(np.float32)
            euler= torch.from_numpy(euler).cuda()
        return euler

    def Rotation_to_euler(self, rotation):
        if isinstance(rotation, torch.Tensor):
            rotation = rotation.cpu().numpy()
            euler = Rotation.from_matrix(rotation).as_euler('zxy').astype(np.float32)
            euler= torch.from_numpy(euler).cuda()
        else:
            euler = Rotation.from_matrix(rotation).as_euler('zxy').astype(np.float32)
            euler= torch.from_numpy(euler).cuda()
        return euler
        

    def euler_to_Rotation(self, euler):
        if isinstance(euler, torch.Tensor):
            rotation = Rotation.from_euler('zxy', euler.cpu().numpy()).as_matrix().astype(np.float32)
            rotation = torch.from_numpy(rotation).cuda()
        else:    
            rotation = Rotation.from_euler('zxy', euler).as_matrix().astype(np.float32)
            rotation = torch.from_numpy(rotation).cuda()
        return rotation


    def Transformation_matmul(self, weight, Transformation):
        rotation = Transformation[:3, :3]
        euler = self.Rotation_to_euler(rotation)
        rotation = self.euler_to_Rotation(weight * euler)
        translation = Transformation[:3, 3] * weight
        Transformation[:3, :3] = rotation
        Transformation[:3, 3] = translation
        return Transformation

    def Transformation_div(self, Transformation, weight):
        rotation = Transformation[:3, :3]
        euler = self.Rotation_to_euler(rotation)
        rotation = self.euler_to_Rotation(euler / weight)
        translation = Transformation[:3, 3] / weight
        Transformation[:3, :3] = rotation
        Transformation[:3, 3] = translation
        return Transformation

    def Transformation_add(self, Transformation1, Transformation2):
        rotation1 = Transformation1[:3, :3]
        euler1 = self.Rotation_to_euler(rotation1)
        rotation2 = Transformation2[:3, :3]
        euler2 = self.Rotation_to_euler(rotation2)
        rotation = self.euler_to_Rotation(euler1 + euler2)
        translation = Transformation1[:3, 3] + Transformation2[:3, 3]
        Transformation = Transformation1
        Transformation[:3, :3] = rotation
        Transformation[:3, 3] = translation
        return Transformation
    
    def Transformation_sub(self, Transformation1, Transformation2):
        rotation1 = Transformation1[:3, :3]
        euler1 = self.Rotation_to_euler(rotation1)
        rotation2 = Transformation2[:3, :3]
        euler2 = self.Rotation_to_euler(rotation2)
        rotation = self.euler_to_Rotation(euler1 - euler2)
        translation = Transformation1[:3, 3] - Transformation2[:3, 3]
        Transformation = Transformation1
        Transformation[:3, :3] = rotation
        Transformation[:3, 3] = translation
        return Transformation


    def noise_RT(self, gt_rotation, gt_transation, t, max_noise_R, max_noise_t):
        # t = 20
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        # Ɛ = torch.randn([6])

        #add noise to rotation
        gt_euler = self.Rotation_to_euler(gt_rotation)
        noise_euler = torch.from_numpy(np.random.rand(3) * np.pi * max_noise_R / 180).cuda() # (0, 2 * pi / rotation_range)

        gt_rotation_with_noise = sqrt_alpha_hat * gt_euler + sqrt_one_minus_alpha_hat * noise_euler   
        gt_rotation_with_noise = self.euler_to_Rotation(gt_rotation_with_noise)
        
        #add noise to translation
        noise_transation = np.random.rand(3) * max_noise_t # 
        noise_transation = torch.from_numpy(noise_transation).cuda()

        # TODO: use cyclic noise, or clip noise, because adding guassian noise to cyclic system is not reasonable
        gt_transation_with_noise = sqrt_alpha_hat * gt_transation + sqrt_one_minus_alpha_hat * noise_transation

        # import ipdb; ipdb.set_trace()
        # self.Rotation_to_euler(gt_rotation_with_noise)
        # self.Rotation_to_euler(gt_rotation)

        # tmp = torch.eye(3)
        # tmp[0,0] = 0
        # tmp[1,1] = 0
        # tmp[1,0] = 1
        # tmp[0,1] = 1
        # self.Rotation_to_euler(tmp)

        noised_transform = torch.eye(4)
        noised_transform[:3, :3] = gt_rotation_with_noise
        noised_transform[:3, 3] = gt_transation_with_noise
        noised_transform = noised_transform.cuda()

        return noised_transform, sqrt_alpha_hat, sqrt_one_minus_alpha_hat
    
    def mix_prior_with_GT(self, gt_trans, prior_trans, t):
        # import pdb; pdb.set_trace()
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = 1 - sqrt_alpha_hat

        Trans1 = self.Transformation_matmul(sqrt_alpha_hat, gt_trans)
        Trans2 = self.Transformation_matmul(sqrt_one_minus_alpha_hat, prior_trans)
        noised_transform = self.Transformation_add(Trans1, Trans2)
        # import pdb; pdb.set_trace()

        return noised_transform, sqrt_alpha_hat, sqrt_one_minus_alpha_hat
    
    def mix_prior_with_GT_Slerp(self, gt_trans, prior_trans, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = 1 - sqrt_alpha_hat
        R_gt = gt_trans[:3, :3].cpu().numpy()
        t_gt = gt_trans[:3, 3]

        R_prior = prior_trans[:3, :3].cpu().numpy()
        t_prior = prior_trans[:3, 3]

        Rs = Rotation.from_matrix(np.concatenate((R_prior[None, :, :], R_gt[None, :, :]), axis=0))
        key_times = [0, 1]
        slerp = Slerp(key_times, Rs)
        interp_rot = torch.from_numpy(slerp(sqrt_alpha_hat.cpu().numpy()).as_matrix()).cuda()
        interp_t = t_gt * sqrt_alpha_hat + sqrt_one_minus_alpha_hat * t_prior

        Transformation = torch.eye(4).cuda()
        Transformation[:3, :3] = interp_rot
        Transformation[:3, 3] = interp_t
        # import pdb; pdb.set_trace()
        return Transformation, sqrt_alpha_hat, sqrt_one_minus_alpha_hat

    def init_noise_RT(self, max_noise_R, max_noise_t):
        noise_euler = torch.from_numpy(np.random.rand(3) * np.pi * max_noise_R / 180).cuda() # (0, 2 * pi / rotation_range)
        noise_rotation = self.euler_to_Rotation(noise_euler)
        
        #add noise to translation
        noise_transation = np.random.rand(3) * max_noise_t # 
        noise_transation = torch.from_numpy(noise_transation).cuda()

        noised_transform = torch.eye(4).cuda()
        noised_transform[:3, :3] = noise_rotation
        noised_transform[:3, 3] = noise_transation
        return noised_transform


    def init_noise_images(self, matrix_shape):
        t = 999
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn(matrix_shape[0], matrix_shape[1]).cuda()
        # return torch.clamp(sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, min = 0), Ɛ
        return torch.clamp(sqrt_alpha_hat * Ɛ + sqrt_one_minus_alpha_hat * Ɛ, min = 1e-6), Ɛ, sqrt_alpha_hat, sqrt_one_minus_alpha_hat

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


class DDIMSampler(object):
    def __init__(self, schedule="linear", ddpm_num_timesteps=1000, device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.ddpm_num_timesteps = ddpm_num_timesteps
        self.schedule = schedule
        self.device = device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, diffusion, device, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = diffusion.alpha_hat
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(device)

        self.register_buffer('betas', to_torch(diffusion.beta))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(diffusion.alpha_hat_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
