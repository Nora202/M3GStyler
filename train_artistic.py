#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from torchvision import transforms
import numpy as np
from scene.VGG import VGGEncoder, normalize_vgg
from utils.loss_utils import cal_adain_style_loss, cal_mse_content_loss
from PIL import ImageFile
import clip
import torch.nn.functional as F
from einops import repeat
from scene.style_transfer import calc_mean_std,sample_ode,get_train_tuple
ImageFile.LOAD_TRUNCATED_IMAGES = True
import wandb
import torchvision
import torch.nn as nn
import math
from torchvision.transforms.functional import to_pil_image
from scene.net import vgg,decoder
wandb.init()
device = torch.device('cuda')


def dis_loss( vgg, clip):
    loss = F.mse_loss(vgg, clip)

    return loss

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def getDataLoader(dataset_path, batch_size, sampler, image_side_length=256, num_workers=0):
    transform = T.Compose([
                T.Resize(size=(image_side_length*2, image_side_length*2)),
                T.RandomCrop(image_side_length),
                T.ToTensor(),
            ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler(len(train_dataset)), num_workers=num_workers)

    return dataloader

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0
class new_InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]


def training(dataset, opt, pipe, ckpt_path, decoder_path,mapping_path,content_preserve,ode,n,iters):
    opt.iterations = iters #if not decoder_path else 30_000
    first_iter = 0
    new_path = args.model_path
    os.makedirs(new_path, exist_ok=True)
    tb_writer = prepare_output_and_logger(dataset, new_path)

    content_weight, style_weight, s_a_weight, g_weight=1,10,10,10

    gaussians = GaussianModel(dataset.sh_degree)
    # load the feature reconstructed gaussians ckpt file
    scene = Scene(dataset, gaussians, load_path=ckpt_path)
    vgg_encoder = VGGEncoder().cuda()


    clip_model, _ = clip.load("ViT-B/32", device='cuda')

    # compute the final vgg features for each point, and init pointnet decoder
    gaussians.training_setup_style(opt, decoder_path,mapping_path,ode=ode,n=n)

    # init wikiart dataset
    style_loader = getDataLoader(args.wikiartdir, batch_size=1, sampler=InfiniteSamplerWrapper,
                    image_side_length=256, num_workers=4)
    style_iter = iter(style_loader)

    bg_color = [1]*3 if dataset.white_background else [0]*3
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder_2D=decoder
    vgg_2D=vgg
    decoder_2D.load_state_dict(torch.load("decoder.pth"))
    vgg_2D.load_state_dict(torch.load("vgg_normalised.pth"))
    vgg_2D = nn.Sequential(*list(vgg.children())[:31])
    vgg_2D.to(device)
    decoder_2D.to(device)

    features_cpu = gaussians.final_vgg_features.detach().to("cpu")  
    final_vgg_features_low, final_vgg_features_high = split_frequency(features_cpu)
    final_vgg_features_low = final_vgg_features_low.to("cuda")
    final_vgg_features_high = final_vgg_features_high.to("cuda")

    pe_xyz = positional_encoding(gaussians.get_xyz.detach())  # [N, D]

    for j in range(0,ode+1):

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Artistic training", bar_format='{l_bar}{r_bar}')
        first_iter = 1

        for iteration in range(first_iter, opt.iterations + 1):

            if content_preserve and iteration % 5 == 0:
                decoded_rgb = gaussians.decoder(gaussians.final_vgg_features.detach())  # [N, 3]
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=decoded_rgb)
                rendered_rgb = render_pkg["render"]  # [3, H, W]
                gt_image = viewpoint_cam.original_image.cuda()  # [3, H, W]
                loss = l1_loss(gt_image, rendered_rgb)
                loss.backward()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            views=scene.getTrainCameras().copy()

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            style_img = next(style_iter)[0].cuda()
            gt_image = viewpoint_cam.original_image.cuda()  # [3, H, W]

            # Render
            with torch.no_grad():
                style_img_features = vgg_encoder(normalize_vgg(style_img))
                gt_image_features = vgg_encoder(normalize_vgg(gt_image.unsqueeze(0)))
                _style_image_processed = clip_normalize(normalize_vgg(style_img))
                _sF_CLIP_feature = clip_model.encode_image(_style_image_processed).detach().requires_grad_(True)


            iter_start.record()

            transfered_features, cv_loss= gaussians.style_transfer(
                    final_vgg_features_low.detach(), # point cloud features [N, C]
                    style_img_features.relu3_1,
                    _sF_CLIP_feature,
                    train_step=j,
                )


            alpha=0.1

            #override_rgb=decoded_rgb+ alpha * gaussians.style_transfer.high_compress(final_vgg_features_high.detach())
            if j==0 or j==1:
                override_rgb = gaussians.decoder(transfered_features)
            if j==2:
                override_rgb = gaussians.decoder(transfered_features + alpha * final_vgg_features_high)
            if j==3:
                belta=0.1
                pe_proj = gaussians.style_transfer.pe_proj(pe_xyz)
                override_rgb = transfered_features+ belta* pe_proj+alpha * final_vgg_features_high
                override_rgb = gaussians.decoder(override_rgb)

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=override_rgb)
            rendered_rgb = render_pkg["render"]  # [3, H, W]

            # style loss and content loss
            rendered_rgb_features = vgg_encoder(normalize_vgg(rendered_rgb.unsqueeze(0)))

            style_loss = 0.
            for style_feature, image_feature in zip(style_img_features, rendered_rgb_features):
                style_loss += cal_adain_style_loss(style_feature, image_feature)


            content_loss = cal_mse_content_loss(gt_image_features.relu4_1, rendered_rgb_features.relu4_1)

            if args.loss_gan==True:
                valid, fake = 1, 0

                with torch.no_grad():
                    style_image_2d = style_transfer_2d(vgg_2D, decoder_2D, gt_image.unsqueeze(0), style_img, alpha=1.0)
                    H, W = rendered_rgb.shape[1:]
                    style_image_2d = F.interpolate(style_image_2d, size=(H, W), mode='bilinear', align_corners=False)
                    loss_gan_g = gaussians.style_transfer.discriminator.compute_loss(rendered_rgb, valid)
                    loss = content_loss * content_weight + style_loss * style_weight + cv_loss * s_a_weight + g_weight * loss_gan_g

                if args.loss==True:
                    with torch.no_grad():
                        _, style_image_2d_high = split_frequency(style_image_2d[0].cpu())
                        style_image_2d_high = style_image_2d_high.to('cuda')
                        style_image_2d_high = vgg_encoder(normalize_vgg(style_image_2d_high.unsqueeze(0).detach()))
                        _, rendered_rgb_high = split_frequency(rendered_rgb.cpu())
                        rendered_rgb_high=rendered_rgb_high.to('cuda')
                        rendered_rgb_high_features = vgg_encoder(normalize_vgg(rendered_rgb_high.unsqueeze(0)))

                    loss_cross = 0.0
                    for style_feature, image_feature in zip(rendered_rgb, style_image_2d[0]):
                        loss_cross += cal_adain_style_loss(image_feature, style_feature)

                    loss_gan_g = gaussians.style_transfer.discriminator.compute_loss(rendered_rgb_high, valid)
                    loss = content_loss * content_weight + style_loss * style_weight + cv_loss * s_a_weight + 10 * loss_cross+g_weight * loss_gan_g
                    loss.backward()
                    gaussians.optimizer_1.step()
                    gaussians.optimizer_1.zero_grad(set_to_none=True)

                    del rendered_rgb_high_features, style_image_2d_high, rendered_rgb_high
                    del gaussians.final_vgg_features_low, gaussians.final_vgg_features_high

                else:

                    loss_gan_g = gaussians.style_transfer.discriminator.compute_loss(rendered_rgb_high, valid)
                    loss = content_loss  + style_loss * style_weight +g_weight * loss_gan_g+cv_loss*s_a_weight

                gaussians.optimizer_GAN.zero_grad()
                loss.backward()
                gaussians.optimizer_GAN.step()


                loss_gan_d = (
                                     gaussians.style_transfer.discriminator.compute_loss(style_image_2d_high.detach(),
                                                                                         valid) +
                                     gaussians.style_transfer.discriminator.compute_loss(rendered_rgb_high_features.detach(), fake)
                             ) * 0.5
                gaussians.optimizer_Dis.zero_grad()
                loss_gan_d.backward()
                gaussians.optimizer_Dis.step()
            else:
                loss = content_loss + style_loss * style_weight  + cv_loss * s_a_weight
                gaussians.optimizer_GAN.zero_grad()
                loss.backward()
                gaussians.optimizer_GAN.step()

            torch.cuda.empty_cache()
            iter_end.record()

            torch.cuda.empty_cache()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1)
                if iteration == opt.iterations:
                    progress_bar.close()

                wandb.log({'train_loss/loss': loss.item(), 'iteration': iteration})
                wandb.log({'train_loss/cv_loss': cv_loss.item(), 'iteration': iteration})
                wandb.log({'train_loss/content_loss': content_loss.item(), 'iteration': iteration})
                wandb.log({'train_loss/style_loss': style_loss.item(), 'iteration': iteration})
                #wandb.log({'train_loss/loss_gan_g': loss_gan_g.item(), 'iteration': iteration})
                #wandb.log({'train_loss/loss_gan_d': loss_gan_d.item(), 'iteration': iteration})
                #wandb.log({'train_loss/loss_texture': loss_texture.item(), 'iteration': iteration})

        os.makedirs(args.model_path+ "/chkpnt/" , exist_ok=True)
        torch.save(gaussians.capture(is_style_model=True), args.model_path + "/chkpnt/" + str(j)+'_'+str(n) + "_gaussians.pth")

def prepare_output_and_logger(args,new_path):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

    # Set up output folder
    print("Output folder: {}".format(new_path))

    with open(os.path.join(new_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(new_path)

    return tb_writer

def clip_normalize(image):
    image = F.interpolate(image,size=224,mode='bicubic')
    # image = self.avg_pool(self.upsample(image))

    b, *_ = image.shape
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    mean = repeat(mean.view(1, -1, 1, 1), '1 ... -> b ...', b=b)
    std = repeat(std.view(1, -1, 1, 1), '1 ... -> b ...', b=b)

    image = (image - mean) / std
    return image

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--mapping_path", type=str, default='MLP2_iter_160000.pth')
    parser.add_argument("--rendering_mode", type=str, default="rgb", choices=["rgb", "feature"])
    parser.add_argument("--wikiartdir", type=str, default="datasets/archive")
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--style_weight", type=float, default=1.)
    parser.add_argument("--content_weight", type=float, default=1.)
    parser.add_argument("--s_a_weight", type=float, default=1.)
    parser.add_argument("--g_weight", type=float, default=1.)
    parser.add_argument("--content_preserve", action='store_true', default=False)
    parser.add_argument("--loss_texture", action='store_true', default=False)
    parser.add_argument("--loss_cross", action='store_true', default=False)
    parser.add_argument("--clip_text", default=False)
    parser.add_argument("--ode", type=int,default=0)
    parser.add_argument("--n", type=int,default=100)
    parser.add_argument('--iters', type=int, default=10000)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]
    #print(args.source_path)
    #print(args.exp_name)

    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "artistic", args.exp_name+'_'+str(args.iters))
    print("Optimizing " + args.model_path + (' with content_preserve' if args.content_preserve else ''))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.ckpt_path, args.decoder_path, None,False,args.ode,args.n,args.iters)

    # All done
    print("\nArtistic training complete.")
