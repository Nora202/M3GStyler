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
import cv2
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spherical_sample_path, generate_spiral_path, generate_spherify_path, gaussian_poses, circular_poses
import torch.nn.functional as F
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from scene.VGG import VGGEncoder, normalize_vgg
import clip
from einops import repeat
from utils.loss_utils import calc_mean_std
from scene.style_transfer import get_train_tuple,sample_ode
from train_artistic import split_frequency, positional_encoding
import datetime
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
def render_set(model_path, name, views, gaussians, pipeline, background,i, style=None,from_text=False,render_video=True):
    pe_xyz = positional_encoding(gaussians.get_xyz.detach())
    if style:
        (style_img, style_name) = style
        render_path = os.path.join(model_path, name, str(style_name), "renders")
        makedirs(render_path, exist_ok=True)
        vgg_encoder = VGGEncoder().cuda()
        clip_model, _ = clip.load("ViT-B/32", device='cuda')
        style_img_features = vgg_encoder(normalize_vgg(style_img))
        _style_image_processed = clip_normalize(normalize_vgg(style_img))
        _sF_CLIP_feature = clip_model.encode_image(_style_image_processed)
        with ((torch.no_grad())):
            features_cpu = gaussians.final_vgg_features.detach().to("cpu")
            final_vgg_features_low, final_vgg_features_high = split_frequency(features_cpu)
            final_vgg_features_low = final_vgg_features_low.to("cuda")
            final_vgg_features_high = final_vgg_features_high.to("cuda")
        transfered_features, _ = gaussians.style_transfer(
            final_vgg_features_low.detach(),  # point cloud features [N, C]
            style_img_features.relu3_1,
            _sF_CLIP_feature,
             train_step=i,text_inference=False
        )

        alpha=0.1
        if i == 0 or i == 1:
            override_rgb = gaussians.decoder(transfered_features)
        if i == 2:
            override_rgb = gaussians.decoder(transfered_features + alpha * final_vgg_features_high)
        if i == 3:
            belta = 0.1
            pe_proj = gaussians.style_transfer.pe_proj(pe_xyz)
            override_rgb = transfered_features + belta * pe_proj + alpha * final_vgg_features_high
            override_rgb = gaussians.decoder(override_rgb)
        final_idx=0
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background, override_color=override_rgb )["render"]
            rendering = rendering.clamp(0, 1)
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            final_idx=idx
        if final_idx<50:
            final_idx=200
        elif final_idx<100:
            final_idx=400
        else:
            final_idx=600

        if render_video==True:
            render_video_path = os.path.join(model_path, name, str(style_name), "renders_video")
            makedirs(render_video_path, exist_ok=True)
            view = views[0]
            for idx, pose in enumerate(tqdm(generate_ellipse_path(views, final_idx), desc="Rendering progress")):
                view.world_view_transform = torch.tensor(
                    getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
                view.full_proj_transform = (
                    view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
                view.camera_center = view.world_view_transform.inverse()[3, :3]
                rendering = render(view, gaussians, pipeline, background, override_color=override_rgb)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_video_path, '{0:05d}'.format(idx) + ".png"))
            video_name = os.path.join(render_video_path, 'output_video.mp4')
            images_to_video(render_video_path, video_name)

    if args.text is not None and len(args.text) > 0:

        text=args.text
        text_fill = text[0].replace(' ', '_')
        render_path = os.path.join(model_path, name, str(text_fill)[:], "renders")

        makedirs(render_path, exist_ok=True)

        clip_model, _ = clip.load("ViT-B/32", device='cuda')
        kk = clip.tokenize(args.text, truncate=True).cuda()
        _sF_CLIP_feature = clip_model.encode_text(kk)
        with ((torch.no_grad())):
            features_cpu = gaussians.final_vgg_features.detach().to("cpu")
            final_vgg_features_low, final_vgg_features_high = split_frequency(features_cpu)
            final_vgg_features_low = final_vgg_features_low.to("cuda")
            final_vgg_features_high = final_vgg_features_high.to("cuda")
        transfered_features = gaussians.style_transfer(
            cF=final_vgg_features_low.detach(),  # point cloud features [N, C]
            sF=None,
            clip_feature=_sF_CLIP_feature,
             train_step=i, text_inference=True
        )
        alpha=0.1
        if i == 0 or i == 1:
            override_rgb = gaussians.decoder(transfered_features)
        if i == 2:
            override_rgb = gaussians.decoder(transfered_features + alpha * final_vgg_features_high)
        if i == 3:
            belta = 0.1
            pe_proj = gaussians.style_transfer.pe_proj(pe_xyz)
            override_rgb = transfered_features + belta * pe_proj + alpha * final_vgg_features_high
            override_rgb = gaussians.decoder(override_rgb)

        final_idx=0
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background, override_color=override_rgb)["render"]
            rendering = rendering.clamp(0, 1)
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            final_idx = idx
        if final_idx < 50:
            final_idx = 200
        elif final_idx < 100:
            final_idx = 400
        else:
            final_idx = 600

        if render_video == True:
            render_video_path = os.path.join(model_path, name, str(text_fill)[:], "renders_video")
            makedirs(render_video_path, exist_ok=True)
            view = views[0]
            for idx, pose in enumerate(tqdm(generate_ellipse_path(views, n_frames=final_idx), desc="Rendering progress")):
                view.world_view_transform = torch.tensor(
                    getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
                view.full_proj_transform = (
                    view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
                view.camera_center = view.world_view_transform.inverse()[3, :3]
                rendering = render(view, gaussians, pipeline, background, override_color=override_rgb)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_video_path, '{0:05d}'.format(idx) + ".png"))
            video_name = os.path.join(render_video_path, 'output_video.mp4')
            images_to_video(render_video_path, video_name)


def render_sets(dataset : ModelParams, pipeline : PipelineParams, style_img_path, text,render_video):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        if style_img_path:
            for i in range(0,4):
                ckpt_path = os.path.join(dataset.model_path, "chkpnt/"+str(i)+"_100_gaussians.pth")
                scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

                # read style image
                trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
                style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
                style_name = Path(style_img_path).stem
                style = (style_img, style_name)
                bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                render_set(dataset.model_path, "train_"+str(i),  scene.getTrainCameras(), gaussians, pipeline,background,i, style, False,render_video)

        if text!='':
            for i in range(0, 4):
                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                ckpt_path = os.path.join(dataset.model_path, "chkpnt/"+str(i)+"_100_gaussians.pth")
                scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)
                render_set(dataset.model_path, "train_"+str(i), scene.getTrainCameras(), gaussians, pipeline,
                               background,i, style=None,from_text=True, render_video=render_video)



def images_to_video(image_folder, video_file, fps=24):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 确保图片是按顺序处理

    # 从第一张图片获取尺寸信息
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以更改为其他编解码器
    video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
    i=0
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        if i>10:
            os.remove(os.path.join(image_folder, image))
        i=i+1
    video.release()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--style", nargs='+', default='', type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--from_vgg",default=False)
    parser.add_argument("--text", nargs='+', default='', type=str)
    parser.add_argument("--render_video", action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.text is not None and len(args.text) > 0:
        render_sets(model.extract(args), pipeline.extract(args), None,args.text,args.render_video)
    elif args.style is not None and len(args.style) > 0:
        render_sets(model.extract(args), pipeline.extract(args), args.style[0],None, args.render_video)

