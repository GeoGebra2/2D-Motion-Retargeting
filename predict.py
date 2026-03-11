import os
import re
from scipy.ndimage import gaussian_filter1d
import torch
import argparse
import numpy as np
from dataset import get_meanpose
from model import get_autoencoder
from functional.visualization import motion2video, hex2rgb
from functional.motion import preprocess_motion2d, postprocess_motion2d, openpose2motion, ntu2motion, base15_to_ntu25_2d, write_ntu_skeleton
from functional.utils import ensure_dir, pad_to_height
from common import config


def handle2x(config, args):
    h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
    h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)

    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    mean_pose, std_pose = get_meanpose(config)

    if args.vid1_json_dir is not None:
        input1 = openpose2motion(args.vid1_json_dir, scale=scale1, max_frame=args.max_length)
    else:
        input1 = ntu2motion(args.ntu1, max_frame=args.max_length)
    if args.vid2_json_dir is not None:
        input2 = openpose2motion(args.vid2_json_dir, scale=scale2, max_frame=args.max_length)
    else:
        input2 = ntu2motion(args.ntu2, max_frame=args.max_length)
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)

    out12 = net.transfer(input1, input2)
    out21 = None if args.only_out12 else net.transfer(input2, input1)

    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    out12 = postprocess_motion2d(out12, mean_pose, std_pose, w2 // 2, h2 // 2)
    if out21 is not None:
        out21 = postprocess_motion2d(out21, mean_pose, std_pose, w1 // 2, h1 // 2)

    if not args.disable_smooth:
        out12 = gaussian_filter1d(out12, sigma=2, axis=-1)
        if out21 is not None:
            out21 = gaussian_filter1d(out21, sigma=2, axis=-1)

    if args.out_dir is not None:
        save_dir = args.out_dir
        ensure_dir(save_dir)
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)
        if args.only_out12:
            np.savez(os.path.join(save_dir, 'results.npz'),
                     input1=input1,
                     input2=input2,
                     out12=out12)
        else:
            np.savez(os.path.join(save_dir, 'results.npz'),
                     input1=input1,
                     input2=input2,
                     out12=out12,
                     out21=out21)
        if getattr(args, 'save_skeleton', False):
            def parse_codes(name):
                m = re.search(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', name)
                if not m:
                    return None
                return {'S': m.group(1), 'C': m.group(2), 'P': m.group(3), 'R': m.group(4), 'A': m.group(5)}
            def build_name(c):
                return f"S{c['S']}C{c['C']}P{c['P']}R{c['R']}A{c['A']}.skeleton"
            out12_name = 'out12.skeleton'
            out21_name = 'out21.skeleton'
            if args.ntu1 is not None and args.ntu2 is not None:
                ca = parse_codes(os.path.basename(args.ntu1))
                cb = parse_codes(os.path.basename(args.ntu2))
                if ca and cb:
                    c12 = {'S': cb['S'], 'C': cb['C'], 'P': ca['P'], 'R': cb['R'], 'A': ca['A']}
                    c21 = {'S': ca['S'], 'C': ca['C'], 'P': ca['P'], 'R': ca['R'], 'A': cb['A']}
                    out12_name = build_name(c12)
                    out21_name = build_name(c21)
            if getattr(args, 'fname_suffix', None):
                suf = str(args.fname_suffix)
                if suf:
                    if out12_name.endswith('.skeleton'):
                        out12_name = out12_name[:-9] + f"F{suf}.skeleton"
                    if (not args.only_out12) and out21_name.endswith('.skeleton'):
                        out21_name = out21_name[:-9] + f"F{suf}.skeleton"
            out12_25 = base15_to_ntu25_2d(out12)
            write_ntu_skeleton(os.path.join(save_dir, out12_name), out12_25)
            if not args.only_out12:
                out21_25 = base15_to_ntu25_2d(out21)
                write_ntu_skeleton(os.path.join(save_dir, out21_name), out21_25)
        if (not args.no_video) and args.render_video:
            print("Generating videos...")
            motion2video(input1, h1, w1, os.path.join(save_dir, 'input1.mp4'), color1, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(input2, h2, w2, os.path.join(save_dir,'input2.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(out12, h2, w2, os.path.join(save_dir,'out12.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            if not args.only_out12:
                motion2video(out21, h1, w1, os.path.join(save_dir,'out21.mp4'), color1, args.transparency,
                             fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def handle3x(config, args):
    h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
    h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)
    h3, w3, scale3 = pad_to_height(config.img_size[0], args.img2_height, args.img3_width)

    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    mean_pose, std_pose = get_meanpose(config)

    if args.vid1_json_dir is not None:
        input1 = openpose2motion(args.vid1_json_dir, scale=scale1, max_frame=args.max_length)
    else:
        input1 = ntu2motion(args.ntu1, max_frame=args.max_length)
    if args.vid2_json_dir is not None:
        input2 = openpose2motion(args.vid2_json_dir, scale=scale2, max_frame=args.max_length)
    else:
        input2 = ntu2motion(args.ntu2, max_frame=args.max_length)
    if args.vid3_json_dir is not None:
        input3 = openpose2motion(args.vid3_json_dir, scale=scale3, max_frame=args.max_length)
    else:
        input3 = ntu2motion(args.ntu3, max_frame=args.max_length)
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    input3 = preprocess_motion2d(input3, mean_pose, std_pose)
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    input3 = input3.to(config.device)

    out = net.transfer_three(input1, input2, input3)

    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    input3 = postprocess_motion2d(input3, mean_pose, std_pose, w2 // 2, h2 // 2)
    out = postprocess_motion2d(out, mean_pose, std_pose, w2 // 2, h2 // 2)

    if not args.disable_smooth:
        out = gaussian_filter1d(out, sigma=2, axis=-1)

    if args.out_dir is not None:
        save_dir = args.out_dir
        ensure_dir(save_dir)
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)
        color3 = hex2rgb(args.color3)
        np.savez(os.path.join(save_dir, 'results.npz'),
                 input1=input1,
                 input2=input2,
                 input3=input3,
                 out=out)
        if getattr(args, 'save_skeleton', False):
            out_name = 'out.skeleton'
            if args.ntu1 is not None and args.ntu2 is not None:
                m1 = re.search(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', os.path.basename(args.ntu1) or '')
                m2 = re.search(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', os.path.basename(args.ntu2) or '')
                m3 = re.search(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', os.path.basename(args.ntu3) or '') if getattr(args, 'ntu3', None) else None
                if m1 and m2:
                    c_code = m3.group(2) if m3 else m2.group(2)
                    c = {'S': m2.group(1), 'C': c_code, 'P': m2.group(3), 'R': m2.group(4), 'A': m1.group(5)}
                    out_name = f"S{c['S']}C{c['C']}P{c['P']}R{c['R']}A{c['A']}.skeleton"
            if getattr(args, 'fname_suffix', None):
                suf = str(args.fname_suffix)
                if suf and out_name.endswith('.skeleton'):
                    out_name = out_name[:-9] + f"F{suf}.skeleton"
            out_25 = base15_to_ntu25_2d(out)
            write_ntu_skeleton(os.path.join(save_dir, out_name), out_25)
        if (not args.no_video) and args.render_video:
            print("Generating videos...")
            motion2video(input1, h1, w1, os.path.join(save_dir,'input1.mp4'), color1, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(input2, h2, w2, os.path.join(save_dir,'input2.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(input3, h3, w3, os.path.join(save_dir,'input3.mp4'), color3, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(out, h2, w2, os.path.join(save_dir,'out.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, choices=['skeleton', 'view', 'full'], required=True,
                        help='which structure to use.')
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('-v1', '--vid1_json_dir', type=str, help="video1's openpose json directory")
    parser.add_argument('-v2', '--vid2_json_dir', type=str, help="video2's openpose json directory")
    parser.add_argument('-v3', '--vid3_json_dir', type=str, help="video3's openpose json directory")
    parser.add_argument('--ntu1', type=str, help="ntu skeleton file for input1")
    parser.add_argument('--ntu2', type=str, help="ntu skeleton file for input2")
    parser.add_argument('--ntu3', type=str, help="ntu skeleton file for input3")
    parser.add_argument('--save_skeleton', action='store_true', help="save outputs as NTU .skeleton files")
    parser.add_argument('--fname_suffix', type=str, help="append Fxxx to skeleton filename (e.g., 001)")
    parser.add_argument('--only_out12', action='store_true', help="only save/compute out12 in 2-input mode")
    parser.add_argument('--no_video', action='store_true', help="do not render videos")
    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height")
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width")
    parser.add_argument('-h2', '--img2_height', type=int, help="video2's height")
    parser.add_argument('-w2', '--img2_width', type=int, help="video2's width")
    parser.add_argument('-h3', '--img3_height', type=int, help="video3's height")
    parser.add_argument('-w3', '--img3_width', type=int, help="video3's width")
    parser.add_argument('-o', '--out_dir', type=str, default='./outputs', help="output saving directory")
    parser.add_argument('--render_video', type=bool, default=True, help="whether to save rendered video")
    parser.add_argument('--fps', type=float, default=25, help="fps of output video")
    parser.add_argument('--save_frame', action='store_true', help="to save rendered video frames")
    parser.add_argument('--color1', type=str, default='#a50b69#b73b87#db9dc3', help='color1')
    parser.add_argument('--color2', type=str, default='#4076e0#40a7e0#40d7e0', help='color2')
    parser.add_argument('--color3', type=str, default='#ff8b06#ffb431#ffcd9d', help='color3')
    parser.add_argument('--disable_smooth', action='store_true',
                        help="disable gaussian kernel smoothing")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
    parser.add_argument('--max_length', type=int, default=120,
                        help='maximum input video length')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    config.initialize(args)

    if args.name == 'full':
        handle3x(config, args)
    else:
        handle2x(config, args)


if __name__ == '__main__':
    main()
