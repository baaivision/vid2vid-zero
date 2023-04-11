from __future__ import annotations

import datetime
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import gradio as gr
import slugify
import torch
import huggingface_hub
from huggingface_hub import HfApi
from omegaconf import OmegaConf


ORIGINAL_SPACE_ID = 'BAAI/vid2vid-zero'
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)


class Runner:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token

        self.checkpoint_dir = pathlib.Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

    def download_base_model(self, base_model_id: str, token=None) -> str:
        model_dir = self.checkpoint_dir / base_model_id
        org_name = base_model_id.split('/')[0]
        org_dir = self.checkpoint_dir / org_name
        if not model_dir.exists():
            org_dir.mkdir(exist_ok=True)
        print(f'https://huggingface.co/{base_model_id}')
        if token == None:
            subprocess.run(shlex.split(f'git lfs install'), cwd=org_dir)
            subprocess.run(shlex.split(
                f'git lfs clone https://huggingface.co/{base_model_id}'),
                            cwd=org_dir)
            return model_dir.as_posix()
        else:
            temp_path = huggingface_hub.snapshot_download(base_model_id, use_auth_token=token)
            print(temp_path, org_dir)
            # subprocess.run(shlex.split(f'mv {temp_path} {model_dir.as_posix()}'))
            # return model_dir.as_posix()
            return temp_path

    def join_model_library_org(self, token: str) -> None:
        subprocess.run(
            shlex.split(
                f'curl -X POST -H "Authorization: Bearer {token}" -H "Content-Type: application/json" {URL_TO_JOIN_MODEL_LIBRARY_ORG}'
            ))

    def run_vid2vid_zero(
        self,
        model_path: str,
        input_video: str,
        prompt: str,
        n_sample_frames: int,
        sample_start_idx: int,
        sample_frame_rate: int,
        validation_prompt: str,
        guidance_scale: float,
        resolution: str,
        seed: int,
        remove_gpu_after_running: bool,
        input_token: str = None,
    ) -> str:

        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        if input_video is None:
            raise gr.Error('You need to upload a video.')
        if not prompt:
            raise gr.Error('The input prompt is missing.')
        if not validation_prompt:
            raise gr.Error('The validation prompt is missing.')

        resolution = int(resolution)
        n_sample_frames = int(n_sample_frames)
        sample_start_idx = int(sample_start_idx)
        sample_frame_rate = int(sample_frame_rate)

        repo_dir = pathlib.Path(__file__).parent
        prompt_path = prompt.replace(' ', '_')
        output_dir = repo_dir / 'outputs' / prompt_path
        output_dir.mkdir(parents=True, exist_ok=True)

        config = OmegaConf.load('configs/black-swan.yaml')
        config.pretrained_model_path = self.download_base_model(model_path, token=input_token)

        # we remove null-inversion & use fp16 for fast inference on web demo
        config.mixed_precision = "fp16"
        config.validation_data.use_null_inv = False

        config.output_dir = output_dir.as_posix()
        config.input_data.video_path = input_video.name  # type: ignore
        config.input_data.prompt = prompt
        config.input_data.n_sample_frames = n_sample_frames
        config.input_data.width = resolution
        config.input_data.height = resolution
        config.input_data.sample_start_idx = sample_start_idx
        config.input_data.sample_frame_rate = sample_frame_rate

        config.validation_data.prompts = [validation_prompt]
        config.validation_data.video_length = 8
        config.validation_data.width = resolution
        config.validation_data.height = resolution
        config.validation_data.num_inference_steps = 50
        config.validation_data.guidance_scale = guidance_scale

        config.input_batch_size = 1
        config.seed = seed

        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

        command = f'accelerate launch test_vid2vid_zero.py --config {config_path}'
        subprocess.run(shlex.split(command))

        output_video_path = os.path.join(output_dir, "sample-all.mp4")
        print(f"video path for gradio: {output_video_path}")
        message = 'Running completed!'
        print(message)

        if remove_gpu_after_running:
            space_id = os.getenv('SPACE_ID')
            if space_id:
                api = HfApi(
                    token=self.hf_token if self.hf_token else input_token)
                api.request_space_hardware(repo_id=space_id,
                                           hardware='cpu-basic')

        return output_video_path
