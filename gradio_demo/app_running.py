#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from gradio_demo.runner import Runner


def create_demo(runner: Runner,
                pipe: None = None) -> gr.Blocks:
    hf_token = os.getenv('HF_TOKEN')
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('Input Data')
                    input_video = gr.File(label='Input video')
                    input_prompt = gr.Textbox(
                        label='Input prompt',
                        max_lines=1,
                        placeholder='A car is moving on the road.')
                    gr.Markdown('''
                        - Upload a video and write a `Input Prompt` that describes the video.
                        ''')

            with gr.Column():
                with gr.Box():
                    gr.Markdown('Input Parameters')
                    with gr.Row():
                        model_path = gr.Text(
                            label='Path to off-the-shelf model',
                            value='CompVis/stable-diffusion-v1-4',
                            max_lines=1)
                        resolution = gr.Dropdown(choices=['512', '768'],
                                                 value='512',
                                                 label='Resolution',
                                                 visible=False)

                    with gr.Accordion('Advanced settings', open=False):
                        sample_start_idx = gr.Number(
                            label='Start Frame Index',value=0)
                        sample_frame_rate = gr.Number(
                            label='Frame Rate',value=1)
                        n_sample_frames = gr.Number(
                            label='Number of Frames',value=8)
                        guidance_scale = gr.Number(
                            label='Guidance Scale', value=7.5)
                        seed = gr.Slider(label='Seed',
                                         minimum=0,
                                         maximum=100000,
                                         step=1,
                                         randomize=True,
                                         value=33)
                        input_token = gr.Text(label='Hugging Face Write Token',
                                              placeholder='',
                                              visible=False if hf_token else True)
                    gr.Markdown('''
                        - Upload input video or choose an exmple blow
                        - Set hyperparameters & click start
                        - It takes a few minutes to download model first
                    ''')

        with gr.Row():
            with gr.Column():
                validation_prompt = gr.Text(
                    label='Validation Prompt',
                    placeholder=
                    'prompt to test the model, e.g: a Lego man is surfing')

        remove_gpu_after_running = gr.Checkbox(
            label='Remove GPU after running',
            value=False,
            interactive=bool(os.getenv('SPACE_ID')),
            visible=False)

        with gr.Row():
            result = gr.Video(label='Result')

        # examples
        with gr.Row():
            examples = [
                [
                    'CompVis/stable-diffusion-v1-4',
                    "data/car-moving.mp4",
                    'A car is moving on the road.',
                    8, 0, 1,
                    'A jeep car is moving on the desert.',
                    7.5, 512, 33,
                    False, None,
                ],

                [
                    'CompVis/stable-diffusion-v1-4',
                    "data/black-swan.mp4",
                    'A blackswan is swimming on the water.',
                    8, 0, 4,
                    'A white swan is swimming on the water.',
                    7.5, 512, 33,
                    False, None,
                ],

                [
                    'CompVis/stable-diffusion-v1-4',
                    "data/child-riding.mp4",
                    'A child is riding a bike on the road.',
                    8, 0, 1,
                    'A lego child is riding a bike on the road.',
                    7.5, 512, 33,
                    False, None,
                ],

                [
                    'CompVis/stable-diffusion-v1-4',
                    "data/car-turn.mp4",
                    'A jeep car is moving on the road.',
                    8, 0, 6,
                    'A jeep car is moving on the snow.',
                    7.5, 512, 33,
                    False, None,
                ],

                [
                    'CompVis/stable-diffusion-v1-4',
                    "data/rabbit-watermelon.mp4",
                    'A rabbit is eating a watermelon.',
                    8, 0, 6,
                    'A puppy is eating an orange.',
                    7.5, 512, 33,
                    False, None,
                ],

            ]
            gr.Examples(examples=examples,
                        fn=runner.run_vid2vid_zero,
                        inputs=[
                            model_path, input_video, input_prompt,
                            n_sample_frames, sample_start_idx, sample_frame_rate,
                            validation_prompt, guidance_scale, resolution, seed,
                            remove_gpu_after_running,
                            input_token,
                        ],
                        outputs=result,
                        cache_examples=os.getenv('SYSTEM') == 'spaces'
                        )

        # run
        run_button_vid2vid_zero = gr.Button('Start vid2vid-zero')
        run_button_vid2vid_zero.click(
            fn=runner.run_vid2vid_zero,
            inputs=[
                model_path, input_video, input_prompt,
                n_sample_frames, sample_start_idx, sample_frame_rate,
                validation_prompt, guidance_scale, resolution, seed,
                remove_gpu_after_running,
                input_token,
            ],
            outputs=result)

    return demo


if __name__ == '__main__':
    hf_token = os.getenv('HF_TOKEN')
    runner = Runner(hf_token)
    demo = create_demo(runner)
    demo.queue(max_size=1).launch(share=False)
