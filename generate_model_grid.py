from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
from io import StringIO
from PIL import Image
import numpy as np
import os

import modules.scripts as scripts
import gradio as gr
from modules import images, sd_samplers
from modules.paths import models_path
from modules.hypernetworks import hypernetwork
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import re


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)

def edit_prompt(p,x,z):
    p.prompt = z + " " + x


def apply_order(p, x, xs):
    token_order = []

    # Initally grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt
    

def build_samplers_dict():
    samplers_dict = {}
    for i, sampler in enumerate(sd_samplers.all_samplers):
        samplers_dict[sampler.name.lower()] = i
        for alias in sampler.aliases:
            samplers_dict[alias.lower()] = i
    return samplers_dict


def apply_sampler(p, x, xs):
    sampler_index = build_samplers_dict().get(x.lower(), None)
    if sampler_index is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_index = sampler_index


def confirm_samplers(p, xs):
    samplers_dict = build_samplers_dict()
    for x in xs:
        if x.lower() not in samplers_dict.keys():
            raise RuntimeError(f"Unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    modules.sd_models.reload_model_weights(shared.sd_model, info)
    p.sd_model = shared.sd_model


def confirm_checkpoints(p, xs):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def apply_hypernetwork(p, x, xs):
    if x.lower() in ["", "none"]:
        name = None
    else:
        name = hypernetwork.find_closest_hypernetwork_name(x)
        if not name:
            raise RuntimeError(f"Unknown hypernetwork: {x}")
    hypernetwork.load_hypernetwork(name)


def apply_hypernetwork_strength(p, x, xs):
    hypernetwork.apply_strength(x)


def confirm_hypernetworks(p, xs):
    for x in xs:
        if x.lower() in ["", "none"]:
            continue
        if not hypernetwork.find_closest_hypernetwork_name(x):
            raise RuntimeError(f"Unknown hypernetwork: {x}")


def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x

# AxisOption = namedtuple("AxisOption", ["label", "type", "apply", "format_value", "confirm"])
# AxisOptionImg2Img = namedtuple("AxisOptionImg2Img", ["label", "type", "apply", "format_value", "confirm"])


def draw_xy_grid(p, xs, ys, zs, x_labels, y_labels, cell, draw_legend, include_lone_images):
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]

    # Temporary list of all the images that are generated to be populated into the grid.
    # Will be filled with empty images for any individual step that fails to process properly
    image_cache = []

    processed_result = None
    cell_mode = "P"
    cell_size = (1,1)

    state.job_count = len(xs) * len(ys) * p.n_iter

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"
            z = zs[iy]
            processed:Processed = cell(x, y, z)
            try:
                # this dereference will throw an exception if the image was not processed
                # (this happens in cases such as if the user stops the process from the UI)
                processed_image = processed.images[0]
                
                if processed_result is None:
                    # Use our first valid processed result as a template container to hold our full results
                    processed_result = copy(processed)
                    cell_mode = processed_image.mode
                    cell_size = processed_image.size
                    processed_result.images = [Image.new(cell_mode, cell_size)]

                image_cache.append(processed_image)
                if include_lone_images:
                    processed_result.images.append(processed_image)
                    processed_result.all_prompts.append(processed.prompt)
                    processed_result.all_seeds.append(processed.seed)
                    processed_result.infotexts.append(processed.infotexts[0])
            except:
                image_cache.append(Image.new(cell_mode, cell_size))

    if not processed_result:
        print("Unexpected error: draw_xy_grid failed to return even a single processed image")
        return Processed()

    grid = images.image_grid(image_cache, rows=len(ys))
    if draw_legend:
        grid = images.draw_grid_annotations(grid, cell_size[0], cell_size[1], hor_texts, ver_texts)

    processed_result.images[0] = grid

    return processed_result


class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.hypernetwork = opts.sd_hypernetwork
        self.model = shared.sd_model

    def __exit__(self, exc_type, exc_value, tb):
        modules.sd_models.reload_model_weights(self.model)

        hypernetwork.load_hypernetwork(self.hypernetwork)
        hypernetwork.apply_strength()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")

class Script(scripts.Script):
    def title(self):
        return "Generate Model Grid"

    def ui(self, is_img2img):
        filenames = []
        z_valuez = ''
        dirpath = os.path.join(models_path, 'Stable-diffusion')
        for path in os.listdir(dirpath):
            if path.endswith('.ckpt') or path.endswith('.safetensors'):
                filenames.append(path)
            else:
                if os.path.isdir(os.path.join(dirpath,path)):
                  for subpath in os.listdir(os.path.join(dirpath,path)):
                    if subpath.endswith('.ckpt') or path.endswith('.safetensors'):
                      filenames.append(subpath)
            
        filenames.append('model.ckpt')
        
        with gr.Row():
            x_values = gr.Textbox(label="Prompts, separated with &", lines=1)

        with gr.Row():
            y_values = gr.CheckboxGroup(filenames, label="Checkpoint file names, including file ending", lines=1)
        
        with gr.Row():
            z_values = gr.Textbox(label="Model tokens", lines=1)

        draw_legend = gr.Checkbox(label='Draw legend', value=True)
        include_lone_images = gr.Checkbox(label='Include Separate Images', value=False)
        no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False)

        return [x_values, y_values, z_values, draw_legend, include_lone_images, no_fixed_seeds]

    def run(self, p, x_values, y_values, z_values, draw_legend, include_lone_images, no_fixed_seeds):
        y_values = ','.join(y_values)
        if not no_fixed_seeds:
            modules.processing.fix_seed(p)

        if not opts.return_grid:
            p.batch_size = 1

        xs = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(x_values), delimiter='&'))]
        ys = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(y_values)))]
        zs = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(z_values)))]

        def cell(x, y, z):
            pc = copy(p)
            edit_prompt(pc, x, z)
            confirm_checkpoints(pc,ys)
            apply_checkpoint(pc, y, ys)

            return process_images(pc)

        with SharedSettingsStackHelper():
            processed = draw_xy_grid(
                p,
                xs=xs,
                ys=ys,
                zs=zs,
                x_labels=xs,
                y_labels=ys,
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images
            )

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "xy_grid", prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        return processed
