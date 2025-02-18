import argparse
import re
from io import BytesIO
import os, os.path as osp
import json
from tqdm import tqdm
import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def image_parser(image_file, sep=","):
    return image_file.split(sep)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", image_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def generate_prompt(vidsitu_visual_prompt, verb, roles, top_5):
    # verb = event_info["Verb"]
    role_list = ["Arg0", "Arg1", "Arg2", "Scene of the Event"]
    filtered_roles = [role for role in roles if any(role.startswith(r) for r in role_list)]

    if top_5:
        roles_json = ",\n    ".join([f'"{role}": ["", "", "", "", ""]' for role in filtered_roles])
    else:
        roles_json = ",\n    ".join([f'"{role}": [""]' for role in filtered_roles])

    return vidsitu_visual_prompt.format(verb=verb, roles_json=roles_json)


def eval_model(model, tokenizer, image_processor, args, images, prompt):
    qs = prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[images_tensor],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def main(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    # Load training data
    with open(args.data_file, 'r') as f:
        train_data = json.load(f)

    # Load checkpoint if exists
    if osp.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Read the vidsitu_visual_prompt from file
    with open(args.vidsitu_visual_prompt_file, 'r') as f:
        vidsitu_visual_prompt = f.read()

    example_images = image_parser(args.example_images, args.sep)
    #print(example_images)
    for clip_info in tqdm(train_data):
    # for i in range(0, len(train_data),3):
    #     r = random.randint(0,2)
    #     clip_info = train_data[i+r]
        clip_name = clip_info["Ev1"]["vid_seg_int"]
        if clip_name in results:
            continue  # Skip already processed clips

        clip_results = {}
        for event_index, event_key in enumerate(["Ev1", "Ev2", "Ev3", "Ev4", "Ev5"]):
            event_info = clip_info[event_key]
            event_results = []

            if event_index == 0:
                frame_indices = range(3)
            else:
                frame_indices = range(3 + (event_index - 1) * 2, 3 + event_index * 2)

            for frame_index in frame_indices:
                verb = event_info["Verb"]
                roles = event_info['Args'].keys()
                frame_path = os.path.join(args.image_dir, f"{clip_name}", f"frame_{frame_index:02}.jpg")
                #verb = valid[0]['Ev1']['Verb']
                
                if not os.path.exists(frame_path):
                    print("The clip frame dir does not exist.")
                    continue
                prompt = generate_prompt(vidsitu_visual_prompt, verb, roles, args.top_5)

                # Combine example images with the current frame
                current_images = example_images + [frame_path]
                images = load_images(current_images)

                output = eval_model(model, tokenizer, image_processor, args, images, prompt)
                #print(output)
                try:
                    parsed_output = json.loads(output)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {clip_name} event {event_key} frame {frame_index}: {e}")
                    parsed_output = None

                event_results.append({f"frame_{frame_index:02}": parsed_output})

            clip_results[event_key] = event_results

        results[clip_name] = clip_results

        # Save checkpoint after each clip
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file")
    parser.add_argument("--vidsitu_visual_prompt_file", type=str, required=True, help="Path to the file containing the template for the visual prompt")
    parser.add_argument("--example_images", type=str, required=True, help="Comma-separated list of example images")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_5", action='store_true', help="Generate top 5 nouns for each role if set; otherwise, generate only top 1 noun")
    args = parser.parse_args()

    main(args)