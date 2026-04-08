import copy
import random
import re
from typing import Dict, Sequence

import logging
import torch
import transformers

from .utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    EXPLANATORY_QUESTION_LIST,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    TASK_IMAGE_SINGLE_SEG, TASK_VIDEO_SINGLE_SEG, TASK_IMAGE_MULTI_SEG, TASK_VIDEO_MULTI_SEG, TASK_IMAGE_TEXT_ONLY, TASK_VIDEO_TEXT_ONLY,
    UNIFIED_LONG_QUESTION_LIST,
    UNIFIED_SHORT_QUESTION_LIST,
    VIRST_ANSWER_LIST,
)
from utils import conversation as conversation_lib
from utils.argument import DataArguments

def get_qa_pair(
    expression,
    image_len: int = 0,
    has_image: bool = False,
    seg_token_num = 10,
    modality = "image",
):
    long_question_list = UNIFIED_LONG_QUESTION_LIST
    short_question_list = UNIFIED_SHORT_QUESTION_LIST
    answer_list = VIRST_ANSWER_LIST

    if expression[-1] == "?":
        question = random.choice(long_question_list).format(sent=expression, modality=modality)
    else:
        question = random.choice(short_question_list).format(sent=expression, modality=modality)

    seg_replace = ", ".join(f'({i}) [SEG]' for i in range(seg_token_num))
    answer = random.choice(answer_list).format(seg=seg_replace)

    return question, answer 
def preprocess_virst(
    expression, 
    tokenizer: transformers.PreTrainedTokenizer, 
    has_image: bool = False, 
    max_len=2048, 
    system_message: str = "You are a helpful assistant.",
    seg_token_num = 10,
    modality = "image",
    add_explanation = None, # extra explanation for the image
    only_text = False,
    task_prompt = TASK_VIDEO_MULTI_SEG,
    only_question = False, # for inference
) -> Dict:
    if task_prompt not in [TASK_IMAGE_SINGLE_SEG, TASK_VIDEO_SINGLE_SEG, TASK_IMAGE_MULTI_SEG, TASK_VIDEO_MULTI_SEG, TASK_IMAGE_TEXT_ONLY, TASK_VIDEO_TEXT_ONLY]:
        raise ValueError(f"Invalid task prompt: {task_prompt}")

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[0:2] # for qwen2_5
    
    unmask_tokens_idx =  [198, im_start, im_end]

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []

    question_auto, answer = get_qa_pair(expression, has_image=has_image, seg_token_num=10, modality=modality)
    
    if add_explanation is not None:
        if only_text: # text-only answer case 
            if has_image:
                question = f"Reference {modality.capitalize()}: " + DEFAULT_IMAGE_TOKEN + expression.lower()
            else:
                question = expression.lower()
            
            answer = add_explanation
        else:
            question = random.choice(EXPLANATORY_QUESTION_LIST).format(sent=expression.lower(), modality=modality)
            answer = answer + " " + add_explanation
    else:
        question = question_auto
    
    if not only_question:
        conv = [   
            {"role" : "system", "content" : system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    else:
        conv = [   
            {"role" : "system", "content" : system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": ""}
        ]

    encode_id = tokenizer.apply_chat_template(conv, tokenize=True, add_generation_prompt=False)
    input_ids += encode_id 
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    
    targets = input_ids.clone()
    
    cur = 0
    for msg in conv:
        encode_id = tokenizer.apply_chat_template([msg], tokenize=True, add_generation_prompt=False)
        length = len(encode_id)
        
        if msg["role"] in ["user", "system"]:
            targets[cur: cur + length] = IGNORE_INDEX
        cur += length 

    assert len(input_ids) == len(targets), f"{len(input_ids)} != {len(targets)}"
    
    for idx, encode_id in enumerate(input_ids):
        if encode_id in unmask_tokens_idx:
            targets[idx] = encode_id
        if encode_id == image_token_index:
            input_ids[idx] = IMAGE_TOKEN_INDEX
    
    del tokenizer
    return dict(
        input_ids=input_ids,  # tensor(1 x seq_len)
        labels=targets,  # tensor(1 x seq_len)
        conv=conv,
        question=expression
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[0:2] # for qwen2_5
    unmask_tokens_idx =  [198, im_start, im_end]

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []

    for i, source in enumerate(sources):
        

        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    del tokenizer
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """


    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments, msg="") -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

            if msg.rstrip() != "":
                replace_token = replace_token + msg.rstrip() + " "
            
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources
