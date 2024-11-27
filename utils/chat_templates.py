templates = {
    "chatml": {
        "system": "<|im_start|>system\n",
        "user": "<|im_start|>user\n",
        "assistant": "<|im_start|>assistant\n",
        "suffix": "<|im_end|>\n"
    },
    "llama3-instruct": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "suffix": "<|eot_id|>"
    },
    "gemma": {
        "system": "<start_of_turn>user\n",
        "user": "<start_of_turn>user\n",
        "assistant": "<start_of_turn>model\n",
        "suffix": "<end_of_turn>"
    },
}
    
def get_template(template):
    return templates[template]