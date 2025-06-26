from datasets import load_dataset

ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
import json

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant"
} 

with open("./finetune/sharegpt.jsonl", "w") as f:
    for item in ds["train"]:
        conversations = item['conversations']
        new_conversations = []

        for message in conversations:
            new_role = ROLE_MAPPING[message['from']]
            content = message['value']
            new_conversations.append({
                "role": new_role,
                "content": content
            })

        row = {
            "id": item["id"],
            "conversations": new_conversations
        }
        
        f.write(json.dumps(row) + "\n")
