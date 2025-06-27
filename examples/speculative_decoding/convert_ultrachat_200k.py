from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k")
import json

with open("./finetune/ultrachat_200k.jsonl", "w") as f:
    for item in ds["train_sft"]:
        conversations = item['messages']
        new_conversations = []

        for message in conversations:
            new_role = message['role']
            content = message['content']
            new_conversations.append({
                "role": new_role,
                "content": content
            })

        row = {
            "id": item["prompt_id"],
            "conversations": new_conversations
        }
        
        f.write(json.dumps(row) + "\n")