import json
import copy
import torch
from typing import Dict, List
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def load_instruction(path: str) -> str:
    """Load system instruction text from file."""
    with open(path, 'r', encoding='utf-8') as fp:
        return fp.read()


class MedicalMultiTurnDataset(Dataset):
    """
    Dataset for multi-turn medical conversations.
    Each conversation is split into multiple training samples,
    where each sample ends with an assistant's response.
    """

    def __init__(self, tokenizer, data_path: str, system_prompt_path: str):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.system_prompt = load_instruction(system_prompt_path)
        self.conversations = []

        print(f"Loading data from {data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as fp:
            raw_data = json.load(fp)

        # Split each case into multiple conversations
        for case in raw_data:
            conversations = self.split_into_conversations(case)
            self.conversations.extend(conversations)

        print(f"Total conversations after splitting: {len(self.conversations)}")

    def split_into_conversations(self, case: List[Dict]) -> List[List[Dict]]:
        """Split a full case into multiple conversation turns."""
        conversations = []
        current_messages = [{"role": "system", "content": self.system_prompt}]

        for i, message in enumerate(case):
            current_messages.append(message)

            # End of a turn when encountering an assistant message
            if message["role"] == "assistant":
                conversations.append(current_messages.copy())

                # Prepare for the next turn if not the last message
                if i < len(case) - 1:
                    current_messages = current_messages.copy()

        return conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        return self.preprocess(conversation)

    def preprocess(self, conversation: List[Dict]) -> Dict[str, torch.Tensor]:
        """Preprocess the conversation into tokenized format for training."""

        # Tokenize full conversation (including last assistant reply)
        formatted_example = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize conversation without last assistant reply
        train_conversation = conversation[:-1]
        formatted_source = self.tokenizer.apply_chat_template(
            train_conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        example_tokenized = self.tokenizer(
            formatted_example,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        source_tokenized = self.tokenizer(
            formatted_source,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = example_tokenized["input_ids"][0]
        attention_mask = example_tokenized["attention_mask"][0]
        source_input_ids = source_tokenized["input_ids"][0]

        # Create labels with IGNORE_INDEX for non-assistant response tokens
        labels = copy.deepcopy(input_ids)
        source_len = len(source_input_ids)
        labels[:source_len] = IGNORE_INDEX

        # Truncate if sequence exceeds max length
        if len(input_ids) > self.tokenizer.model_max_length:
            input_ids = input_ids[-self.tokenizer.model_max_length:]
            labels = labels[-self.tokenizer.model_max_length:]
            attention_mask = attention_mask[-self.tokenizer.model_max_length:]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )