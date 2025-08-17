import torch
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import os
from transformers import Trainer
from Dataset import EHRGeneratorDataset


IGNORE_INDEX = -100
SEP = '<SEP>'

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    train_root_path: str = field(default='', metadata={"help": "Path to the training data."})
    eval_root_path: str = field(default='', metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels",'attention_mask'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = EHRGeneratorDataset(tokenizer=tokenizer, data_path=data_args.train_root_path, ref_range_json_path="reference_range.json", use_ref_range_posibility=0) # for this version, we haven't use ref range.
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.eval_root_path:
        eval_dataset = EHRGeneratorDataset(tokenizer=tokenizer, data_path=data_args.eval_root_path,  ref_range_json_path="reference_range.json", use_ref_range_posibility=1)
        return dict(train_dataset=train_dataset, eval_dataset = eval_dataset, data_collator=data_collator)
    return dict(train_dataset=train_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(training_args.output_dir)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    
    # 添加special tokens
    special_tokens = [SEP]
    new_tokens = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)
    
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {len(new_tokens)} new special tokens: {new_tokens}")
        print(f"Model embedding size resized to: {len(tokenizer)}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
