import os
import torch
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from transformers import Trainer
from Dataset import MedicalMultiTurnDataset

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    """Arguments for dataset paths."""
    train_root_path: str = field(default='', metadata={"help": "Path to the training data."})
    eval_root_path: str = field(default='', metadata={"help": "Path to the evaluation data."})
    system_prompt_path: str = field(default='', metadata={"help": "Path to system prompt text."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for training configuration."""
    output_dir: str = field(default="")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for the model."},
    )


@dataclass
class DataCollatorForSupervisedDataset:
    """Data collator for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", 'attention_mask')
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    """Prepare dataset and data collator for training."""
    train_dataset = MedicalMultiTurnDataset(
        tokenizer=tokenizer,
        data_path=data_args.train_root_path,
        system_prompt_path=data_args.system_prompt_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)


def train():
    """Main training loop."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()