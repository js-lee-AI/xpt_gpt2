import os
from tqdm import tqdm

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Model,
    GPT2PreTrainedModel,
    get_scheduler,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import  CausalLMOutputWithCrossAttentions

import torch
from torch.utils.data import DataLoader
import pickle
from utils import chunk_dataset, GPT2_tokenizing
from collections import OrderedDict

import os
import argparse
import math

class XPTGPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, new_vocab_config):
        super().__init__(config)

        # self.config = GPT2Config.from_pretrained("gpt2")
        self.config = config
        self.config.vocab_size = new_vocab_config.vocab_size  # kobart tokenizer
        self.gpt2_body = GPT2Model.from_pretrained(self.config._name_or_path)

        self.input_itls = self.build_itl(self.config)
        self.output_itls = self.build_itl(self.config)

        # self.embedding_size = self.gpt2_body.get_input_embeddings().weight.size(1) # same as self.config.n_embd
        self.new_embedding = torch.nn.Embedding(
            self.config.vocab_size+1, self.config.n_embd  # 51201 x 768
        )  # kobart tokenizer

        self.gpt2_body.set_input_embeddings(
            self.new_embedding
        )  # change input embed layer

        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )  # 768 x vocab_size

        # tie
        self.post_init()

    # def forward(self, input_ids=None, attention_mask=None, labels=None):
    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        embedding_out = self.new_embedding(input_ids)
        out = self.input_itls(embedding_out)[0]

        out = self.gpt2_body(
            inputs_embeds=out,
            attention_mask=attention_mask,
        ).last_hidden_state  # batch_size * sequence_length * embed_size

        out = self.output_itls(out)[0]
        logits = self.lm_head(out)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        # return {"loss": loss, "logits": logits}
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def build_itl(self, config):
        layer = GPT2Block(config)
        layer.apply(self._init_weights)

        return layer


def _freeze_pahase_1(model):
    for name, param in model.named_parameters():
        if "gpt2_body" in name and "gpt2_body.wte" not in name:
            param.require_grad = False


def _unfreeze_pahase_2(model):
    for name, param in model.named_parameters():
        param.require_grad = True


def tokenize_and_binarize_dataset(dataset, tokenizer, max_length, dataset_path, n=3):
    dataset_iterator = chunk_dataset(dataset, n=n)
    print("start tokenization...")

    input_ids_list = []
    for chunk_data in tqdm(dataset_iterator, total=n+1):
        # tokenized_output = GPT2_tokenizing(
        #     dataset=chunk_data, tokenizer=tokenizer, max_length=max_length
        # )
        input_ids_list.append(GPT2_tokenizing(
            dataset=chunk_data, tokenizer=tokenizer, max_length=max_length
        ))
    input_ids_tensor = torch.cat([input_ids for input_ids in input_ids_list], dim=0).contiguous()
    print(input_ids_tensor.size())
        # with open(dataset_path, "ab") as f:
        #     pickle.dump(tokenized_output, f)
    with open(dataset_path, "wb") as f:
        pickle.dump(input_ids_tensor, f)

    return input_ids_tensor


def check_exist_pickle_file(dataset_path):

    return os.path.isfile(dataset_path)


def load_pickle_file(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


if __name__ == "__main__":

    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cache_dir="./cache_dir",
    )

    dataset_path = "./dataset/tokenized_dataset.pickle"
    if check_exist_pickle_file(dataset_path):
        tokenized_output = load_pickle_file(dataset_path)
    else:
        dataset = str()
        with open(
                "/mnt/raid6/omanma1928/projects/mrst2/dataset/train-valid-test/nikl_extractive.txt",
                "r",
        ) as f:
            lines = f.readlines()
            for txt in lines:
                if txt != "\n":
                    dataset += txt.replace("\n", tokenizer.eos_token)
        print(f'total characters in your dataset:  {len(dataset)}')

        tokenized_output = tokenize_and_binarize_dataset(
            dataset=dataset[:len(dataset)//4], # 일단 1/4만 사용
            tokenizer=tokenizer,
            max_length=1024,
            dataset_path=dataset_path,
            n=30,
        )
        del dataset

    import wandb
    wandb.init(name="XPT_gpt2", project="Multilingual_transfer")

    # train
    os.environ["CUDA_VISIBLE_DEVICES"] = "3, 6"
    trained_ckpt_path = False  # './ckpt/ph1_5000_step_ckpt'
    # trained_ckpt_path = './ckpt/ph1_5000_step_ckpt'
    if trained_ckpt_path:
        model = XPTGPT2LMHeadModel.from_pretrained(trained_ckpt_path, tokenizer)
    else:
        model = XPTGPT2LMHeadModel.from_pretrained("gpt2", tokenizer)

    model = torch.nn.DataParallel(model, output_device=1)
    device = torch.cuda.current_device()
    model = model.to(device)

    # training config
    num_phase1_training_steps = 90000  # 1000000
    num_phase2_training_steps = 90000  # 1000000
    total_eval_batch_size = 1000 # 1000
    per_eval_steps = 1000
    per_step_save_ckpt = 10000 # 10000
    log_interval = 1
    train_batch_size = 16
    eval_batch_size = 16

    print(f'number of total tokens: {torch.numel(tokenized_output)}')

    print(f'size of training tensor: {tokenized_output[:-total_eval_batch_size].size()}')
    print(f'size of eval tensor: {tokenized_output[-total_eval_batch_size:].size()}')

    train_data_loader = DataLoader(
        tokenized_output[:-total_eval_batch_size],
        batch_size=train_batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    eval_data_loader = DataLoader(
        tokenized_output[-total_eval_batch_size:],
        batch_size=eval_batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=3e-7
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_phase1_training_steps,
    )

    # phase 1
    progress_bar = tqdm(range(num_phase1_training_steps))
    current_steps = 0

    _freeze_pahase_1(model) # freeze body except new embedding layer
    while True:
        if current_steps >= num_phase1_training_steps:
            break

        for train_ids in train_data_loader:
            if current_steps >= num_phase1_training_steps:
                break

            model.train()
            loss = model(
                input_ids=train_ids.to(device),
                labels=train_ids.to(device)
            ).loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            current_steps += 1
            if current_steps % log_interval == 0:
                ppl = torch.exp(loss)
                progress_bar.set_description("step %d | loss %.04f  ppl %.02f  lr %10.2e " % (current_steps, loss, ppl, lr_scheduler.get_last_lr()[0]))
                wandb.log(
                    data={"phase1/train_loss": loss, "phase1/train_ppl": ppl, "phase1/train_lr": lr_scheduler.get_last_lr()[0]},
                    step=current_steps,
                )
            progress_bar.update(1)

            # evaluation
            if current_steps % per_eval_steps == 0:
                print("evaluation...")
                eval_losses = []
                model.eval()
                for eval_ids in tqdm(eval_data_loader):
                    with torch.no_grad():
                        eval_per_step_loss = model(
                            input_ids=eval_ids.to(device),
                            labels=eval_ids.to(device)
                        ).loss.mean().item()
                    eval_losses.append(eval_per_step_loss)
                eval_loss = sum(eval_losses) / len(eval_losses)

                eval_ppl = math.exp(eval_loss)
                wandb.log(
                    data={"phase1/eval_loss": eval_loss, "phase1/eval_ppl": eval_ppl},
                    step=current_steps,
                )

            if current_steps % per_step_save_ckpt == 0:
                # model.save_pretrained(f'./ckpt/ph1_{current_steps}_step_ckpt')
                os.makedirs(f"./ckpt/ph1_{current_steps}_step_ckpt", exist_ok=True)
                # torch.save(model.state_dict(), f"./ckpt/ph1_{current_steps}_step_ckpt/model.pt")
                model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(f'./ckpt/ph1_{current_steps}_step_ckpt')
            # tokenizer.save_pretrained(f"./ckpt")

    # phase 2
    progress_bar = tqdm(range(num_phase2_training_steps))
    total_training_steps = current_steps + num_phase2_training_steps

    train_data_loader = DataLoader(
        tokenized_output[:-total_eval_batch_size],
        batch_size=train_batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    eval_data_loader = DataLoader(
        tokenized_output[-total_eval_batch_size:],
        batch_size=eval_batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    print("...start phase 2")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=3e-7
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_phase2_training_steps,
    )

    _unfreeze_pahase_2(model) # unfreeze
    while True:
        if current_steps >= num_phase1_training_steps:
            break

        for train_ids in train_data_loader:
            if current_steps >= total_training_steps:
                break

            model.train()
            loss = model(
                input_ids=train_ids.to(device),
                labels=train_ids.to(device)
            ).loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            current_steps += 1
            if current_steps % log_interval == 0:
                ppl = torch.exp(loss)
                progress_bar.set_description("step %d | loss %.04f  ppl %.02f  lr %10.2e " % (current_steps, loss, ppl, lr_scheduler.get_last_lr()[0]))
                wandb.log(
                    data={"phase2/loss": loss, "phase2/ppl": ppl, "phase2/lr": lr_scheduler.get_last_lr()[0]},
                    step=current_steps,
                )
            progress_bar.update(1)

            # evaluation
            if current_steps % per_eval_steps == 0:
                print("evaluation...")
                eval_losses = []
                model.eval()
                for eval_ids in tqdm(eval_data_loader):
                    with torch.no_grad():
                        eval_per_step_loss = model(
                            input_ids=eval_ids.to(device),
                            labels=eval_ids.to(device)
                        ).loss.mean().item()
                    eval_losses.append(eval_per_step_loss)
                eval_loss = sum(eval_losses) / len(eval_losses)
                eval_ppl = math.exp(eval_loss)
                wandb.log(
                    data={"phase2/eval_loss": eval_loss, "phase2/eval_ppl": eval_ppl},
                    step=current_steps,
                )

            if current_steps % per_step_save_ckpt == 0:
                # model.save_pretrained(f'./ckpt/ph2_{current_steps}_step_ckpt')
                os.makedirs(f"./ckpt/ph2_{current_steps}_step_ckpt", exist_ok=True)
                # torch.save(model, f"./ckpt/ph2_{current_steps}_step_ckpt")
                model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(f'./ckpt/ph2_{current_steps}_step_ckpt')
                # tokenizer.save_pretrained(f"./ckpt/ph1_{current_steps}_step_ckpt")

    model_to_save.save_pretrained(f'./ckpt/ph2_{current_steps}_step_ckpt')
