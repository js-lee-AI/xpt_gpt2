import torch
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
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class XPTGPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, new_vocab_config):
        super().__init__(config)

        # self.config = GPT2Config.from_pretrained("gpt2")
        self.config = config
        self.config.vocab_size = new_vocab_config.vocab_size
        self.gpt2_body = GPT2Model.from_pretrained(self.config._name_or_path)

        self.input_itls = self.build_itl(self.config)
        self.output_itls = self.build_itl(self.config)

        # self.embedding_size = self.gpt2_body.get_input_embeddings().weight.size(1) # same as self.config.n_embd
        self.new_embedding = torch.nn.Embedding(
            self.config.vocab_size, self.config.n_embd  # 30000 x 768
        )  # kobart tokenizer

        self.gpt2_body.set_input_embeddings(
            self.new_embedding
        )  # change input embed layer

        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )  # 768 x vocab_size

    def forward(
        self,
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

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

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


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cache_dir="cache_dir",
    )
    model = XPTGPT2LMHeadModel.from_pretrained('./ckpt/ph1_30000_step_ckpt', tokenizer)
    imput_prompt = "검찰 총장은"
    input_ids = tokenizer.encode(imput_prompt, return_tensors='pt')
    output_ids = model.generate(
        input_ids,
        num_beams=5,
        max_length=50,
        no_repeat_ngram_size=2,
        num_return_sequences=3,
        early_stopping=True,
        top_k=50,
    )
    print(output_ids)

    print(tokenizer.batch_decode(output_ids, skip_special_tokens=False))
