from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle


def GPT2_tokenizing(dataset, tokenizer, max_length=1024):
    output = tokenizer(
                dataset,
                return_tensors="pt",
                # max_length=max_length,
             ).input_ids[0]

    for_drop_tokens_by_max_length = max_length * (output.size()[0] // max_length)
    output = output[:for_drop_tokens_by_max_length] # drop_last
    output = output.view(-1, max_length).contiguous()

    return output


def chunk_dataset(dataset, n):
    for chunk_size in range(0, len(dataset), len(dataset)//n):
        yield dataset[chunk_size:chunk_size+len(dataset)//n]


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
                                              bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>', cache_dir='./cache_dir')

    dataset = str()
    with open('/mnt/raid6/omanma1928/projects/mrst2/dataset/train-valid-test/nikl_extractive.txt', 'r') as f:
        lines = f.readlines()
        for txt in lines:
            if txt != '\n':
                dataset += txt.replace('\n', tokenizer.eos_token)

    dataset_iterator = chunk_dataset(dataset, n=3)
    for chunk_data in tqdm(dataset_iterator):
        tokenized_output = GPT2_tokenizing(chunk_data, tokenizer=tokenizer, max_length=1024)


        with open('./dataset/tokenized_dataset.pickle', 'ab+') as f:
            pickle.dump(tokenized_output, f)
