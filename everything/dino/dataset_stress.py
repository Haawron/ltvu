from datasets import load_dataset

dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"].select(range(100_000))

import torch.utils.data as data
dl = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

from tqdm import tqdm
for batch in tqdm(dl):
	pass
