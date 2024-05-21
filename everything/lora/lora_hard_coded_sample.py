import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Define LoRA configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q', 'v'],  # Specify the target modules for LoRA
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config).to(device)

# Hardcoded 16-sample training dataset
train_data = {
    'input_text': [
        'Translate English to Korean: The house is wonderful.',
        'Translate English to Korean: The book is on the table.',
        'Translate English to Korean: He is reading a book.',
        'Translate English to Korean: The cat is sleeping.',
        'Translate English to Korean: She is cooking dinner.',
        'Translate English to Korean: The car is fast.',
        'Translate English to Korean: The weather is nice today.',
        'Translate English to Korean: The boy is playing football.',
        'Translate English to Korean: I am learning Korean.',
        'Translate English to Korean: The flowers are beautiful.',
        'Translate English to Korean: He likes to travel.',
        'Translate English to Korean: She is a good teacher.',
        'Translate English to Korean: The sun is shining.',
        'Translate English to Korean: The cake tastes delicious.',
        'Translate English to Korean: They are watching a movie.',
        'Translate English to Korean: He is riding a bicycle.'
    ],
    'target_text': [
        '집이 멋져요.',
        '책이 테이블 위에 있어요.',
        '그는 책을 읽고 있어요.',
        '고양이가 자고 있어요.',
        '그녀는 저녁을 요리하고 있어요.',
        '차가 빠르다.',
        '오늘 날씨가 좋아요.',
        '소년이 축구를 하고 있어요.',
        '나는 한국어를 배우고 있어요.',
        '꽃들이 아름다워요.',
        '그는 여행하는 것을 좋아해요.',
        '그녀는 좋은 선생님이에요.',
        '태양이 빛나고 있어요.',
        '케이크가 맛있어요.',
        '그들은 영화를 보고 있어요.',
        '그는 자전거를 타고 있어요.'
    ]
}

# Hardcoded 4-sample validation dataset
valid_data = {
    'input_text': [
        'Translate English to Korean: The child is drawing a picture.',
        'Translate English to Korean: She is singing a song.',
        'Translate English to Korean: The stars are bright tonight.',
        'Translate English to Korean: He is drinking coffee.'
    ],
    'target_text': [
        '아이가 그림을 그리고 있어요.',
        '그녀는 노래를 부르고 있어요.',
        '오늘 밤 별이 밝아요.',
        '그는 커피를 마시고 있어요.'
    ]
}

# Create Datasets
train_dataset = Dataset.from_dict(train_data)
valid_dataset = Dataset.from_dict(valid_data)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'valid': valid_dataset
})

# Tokenize the datasets
def preprocess_function(examples):
    inputs = tokenizer(examples['input_text'], padding='max_length', truncation=True, max_length=36, return_tensors='pt')
    targets = tokenizer(examples['target_text'], padding='max_length', truncation=True, max_length=36, return_tensors='pt')
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    num_train_epochs=1000,
    logging_steps=10,
    save_steps=10,
    save_total_limit=1,
    evaluation_strategy='epoch',
    remove_unused_columns=False,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./lora-tuned-model')
tokenizer.save_pretrained('./lora-tuned-model')

# Load the trained model for inference
trained_model = AutoModelForSeq2SeqLM.from_pretrained('./lora-tuned-model').to(device)
trained_tokenizer = AutoTokenizer.from_pretrained('./lora-tuned-model')

# Function for generating predictions
def translate_text(input_text):
    inputs = trained_tokenizer(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = trained_model.generate(**inputs, max_length=36, num_beams=4, early_stopping=True)
    return trained_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example input
for i in range(16):
    sample_input = train_data['input_text'][i]
    translation = translate_text(sample_input)
    print("[Input]:", sample_input)
    print("[Translation]:", translation)
    print()
for i in range(4):
    sample_input = valid_data['input_text'][i]
    translation = translate_text(sample_input)
    print("[Input]:", sample_input)
    print("[Translation]:", translation)
    print()
# sample_input = "Translate English to Korean: The cat is on the roof."
# translation = translate_text(sample_input)
# print("[Input]:", sample_input)
# print("[Translation]:", translation)
