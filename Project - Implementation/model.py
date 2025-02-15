import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import pandas as pd
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class FinetuneTagger:

    def __init__(
            self,
            train_file,
            val_file,
            basemodel='distilbert-base-uncased',
			epochs=3,  #20
            batch_size=64,  #20
            lr=5e-5
        ):

        self.train_ratio = 0.8
        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.token_max_len = 64
        self.trainfile = train_file
        self.valfile = val_file
        self.basemodel = basemodel
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model = DistilBertForSequenceClassification.from_pretrained(basemodel, num_labels = 2).to(device)           
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-8)
        self.load_data()   

    def load_data(self):
        self.training_data = pd.read_csv(self.trainfile).dropna()
        self.testing_data = self.training_data.head(int(len(self.training_data) * (1-self.train_ratio)))
        self.training_data = self.training_data.iloc[len(self.testing_data):]
        self.val_data = pd.read_csv(self.valfile).dropna()
         
    def gen_training_seq(self, data_list):
        sentence_list = self.tokenizer(data_list, is_split_into_words=False, truncation=True, padding='max_length', add_special_tokens=False, max_length=self.token_max_len)
        return sentence_list
    
    def get_training_data(self, start_ind, count):
        batch = []
        for i in range(start_ind, start_ind+count):
            batch.append(self.titles[start_ind+i])
        return batch
    
    def train(self):
        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            self.training_data = self.training_data.sample(frac=1).reset_index(drop=True)
            self.titles = self.training_data['title'].to_numpy()
            self.label = self.training_data['label'].to_numpy()
            print("current epoch: %d" % (epoch+1))
            with tqdm.tqdm(DataLoader([val for val in zip(self.titles, self.label)], batch_size=self.batch_size)) as train_iter:
                for titles, labels in train_iter:
                    output_batch = []
                    temp_batch = self.gen_training_seq(titles)
                    input_id_batch = temp_batch['input_ids']
                    input_attn_batch = temp_batch['attention_mask']
                    input_id_batch = torch.reshape(torch.LongTensor(input_id_batch).to(device), (len(input_id_batch), self.token_max_len))
                    input_attn_batch = torch.reshape(torch.LongTensor(input_attn_batch).to(device), (len(input_attn_batch), self.token_max_len))
                    output_batch = labels.to(device)
                    self.model.zero_grad()
                    loss = self.model(input_id_batch, attention_mask=input_attn_batch, labels=output_batch).loss

                    loss.backward()
                    self.optimizer.step()
                    train_iter.set_description("loss: %f" % loss)

    def get_acc(self, logits, labels):
        pred_label = np.argmax(logits, axis=1)
        return np.sum(pred_label == labels)

    def validate(self):
        self.model.eval()
        self.titles = self.val_data['title'].to_numpy()
        self.label = self.val_data['label'].to_numpy()
        total_acc = 0.0
        with tqdm.tqdm(DataLoader([val for val in zip(self.titles, self.label)], batch_size=self.batch_size)) as train_iter:
            for titles, labels in train_iter:
                temp_batch = self.gen_training_seq(titles)
                input_id_batch = temp_batch['input_ids']
                input_attn_batch = temp_batch['attention_mask']
                input_id_batch = torch.reshape(torch.LongTensor(input_id_batch).to(device), (len(input_id_batch), self.token_max_len))
                input_attn_batch = torch.reshape(torch.LongTensor(input_attn_batch).to(device), (len(input_attn_batch), self.token_max_len))

                logits = self.model(input_id_batch, attention_mask=input_attn_batch)
                logits = logits['logits'].detach().cpu().numpy()
                total_acc += self.get_acc(logits, labels.detach().cpu().numpy())
                
        print("validation acc: %f" % (total_acc/len(self.val_data)))

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print("saved model to " + save_path)

    def load(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        print("loaded model from " + save_path)

        
finetuner = FinetuneTagger('data/fake_news_training_set.csv', 'data/fake_news_validation_set.csv')
finetuner.load('model/fine_tuned_model.pt')
finetuner.validate()