import json
import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


class DataSetEssay(Dataset):

    def __init__(self, tokenizer: AutoTokenizer, path_json: str, max_length: int):
        self._essay_text = []
        self._essay_grade = []
        with open(path_json, "r") as file:
            json_list = json.load(file)

        for example in json_list:
            text = example["filename"] + " " + example["text"]
            inputs_ids = torch.tensor(tokenizer.encode(text, max_length=max_length, truncation=True))
            grade = torch.tensor(float(example["grade"]))

            self._essay_text.append(inputs_ids)
            self._essay_grade.append(grade)
        self._essay_text = pad_sequence(self._essay_text, batch_first=True, padding_value=0)

    def __len__(self):
        return len(self._essay_text)

    def __getitem__(self, idx):
        return {"text": self._essay_text[idx], "grade": self._essay_grade[idx]}


class EssayGrade(nn.Module):

    def __init__(self, name_model: str, embedding_size: int):
        super(EssayGrade, self).__init__()
        self._bert = AutoModel.from_pretrained(name_model)
        self._lstm = nn.LSTM(input_size=embedding_size, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.2,
                             batch_first=True)
        self._fc = nn.Linear(64 * 2, 100)
        self._relu = nn.LeakyReLU()
        self._dropout = nn.Dropout(0.2)
        self._output = nn.Linear(100, 1)

        for param in self._bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        x = self._bert(input_ids)[0]

        x, hidden = self._lstm(x)
        x = x[:, -1, :]  # last_time_step
        x = self._fc(x)
        x = self._relu(x)
        x = self._dropout(x)
        x = self._output(x)
        x = x.reshape(-1)

        return x


def predict_loss(essay_dev: json, model, bert_tokenizer, max_length, device):
    criterion = nn.MSELoss()
    essay_grade = []
    essay_predict = []

    for essay in essay_dev:
        text = essay["filename"] + " " + essay["text"]
        inputs_ids = torch.tensor(bert_tokenizer.encode(text, max_length=max_length, truncation=True, padding=True)).unsqueeze(
            0)
        inputs_ids = inputs_ids.to(device)

        with torch.no_grad():
            predict = model(inputs_ids)

        grade_predict = predict.cpu().numpy()[0]
        essay_grade.append(float(essay["grade"]))
        essay_predict.append(grade_predict)

    essay_grade = torch.tensor(essay_grade).to(device)
    essay_predict = torch.tensor(essay_predict).to(device)
    with torch.no_grad():
        loss = criterion(essay_predict, essay_grade)

    return loss.item()


def predict_grades(path_model: str, bert_model: str, path_dataset: str, path_output: str, max_length: int,):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(path_model)
    model.eval()
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = model.to(device)

    with open(path_dataset, "r") as file:
        test_list = json.load(file)

        for test in test_list:
            text = test["filename"] + " " + test["text"]
            inputs_ids = torch.tensor(
                bert_tokenizer.encode(text, max_length=max_length, truncation=True, padding=True)).unsqueeze(0)
            inputs_ids = inputs_ids.to(device)

            with torch.no_grad():
                predit = model(inputs_ids)
            test["grade"] = str(predit.cpu().numpy()[0])

    with open(path_output, "w+") as file:
        json.dump(test_list, file, indent=4)


def train_model(name_model_bert: str, embedding_size: int, max_length: int, path_dataset: str, path_dataset_dev: str,
                path_save_model: str):
    print(path_save_model)
    predict_stop = [sys.maxsize, sys.maxsize]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bert_tokenizer = AutoTokenizer.from_pretrained(name_model_bert)

    dataset_train = DataSetEssay(bert_tokenizer, path_dataset, max_length)
    with open(path_dataset_dev, 'r') as file_read:
        json_dev = json.load(file_read)

    load_dataset = DataLoader(dataset_train, batch_size=5 * 7, shuffle=True, drop_last=True)
    model = EssayGrade(name_model_bert, embedding_size)
    model = model.to(device)

    epochs = 100  # max number of epochs to train
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0

        print("Start epoch: {}".format(epoch))
        for batch_no, item in enumerate(load_dataset):
            text_input = item["text"].to(device)
            grade = item["grade"].to(device)

            optimizer.zero_grad()

            output = model(text_input)

            loss = criterion(output, grade)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_no % 9 == 0:
                print("Batch {} loss: {}".format(batch_no, loss.item()), )
        loss_batch = running_loss / (batch_no + 1)
        dev_loss = predict_loss(json_dev, model, bert_tokenizer, max_length, device)

        print("Epoch: {}, Loss Train: {}, Loss Eval: {}\n".format(epoch, loss_batch, dev_loss))

        if dev_loss <= predict_stop[0] and dev_loss <= predict_stop[1]:
            torch.save(model, "{}-best.pt".format(path_save_model))

        if dev_loss >= predict_stop[0] and dev_loss >= predict_stop[1]:
            break

        predict_stop[0] = predict_stop[1]
        predict_stop[1] = dev_loss
    print("{}".format(path_save_model))
    torch.save(model, "{}.pt".format(path_save_model))


if __name__ == '__main__':

    train_model("../model/bert-pretrained", 256, 32, "../dataset/referate-train.json", "../dataset/referate-dev.json",
                "../model/save-model/bert")