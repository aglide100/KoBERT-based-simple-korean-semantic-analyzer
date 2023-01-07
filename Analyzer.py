import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
# from hanspell import spell_checker
import re
import logging
from quickspacer import Spacer

device_kind = ""

if torch.cuda.is_available():    
    device = torch.device("cuda")
    device_kind="cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    device_kind="cpu"
    print('No GPU available, using the CPU instead.')

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
batch_size = 64
max_len = 64
tok = tokenizer.tokenize

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         
    def __len__(self):
        return (len(self.labels))

def predict(sentence):
    print("----------- \n전달받은 Sentence:")
    
    print(sentence)
    dataset = [[sentence, '0']]
    test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=2)
    model.eval()
    answer = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        for logits in out:
            logits = logits.detach().cpu().numpy()
            answer = np.argmax(logits)

    print("Result : ", answer)
    return answer


model = torch.load('model/SentimentAnalysisKOBert.pt', map_location=torch.device(device_kind))

def remove_unnecessary_word(text):
    text = re.sub('[/[\{\}\[\]\/?|\)*~`!\-_+<>@\#$%&\\\=\(\'\"]+', '', str(text))
    text = re.sub('[a-zA-Z]' , ' ', str(text))
    text = re.sub(' +', ' ', str(text))
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', str(text)) # http로 시작되는 url
    text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ', str(text)) # http로 시작되지 않는 url
    
    spacer = Spacer()
    # text = text.rstrip().lstrip()
    text.replace(" " , "")
    text = spacer.space([text])
    return text[0]

class Analyzer:
    def analyze_word(self, row):
        try:
            result = predict(remove_unnecessary_word(row))
        except:
            print("Get some err")
            return

        return result

if __name__ == "__main__":
    test = Analyzer()
    print(test.analyze_word("누가나한테우유를던졌어아주신선한우유를말이야근데이거진짜로진짜제대로잘돌아가는거맞냐는게바로나의질문이란말이지아니근데이건좀너무한거아닙니까?맞춤법좀잘맞춰서띄어쓰기좀해여"))

# '기쁨' = 0                    
# '불안' = 1                   
# '당황' = 2                    
# '슬픔' = 3                    
# '분노' = 4                    
# '상처' = 5  