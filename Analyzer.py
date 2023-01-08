import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
import re
from quickspacer import Spacer

device_kind = ""

domain = [
    "Happy",
    "Fear",
    "Embarrassed",
    "Sad",
    "Rage",
    "Hurt",
]

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
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        for logits in out:
            logits = logits.detach().cpu().numpy()
            # print(logits)
            print("Max : ",np.argmax(logits))


    print("Result : ", max(logits))
    return [logits, domain[np.argmax(logits)], logits[np.argmax(logits)] ]

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
    # [array([ 5.8531494, -0.6694658, -1.1817917, -0.908142 , -1.2727499, -1.5652826], dtype=float32), 'Happy', 5.8531494]
    print(test.analyze_word("아무런기대도 하지 않았지만 생각보다 괜찮은 결과였다"))
    print(test.analyze_word("이게 맞아?? 이런 방식은 좀 아닌거 같아"))
    print(test.analyze_word("으...의도는 좋은거 같지만 좀 망한거 같은디??"))
    print(test.analyze_word("할 일이 너무 많네요😅 할 일은 항상 끝이 없….🫠"))
    print(test.analyze_word("비둘기, 라이너 릴케 소학교 불러 다하지 봄이 슬퍼하는 봅니다. 한 이제 벌레는 북간도에 까닭입니다. 시와 하나에 차 이름을 나는 묻힌 딴은 봅니다. 이름과, 불러 우는 다하지 어머니, 북간도에 거외다. 별 추억과 멀듯이, 토끼, 아름다운 있습니다. 이름과 많은 헤는 어머님, 때 이런 피어나듯이 아침이 속의 듯합니다. 별 같이 강아지, 별을 별빛이 걱정도 별 당신은 있습니다. 무성할 어머님, 같이 밤을 프랑시스 피어나듯이 비둘기, 이름을 봅니다. 까닭이요, 소녀들의 불러 동경과 이웃 있습니다. 이름과 나의 별 나의 이름자 있습니다. 쉬이 못 하나에 마디씩 별에도 아직 내일 버리었습니다."))

# Happy '기쁨' = 0                    
# Fear '불안' = 1                   
# Embarrassed '당황' = 2                    
# Sad '슬픔' = 3                    
# Rage '분노' = 4                    
# Hurt '상처' = 5  