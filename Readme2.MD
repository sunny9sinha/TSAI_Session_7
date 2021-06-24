Task was to train model we wrote in the class on the following two datasets taken from https://kili-technology.com/blog/chatbot-training-datasets/
1) http://www.cs.cmu.edu/~ark/QA-data/ 
2) https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

These are Natural languange translation kind of task and we have following struture for both task.

download spacy:
```python
python -m spacy download en
````
Load spacy:
```python
spacy_en = spacy.load('en_core_web_sm')
````
Define the tokenize function:
```python

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
````
In CMU dataset task is to generate answers for questions given as input.
For CMU dataset which is a question/answer dataset we defined fields like:

```python
Question = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

Answer = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
````
```python
s08_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cmuQA/Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt", sep='\t',encoding='ISO-8859-1')
s09_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cmuQA/Question_Answer_Dataset_v1.2/S09/question_answer_pairs.txt", sep='\t',encoding='ISO-8859-1')
s10_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cmuQA/Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt", sep='\t', quoting=3,encoding='ISO-8859-1')

s08_data = s08_data.drop(columns=['ArticleTitle','DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile'])
s09_data = s09_data.drop(columns=['ArticleTitle','DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile'])
s10_data = s10_data.drop(columns=['ArticleTitle','DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile'])

QA_data = pd.concat([s08_data,s09_data,s10_data], axis =0)
````
After loading data we converted question and answer columns to string type and changed case to small:

```python
def smallCase(data):
    data['Question'] = data['Question'].str.lower()
    data['Answer'] = data['Answer'].str.lower()
    return data

QA_data.Question = np.array(QA_data.Question).astype(str)
QA_data.Answer = np.array(QA_data.Answer).astype(str)
QA_data = smallCase(pd.DataFrame(QA_data))

print(QA_data.head())

Question Answer
0  was abraham lincoln the sixteenth president of...    yes
1  was abraham lincoln the sixteenth president of...   yes.
2  did lincoln sign the national banking act of 1...    yes
3  did lincoln sign the national banking act of 1...   yes.
4                   did his mother die of pneumonia?     no

````
Using SKLearn we divided data into train, test and validation dataset:

```python

from sklearn.model_selection import train_test_split

train, test_data = train_test_split(QA_data, test_size=0.3)
train = train.reset_index().drop(columns=['index'])
test_data = test_data.reset_index().drop(columns=['index'])
train_data, valid_data = train_test_split(train, test_size=0.25)
train_data = train_data.reset_index().drop(columns=['index'])
valid_data = valid_data.reset_index().drop(columns=['index'])
````
```python
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

(2098, 2)
(700, 2)
(1200, 2)
````
With following parameters we trained the model:
```python
INPUT_DIM = len(Question.vocab)
OUTPUT_DIM = len(Answer.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
````
Training Log:
```python
Epoch: 01 | Time: 0m 2s
	Train Loss: 5.929 | Train PPL: 375.834
	 Val. Loss: 5.238 |  Val. PPL: 188.334
Epoch: 02 | Time: 0m 2s
	Train Loss: 5.083 | Train PPL: 161.283
	 Val. Loss: 5.365 |  Val. PPL: 213.806
Epoch: 03 | Time: 0m 2s
	Train Loss: 4.963 | Train PPL: 143.016
	 Val. Loss: 5.439 |  Val. PPL: 230.202
Epoch: 04 | Time: 0m 2s
	Train Loss: 4.870 | Train PPL: 130.366
	 Val. Loss: 5.497 |  Val. PPL: 244.036
Epoch: 05 | Time: 0m 2s
	Train Loss: 4.783 | Train PPL: 119.492
	 Val. Loss: 5.579 |  Val. PPL: 264.817
Epoch: 06 | Time: 0m 2s
	Train Loss: 4.771 | Train PPL: 118.024
	 Val. Loss: 5.639 |  Val. PPL: 281.226
Epoch: 07 | Time: 0m 2s
	Train Loss: 4.640 | Train PPL: 103.537
	 Val. Loss: 5.662 |  Val. PPL: 287.791
Epoch: 08 | Time: 0m 2s
	Train Loss: 4.597 | Train PPL:  99.199
	 Val. Loss: 5.737 |  Val. PPL: 310.028
Epoch: 09 | Time: 0m 2s
	Train Loss: 4.550 | Train PPL:  94.644
	 Val. Loss: 5.805 |  Val. PPL: 331.919
Epoch: 10 | Time: 0m 2s
	Train Loss: 4.523 | Train PPL:  92.139
	 Val. Loss: 5.805 |  Val. PPL: 331.918
````
Test log:
```python
| Test Loss: 5.263 | Test PPL: 192.987 |
````

**************************************************************************************************************************************************************

For Quora dataset which contains question and duplicate question we have following steps after defining our tokenize_en function:
The task here is to generate a duplicate question for given input question.
Define field as:
```python
question1 = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

question2 = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
            
````
read data:
```python
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/quoraQA/quora_duplicate_questions.tsv", sep='\t')
````
We then filter the dataset and keep only those records where dupicate question is confirmed.
```python
data = data[data['is_duplicate'] == 1]
print(data.head())

    id  qid1  ...                                          question2 is_duplicate
5    5    11  ...  I'm a triple Capricorn (Sun, Moon and ascendan...            1
7    7    15  ...          What should I do to be a great geologist?            1
11  11    23  ...             How can I see all my Youtube comments?            1
12  12    25  ...            How can you make physics easy to learn?            1
13  13    27  ...             What was your first sexual experience?            1
````
We then process the data as follows:
```python
data = data.drop(columns=['id', 'qid1', 'qid2','is_duplicate'])

def smallCase(data):
    data['question1'] = data['question1'].str.lower()
    data['question2'] = data['question2'].str.lower()
    return data
    
data.question1 = np.array(data.question1).astype(str)
data.question2 = np.array(data.question2).astype(str)
data = smallCase(pd.DataFrame(data))    

````

we then divide the data as per following test, train and validation set:
```python
from sklearn.model_selection import train_test_split

train, test_data = train_test_split(data, test_size=0.3)
train = train.reset_index().drop(columns=['index'])
test_data = test_data.reset_index().drop(columns=['index'])
train_data, valid_data = train_test_split(train, test_size=0.25)
train_data = train_data.reset_index().drop(columns=['index'])
valid_data = valid_data.reset_index().drop(columns=['index'])
````
With following parameters we define the model:
```python
INPUT_DIM = len(question1.vocab)
OUTPUT_DIM = len(question2.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
````

Training Log:
```python

Epoch: 01 | Time: 5m 9s
	Train Loss: 5.221 | Train PPL: 185.147
	 Val. Loss: 5.228 |  Val. PPL: 186.498
Epoch: 02 | Time: 5m 9s
	Train Loss: 4.358 | Train PPL:  78.126
	 Val. Loss: 4.727 |  Val. PPL: 112.998
Epoch: 03 | Time: 5m 10s
	Train Loss: 3.826 | Train PPL:  45.856
	 Val. Loss: 4.440 |  Val. PPL:  84.811
Epoch: 04 | Time: 5m 10s
	Train Loss: 3.463 | Train PPL:  31.907
	 Val. Loss: 4.225 |  Val. PPL:  68.366
Epoch: 05 | Time: 5m 10s
	Train Loss: 3.202 | Train PPL:  24.593
	 Val. Loss: 4.155 |  Val. PPL:  63.738
Epoch: 06 | Time: 5m 11s
	Train Loss: 2.980 | Train PPL:  19.691
	 Val. Loss: 4.085 |  Val. PPL:  59.412
Epoch: 07 | Time: 5m 11s
	Train Loss: 2.817 | Train PPL:  16.725
	 Val. Loss: 4.001 |  Val. PPL:  54.656
Epoch: 08 | Time: 5m 10s
	Train Loss: 2.672 | Train PPL:  14.467
	 Val. Loss: 4.005 |  Val. PPL:  54.847
Epoch: 09 | Time: 5m 10s
	Train Loss: 2.560 | Train PPL:  12.935
	 Val. Loss: 3.973 |  Val. PPL:  53.169
Epoch: 10 | Time: 5m 10s
	Train Loss: 2.434 | Train PPL:  11.403
	 Val. Loss: 3.979 |  Val. PPL:  53.480
````
Test log:
```python
| Test Loss: 3.995 | Test PPL:  54.317 |
````
