# TSAI_Session_7
Part 1:
This assignement is about building an LSTM network for sentiment analysis on StanfordSentimentAnalysis Dataset. (http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) which contains movie reviews and sentiment lables which can be considered into 5 labels. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes.
The dataset is loaded from the text files to dataframes, following files were used:  sentiment_labels.txt and datasetSentences.txt These files were uploaded to Colab and used pandas to read and create dataframes.
we have joined sentences to the sentiments and after this data was around 12k records.
The sentiment values were converted to 5 classes using following code:
```python
for i in sentiment['sentiment values'] :
  if i >=0 and i<0.2:
    sentiment_class.append(1)
  elif i>=0.2 and i<0.4:
    sentiment_class.append(2)
  elif i>=0.4 and i<0.6:
    sentiment_class.append(3)
  elif i>=0.6 and i<0.8:
    sentiment_class.append(4)
  else:
    sentiment_class.append(5)
````
After this we converted data to small case and cleaned using regular expression:
```python
def smallCase(data):
  for i in data.index:
    data['sentence'][i] = data['sentence'][i].lower()
  return data
  
import re
def cleanText(data):
  data_small_case = smallCase(data)
  for i in data_small_case.index:
    data_small_case.sentence[i] = re.sub("[^-9A-Za-z ]", "" , data_small_case.sentence[i])
  return data_small_case
````
Then data was divided into train and test dataset in the ratio 70:30
Then we defined LSTM classifier with following hyperparameters:
```python
size_of_vocab = len(sentence.vocab)
embedding_dim = 200
num_hidden_nodes = 300
num_output_nodes = 6
num_layers = 4
dropout = 0.4
````
Model parameters:
```python

classifier(
  (embedding): Embedding(15090, 200)
  (encoder): LSTM(200, 300, num_layers=4, batch_first=True, dropout=0.4)
  (fc): Linear(in_features=300, out_features=6, bias=True)
)
The model has 5,789,406 trainable parameters
````
With Adam optimizer and learning rate of 2e-4,
we trained the model and training logs show:
```python
        Train Loss: 1.560 | Train Acc: 52.21%
	 Test. Loss: 1.534 |  Test. Acc: 51.27% 

	Train Loss: 1.515 | Train Acc: 53.12%
	 Test. Loss: 1.534 |  Test. Acc: 51.29% 

	Train Loss: 1.515 | Train Acc: 53.10%
	 Test. Loss: 1.534 |  Test. Acc: 51.27% 

	Train Loss: 1.513 | Train Acc: 53.12%
	 Test. Loss: 1.534 |  Test. Acc: 51.24% 

	Train Loss: 1.512 | Train Acc: 53.34%
	 Test. Loss: 1.533 |  Test. Acc: 51.24% 

	Train Loss: 1.510 | Train Acc: 53.60%
	 Test. Loss: 1.534 |  Test. Acc: 51.16% 

	Train Loss: 1.509 | Train Acc: 53.93%
	 Test. Loss: 1.533 |  Test. Acc: 51.27% 

	Train Loss: 1.506 | Train Acc: 54.16%
	 Test. Loss: 1.535 |  Test. Acc: 50.93% 

	Train Loss: 1.503 | Train Acc: 54.45%
	 Test. Loss: 1.533 |  Test. Acc: 51.02% 

	Train Loss: 1.501 | Train Acc: 54.57%
	 Test. Loss: 1.536 |  Test. Acc: 50.82% 
````
After training we tried on ten test sentences, and with following mapping of class:
categories = {0: "unknown", 1:"neutral", 2:"negative", 3:"positive", 4:"very positive", 5:"very negative"}

```python
for i in range(10):
  print(test.sentence[i])
  print("predicted: ", classify_sentence(test.sentence[i]))
  print("actual: ", test.sentiments[i])
bon apptit 
predicted:  neutral
actual:  3
one-of-a-kind near-masterpiece 
predicted:  neutral
actual:  3
for movie lovers as well as opera lovers  tosca is a real treat 
predicted:  neutral
actual:  2
excellent performances from jacqueline bisset and martha plimpton grace this deeply touching melodrama 
predicted:  neutral
actual:  3
 a story  an old and scary one  about the monsters we make  and the vengeance they take 
predicted:  neutral
actual:  2
a painfully slow cliche-ridden film filled with more holes than clyde barrow s car 
predicted:  neutral
actual:  2
a smart  arch and rather cold-blooded comedy 
predicted:  neutral
actual:  3
it does nt reach them  but the effort is gratefully received 
predicted:  neutral
actual:  4
cremaster  is at once a tough pill to swallow and a minor miracle of self-expression 
predicted:  neutral
actual:  3
shadyac shoots his film like an m night shyamalan movie  and he frequently maintains the same snail s pace  he just forgot to add any genuine tension 
predicted:  neutral
actual:  3
````
