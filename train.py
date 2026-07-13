import pickle,pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Embedding,SimpleRNN,RepeatVector,TimeDistributed,Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

df=pd.read_csv("technology_sentences.csv")
tok=Tokenizer(oov_token="<OOV>")
tok.fit_on_texts(df.keyword.astype(str).tolist()+df.sentence.astype(str).tolist())
vocab=len(tok.word_index)+1
X=tok.texts_to_sequences(df.keyword.astype(str))
mi=max(len(x) for x in X)
X=pad_sequences(X,maxlen=mi,padding="post")
Y=tok.texts_to_sequences(df.sentence.astype(str))
mo=max(len(y) for y in Y)
Y=pad_sequences(Y,maxlen=mo,padding="post")
Y=to_categorical(Y,num_classes=vocab)
xtr,xte,ytr,yte=train_test_split(X,Y,test_size=0.2,random_state=42)
inp=Input(shape=(mi,))
x=Embedding(vocab,128)(inp)
enc=SimpleRNN(128)(x)
rep=RepeatVector(mo)(enc)
dec=SimpleRNN(128,return_sequences=True)(rep)
out=TimeDistributed(Dense(vocab,activation="softmax"))(dec)
m=Model(inp,out)
m.compile("adam","categorical_crossentropy",metrics=["accuracy"])
m.fit(xtr,ytr,epochs=50,batch_size=4,verbose=1,validation_data=(xte,yte))
m.save("one_to_many_rnn.keras")
with open("tokenizer.pkl","wb") as f: pickle.dump(tok,f)
