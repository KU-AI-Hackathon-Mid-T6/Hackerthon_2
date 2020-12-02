# -*- coding: utf-8 -*-

from konlpy.tag import Okt
import numpy as np
from tqdm import tqdm
import pickle

# pickle 파일을 얻기 위한 코드입니다.
# baseline에서 pickle 파일을 load하고 해당 코드를 추가해야 사용할 수 있습니다.  
# g_train_feature = torch.tensor(g_train_feature, dtype=torch.long).cuda()
# g_test_feature = torch.tensor(g_test_feature,dtype= torch.long).cuda()


#2. 태그 리스트 별로 bi-gram을 만들어준다. 
#예) "홍길동"->["홍길","길동"]
def make_n_gram(NER):
    NER_n = []
    for tag in NER:
        if len(tag)==1: #한 글자의 경우 바로 추가
            NER_n.append(tag)
        else:
            for i in range(len(tag)-1):
                NER_n.append(tag[i:i+2])
    return NER_n

#1. gazette.txt 파일을 읽어와 태그별 리스트에 저장한다. 
def load_gazette():
    file = open( 'gazette.txt', 'r', encoding = "utf-8")

# 단어/라벨 리스트 생성
    OG, LC,DT, PS, TI = [],[],[],[],[]

    i=0
    for line in tqdm(file.readlines()):
        i+=1
        words, tags = line.strip().split('\t')

        if "OG" in tags:
            OG.append(words)
        if "LC" in tags:
            LC.append(words)
        if "DT" in tags:
            DT.append(words)
        if "PS" in tags:
            PS.append(words)
        if "TI" in tags:
            TI.append(words)
    print(len(OG)+len(LC)+len(DT)+len(PS)+len(TI))
            
    OG_n= make_n_gram(OG)
    LC_n= make_n_gram(LC)
    DT_n= make_n_gram(DT)
    PS_n= make_n_gram(PS)
    TI_n= make_n_gram(TI)    
            
    return OG_n, LC_n,DT_n, PS_n, TI_n 

OG_n, LC_n,DT_n, PS_n, TI_n = load_gazette()

# with open ('OG.pickle','wb') as f:
#     pickle.dump(OG_n, f)
# with open ('LC.pickle','wb') as f:
#     pickle.dump(LC_n, f)
# with open ('DT.pickle','wb') as f:
#     pickle.dump(DT_n, f)
# with open ('PS.pickle','wb') as f:
#     pickle.dump(PS_n, f)
# with open ('TI.pickle','wb') as f:
#     pickle.dump(TI_n, f)


def max_length_feature(f, max_length = 120):
    features = np.zeros([max_length, 12],dtype = np.int)
    
    #[0 0 0...0 0 0]은 PAD의 의미를 갖게 된다.
    for i in range(len(f)):
        features[i] = f[i] 
    
    return features

def find_tag(word,al,feature,idx):
    if word in LC_n:
        feature[idx][3]+=1
    if word in OG_n:
        feature[idx][4]+=1                     
    if word in DT_n:
        feature[idx][2]+=1 #이 때는 앞 7개의 태그만 사용한다. 
    if word in PS_n:
        feature[idx][5]+=1
    if word in TI_n:
        feature[idx][6]+=1
    if word not in al:
        feature[idx][1]=1 # 단어가 없으면 O 태그 
    return feature;

#형태소 단위로 list look up을 수행한다. 
def make_feature(sentence,max_length =120):
    feature = np.zeros([max_length,12],dtype= np.int)
    feature_2 =[]
    # <SP> O B_DT B_LC B_OG B_PS B_TI I_DT I_LC I_OG I_PS I_TI 
    #만약 BI태그없이 사용하고 싶다면 
    #[<SP> O DT LC OG PS TI] ->[max_length,7]로 변경, 142번째 줄 [1 0 0 0 0 0 0]로 수정
    
    okt = Okt()
    sentence = sentence.replace("<SP>","$")
    #띄어쓰기 위치 저장
    target = "$"
    index = -1
    sp =[]
    while True:
        index = sentence.find(target, index + 1)
        if index == -1:
            break
        sp.append(index)
    
    sentence = sentence.replace("$"," ")
    
    #line = []
    #line = sentence.split("<SP>") ->띄어쓰기 단위로 lookup할 때 사용한 코드
    
   # okt를 이용해서 형태소 분석을 해준다.
    line = okt.pos(sentence)
    
    al = OG_n+ LC_n+DT_n+ PS_n+ TI_n # O 태그를 얻기 위해 모두 합쳐줌
    
    #! 형태소 단어를 bi_gram으로 잘라서 list look up한다. 
    idx =-1
    for w in line:
        word = w[0]
        idx+=1
        if(idx<120):
            if len(word)-1==0:# 한 글자인 형태소는 잡아내지 못해서 따로 만들어준다. 
              find_tag(word, al, feature, idx) #tag에 속해있는지 확인
            else:
                for i in range(len(word)-1):  
                    find_tag(word[i:i+2], al, feature, idx)
                        
    #I_tag 벡터를 만들어준다.               
    i_feature = np.zeros([max_length,12],dtype= np.int)
    i_feature[:,0:2] =feature[:,0:2]
    i_feature[:,7:12]= feature[:,2:7]
    
    #print(feature[0],i_feature[0])
    # 음절 별로 tagging을 하기 위해서 형태소 길이만큼 feature를 붙인다.
    for idx,w in enumerate(line):
        if idx >= max_length: 
            break;
        word = w[0]
        i=0
        for i in range(len(word)):
            #print(word, feature[idx])
            if i == 0:
                feature_2.append(feature[idx]) #맨 처음에는 B_tag
            else:
                feature_2.append(i_feature[idx]) #그 다음에는 I_tag
    
    for idx,i in enumerate(sp):
        feature_2.insert(i,[1,0,0,0,0,0,0,0,0,0,0,0]) #<SP>인덱스에 sp 벡터 추가
    
    feature_2 = feature_2[:max_length] #max_length로 자르기 
    feature_2 = np.array(feature_2)
    
    #baseline에서 tensor로 바꾸기 위해서는 [max_length,12]여야 한다. 
    #feature_2 = [curr_max_length,12]여서 일정하지 않음
    final_feature = max_length_feature(feature_2) 
    
    #print(sentence,final_feature)
    return final_feature

# 1. 학습, 테스트 문서를 불러와 문장 별로 feature를 얻은 후, features 리스트에 저장한다.
# baseline의 load_data함수와 동일한 역할 
def load_text(filename):
    if filename =="train":
        file = open("ner_train.txt", 'r', encoding = "utf-8")
    else:
        file = open("example.txt", 'r', encoding = "utf-8")
        
    
    features =[]
    
    for line in tqdm(file.readlines()):
        try:
            id, sentence, tags = line.strip().split('\t')
        except:
            id, sentence = line.strip().split('\t')
        # 음절별로 띄어쓰기 되어있는 문장을 붙여준다. 
        sentence = sentence.replace(" ","")
        feature = make_feature(sentence)
        features.append(feature)
    
    return features

features = load_text("train")
with open ('gazette_train_feature.pickle','wb') as f:
    pickle.dump(features, f)

features_2 = load_text("test")
with open ('gazette_test_feature.pickle','wb') as f:
    pickle.dump(features, f)

