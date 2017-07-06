import numpy as np
import os

#embedding the position 
def pos_embed(x):
	if x < -60:
		return 0
	if x >= -60 and x <= 60:
		return x+61
	if x > 60:
		return 122
#find the index of x in y, if x not in y, return -1
def find_index(x,y):
	flag = -1
	for i in range(len(y)):
		if x != y[i]:
			continue
		else:
			return i
	return flag

#reading data
def init():
	
	print 'reading word embedding data...'
	vec = []
	word2id = {}
	f = open('./origin_data/vec.txt')
	f.readline()
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		word2id[content[0]] = len(word2id)
		content = content[1:]
		content = [(float)(i) for i in content]
		vec.append(content)
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	
	dim = 50
	vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
	vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
	vec = np.array(vec,dtype=np.float32)
	
	print 'reading relation to id'
	relation2id = {}	
	f = open('./origin_data/relation2id.txt','r')
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		relation2id[content[0]] = int(content[1])
	f.close()

	#length of sentence is 70
	fixlen = 70
	#max length of position embedding is 60 (-60~+60)
	maxlen = 60

	train_sen = {} #{entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
	train_ans = {} #{entity pair:[label1,label2,...]} the label is one-hot vector


	print 'reading relation type data...'
	f = open('./origin_data/relation2type.txt','r')
	type_head=[]
	type_tail=[]
	while True:
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()		
		rel_name = content[0]
		rel_id= content[1]
		if rel_name == 'NA':
			continue

		head_type=content[2]
		tail_type=content[3]
		
		type_words = head_type.strip().split('/')
 		h_type = ''
		for i in range(len(type_words)):
			words = ''
			word = 0
			if(type_words[i] == ''):
				continue
			tmp = type_words[i].split('_')
			if len(tmp)==1:
				if tmp[0] not in word2id:
                                        word = word2id['UNK']
                                else:
                                        word = word2id[tmp[0]]
                                words += str(word)
			else:
				for j in range(len(tmp)):
					if tmp[j] not in word2id:
						word = word2id['UNK']
					else:
						word = word2id[tmp[j]]
					if j==len(tmp)-1:
						words += str(word)
					else:
						words += str(word) + '_'
			h_type += words + ' '
		
		type_head.append(h_type)

                type_words = tail_type.strip().split('/')
                t_type = ''
                for i in range(len(type_words)):
                        words = ''
                        word = 0
                        if(type_words[i] == ''):
                                continue
                        tmp = type_words[i].split('_')
                        if len(tmp)==1:
                                if tmp[0] not in word2id:
                                        word = word2id['UNK']
                                else:
                                        word = word2id[tmp[0]]
                                words += str(word)
                        else:
                                for j in range(len(tmp)):
                                        if tmp[j] not in word2id:
                                                word = word2id['UNK']
                                        else:
                                                word = word2id[tmp[j]]
                                        if j==len(tmp)-1:
                                                words += str(word)
                                        else:
                                                words += str(word) + '_'
                        t_type += words + ' '

		type_tail.append(t_type)

	print 'head types for relations %d' % len(type_head)
	print 'tail types for relations %d' % len(type_tail)
	print 'head types %s' % type_head
	print 'tail types %s' % type_tail
init()

