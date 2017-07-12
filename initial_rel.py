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

	print 'reading relation type data...'
	f = open('./origin_data/relation2type.txt','r')
	type_head=[]
	type_tail=[]
	head_type2id = {}
	tail_type2id = {}
  	relation2type = {}
	while True:
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()		
		rel_name = content[0]
		rel_id= content[1]
		if rel_name == 'NA':
			continue
    		relation2type[content[0]]=[content[2],content[3]]

		head_type=content[2]
		tail_type=content[3]
		head_type2id['NA'] = 0
		tail_type2id['NA'] = 0
		if not head_type2id.has_key(content[2]):	
			head_type2id[content[2]]=len(head_type2id)
		if not tail_type2id.has_key(content[3]):
			tail_type2id[content[3]]=len(tail_type2id)
		
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

	#print 'head types for relations %d' % len(type_head)
	#print 'tail types for relations %d' % len(type_tail)
	#print 'head types %s' % type_head
	#print 'tail types %s' % type_tail
	type_head = np.array(type_head)
	type_tail = np.array(type_tail)
	#np.save('./data/type_head.npy',type_head)
	#np.save('./data/type_tail.npy',type_tail)
	
	f = open('./origin_data/head_type2id.txt','w')
	for k in head_type2id:
		f.write(str(k) + '\t' + str(head_type2id[k])+ '\n')
	f.close()

        f = open('./origin_data/tail_type2id.txt','w')
        for k in tail_type2id:
                f.write(str(k) + '\t' + str(tail_type2id[k])+ '\n')
	f.close()
	
	#length of sentence is 70
	fixlen = 70
	#max length of position embedding is 60 (-60~+60)
	maxlen = 60

	train_sen = {} #{entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
	train_ans = {} #{entity pair:[label1,label2,...]} the label is one-hot vector
	
	print 'reading train data...'
	f = open('./origin_data/train.txt','r')

	while True:
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()
		#get entity name
		en1 = content[2] 
		en2 = content[3]
		relation = 0
		if content[4] not in relation2id:
			relation = relation2id['NA']
		else:
			relation = relation2id[content[4]]
		if(content[4] not in relation2type):
          		head_id = 0
          		tail_id = 0
    		else:
          		head_id = head_type2id[relation2type[content[4]][0]]
          		tail_id = tail_type2id[relation2type[content[4]][1]]
		#put the same entity pair sentences into a dict
		tup = (en1,en2)
		label_tag = 0
		if tup not in train_sen:
			train_sen[tup]=[]
			train_sen[tup].append([])
			y_id = relation
			label_tag = 0
			label = [0 for i in range(len(relation2id))]
      			label_head = [0 for i in range(len(head_type2id))]
      			label_tail = [0 for i in range(len(tail_type2id))]
			label[y_id] = 1
      			label_head[head_id] = 1
      			label_tail[tail_id] = 1
			train_ans[tup] = []
			train_ans[tup].append([label,label_head, label_tail])
		else:
			y_id = relation
			label_tag = 0
			label = [0 for i in range(len(relation2id))]
			label[y_id] = 1
      			label_head = [0 for i in range(len(head_type2id))]
      			label_tail = [0 for i in range(len(tail_type2id))]
      			label_head[head_id] = 1
      			label_tail[tail_id] = 1     
			
			temp = find_index([label,label_head,label_tail],train_ans[tup])
			if temp == -1:
				train_ans[tup].append([label, label_head, label_tail])
				label_tag = len(train_ans[tup])-1
				train_sen[tup].append([])
			else:
				label_tag = temp

		sentence = content[5:-1]
		
		en1pos = 0
		en2pos = 0
		
		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			if(i == en1pos):
				pos_e1 = 1
			else:
				pos_e1 = 0
			if(i == en2pos):
				pos_e2 = 1
			else:
				pos_e2 = 0
			rel_e1 = pos_embed(i - en1pos)
                        rel_e2 = pos_embed(i - en2pos)
                        output.append([word,rel_e1,rel_e2,pos_e1,pos_e2])

		for i in range(min(fixlen,len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]
			
			output[i][0] = word

		train_sen[tup][label_tag].append(output)

	print('reading test data ...')

	test_sen = {} #{entity pair:[[sentence 1],[sentence 2]...]}
	test_ans = {} #{entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

	f = open('./origin_data/test.txt','r')

	while True:
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()
		en1 = content[2]
		en2 = content[3]
		relation = 0
                if content[4] not in relation2id:
                        relation = relation2id['NA']
                else:
                        relation = relation2id[content[4]]

                if(content[4] not in relation2type):
                        head_id = 0
                        tail_id = 0
                else:
                        head_id = head_type2id[relation2type[content[4]][0]]
                        tail_id = tail_type2id[relation2type[content[4]][1]]
	
		tup = (en1,en2)
		
		if tup not in test_sen:
			test_sen[tup]=[]
			y_id = relation
			label_tag = 0
                        label = [0 for i in range(len(relation2id))]
                        label_head = [0 for i in range(len(head_type2id))]
                        label_tail = [0 for i in range(len(tail_type2id))]
                        label[y_id] = 1
                        label_head[head_id] = 1
                        label_tail[tail_id] = 1
			test_ans[tup] = [label,label_head, label_tail]
		else:
			y_id = relation
			test_ans[tup][0][y_id] = 1
			#print 'test ans tup label head length %s' % test_ans[tup]
			test_ans[tup][1][head_id] = 1
			test_ans[tup][2][tail_id] = 1
			
		sentence = content[5:-1]

		en1pos = 0
		en2pos = 0
		
		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			if(i == en1pos):
                                pos_e1 = 1
                        else:
                                pos_e1 = 0
                        if(i == en2pos):
                                pos_e2 = 1
                        else:
                                pos_e2 = 0
			rel_e1 = pos_embed(i - en1pos)
			rel_e2 = pos_embed(i - en2pos)
			output.append([word,rel_e1,rel_e2,pos_e1,pos_e2])

		for i in range(min(fixlen,len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]

			output[i][0] = word
		test_sen[tup].append(output)
	
	train_x = []
	train_y = []
  	train_y_head = []
  	train_y_tail = []
	test_x = []
	test_y = []
	test_y_head = []
	test_y_tail = []

	print 'organizing train data'
	f = open('./data/train_q&a.txt','w')
	temp = 0
	for i in train_sen:
		if len(train_ans[i]) != len(train_sen[i]):
			print 'ERROR'
		lenth = len(train_ans[i])
		for j in range(lenth):
			train_x.append(train_sen[i][j])
			train_y.append(train_ans[i][j][0])
      			train_y_head.append(train_ans[i][j][1])
      			train_y_tail.append(train_ans[i][j][2])
			f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(train_ans[i][j]))+'\n')
			temp+=1
	f.close()

	print 'organizing test data'
	f = open('./data/test_q&a.txt','w')
	temp=0
	for i in test_sen:		
		test_x.append(test_sen[i])
		#test_y.append(test_ans[i])
                test_y.append(test_ans[i][j][0])
                test_y_head.append(test_ans[i][j][1])
                test_y_tail.append(test_ans[i][j][2])

		tempstr = ''
		for j in range(len(test_ans[i])):
			if test_ans[i][j]!=0:
				tempstr = tempstr+str(j)+'\t'
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+tempstr+'\n')
		temp+=1
	f.close()

	train_x = np.array(train_x)
	train_y = np.array(train_y)
  	train_y_head = np.array(train_y_head)
  	train_y_tail = np.array(train_y_tail)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	test_y_head = np.array(test_y_head)
	test_y_tail = np.array(test_y_tail)

	np.save('./data/vec.npy',vec)
	np.save('./data/train_new_x.npy',train_x)
	np.save('./data/train_new_y.npy',train_y)
  	np.save('./data/train_new_y_head.npy',train_y_head)
  	np.save('./data/train_new_y_tail.npy',train_y_tail)
	np.save('./data/testall_new_x.npy',test_x)
	np.save('./data/testall_new_y.npy',test_y)
        np.save('./data/test_new_y_head.npy',test_y_head)
        np.save('./data/test_new_y_tail.npy',test_y_tail)


def seperate():
	
	print 'reading training data'
	x_train = np.load('./data/train_new_x.npy')

	train_word = []
	train_pos1 = []
	train_pos2 = []
	train_ab_pos1 = []
	train_ab_pos2 = []

	print 'seprating train data'
	for i in range(len(x_train)):
		word = []
		pos1 = []
		pos2 = []
		ab_pos1 = []
		ab_pos2 = []
		for j in x_train[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			temp_ab_pos1 = []
			temp_ab_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
				temp_ab_pos1.append(k[3])
                        	temp_ab_pos2.append(k[4])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
			ab_pos1.append(temp_ab_pos1)
			ab_pos2.append(temp_ab_pos2)
		train_word.append(word)
		train_pos1.append(pos1)
		train_pos2.append(pos2)
		train_ab_pos1.append(ab_pos1)
		train_ab_pos2.append(ab_pos2)

	train_word = np.array(train_word)
	train_pos1 = np.array(train_pos1)
	train_pos2 = np.array(train_pos2)
	train_ab_pos1 = np.array(train_ab_pos1)
	train_ab_pos2 = np.array(train_ab_pos2)
	np.save('./data/train_word.npy',train_word)
	np.save('./data/train_pos1.npy',train_pos1)
	np.save('./data/train_pos2.npy',train_pos2)
	np.save('./data/train_ab_pos1.npy', train_ab_pos1)
	np.save('./data/train_ab_pos2.npy', train_ab_pos2)

	print 'seperating test all data'
	x_test = np.load('./data/testall_new_x.npy')

	test_word = []
	test_pos1 = []
	test_pos2 = []
        test_ab_pos1 = []
        test_ab_pos2 = []


	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []
                ab_pos1 = []
                ab_pos2 = []
		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
	                temp_ab_pos1 = []
        	        temp_ab_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
                                temp_ab_pos1.append(k[3])
                                temp_ab_pos2.append(k[4])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
			ab_pos1.append(temp_ab_pos1)
			ab_pos2.append(temp_ab_pos2)
		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)
		test_ab_pos1.append(ab_pos1)
		test_ab_pos2.append(ab_pos2)

	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)
        test_ab_pos1 = np.array(test_ab_pos1)
        test_ab_pos2 = np.array(test_ab_pos2)	

	np.save('./data/testall_word.npy',test_word)
	np.save('./data/testall_pos1.npy',test_pos1)
	np.save('./data/testall_pos2.npy',test_pos2)
        np.save('./data/test_ab_pos1.npy', test_ab_pos1)
        np.save('./data/test_ab_pos2.npy', test_ab_pos2)

def getsmall():
	
 	print 'reading training data'
	word = np.load('./data/train_word.npy')
	pos1 = np.load('./data/train_pos1.npy')
	pos2 = np.load('./data/train_pos2.npy')
        ab_pos1 = np.load('./data/train_ab_pos1.npy')
        ab_pos2 = np.load('./data/train_ab_pos2.npy')
	y = np.load('./data/train_new_y.npy')
	y_head = np.load('./data/train_new_y_head.npy')
  	y_tail = np.load('./data/train_new_y_tail.npy')
  

	new_word = []
	new_pos1 = []
	new_pos2 = []
	new_ab_pos1 = []
	new_ab_pos2 = []
	new_y = []
	new_y_head = []
	new_y_tail = []

	#we slice some big batch in train data into small batches in case of running out of memory
	print 'get small training data'
	for i in range(len(word)):
		lenth = len(word[i])
		if lenth <= 1000:
			new_word.append(word[i])
			new_pos1.append(pos1[i])
			new_pos2.append(pos2[i])
			new_ab_pos1.append(ab_pos1[i])
			new_ab_pos2.append(ab_pos2[i])
			new_y.append(y[i])
			new_y_head.append(y_head[i])
			new_y_tail.append(y_tail[i])            

		if lenth > 1000 and lenth < 2000:
			
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:])
			
			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:])

			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:])
			
                        new_ab_pos1.append(ab_pos1[i][:1000])
			new_ab_pos1.append(ab_pos1[i][1000:])

                        new_ab_pos2.append(ab_pos2[i][:1000])
                        new_ab_pos2.append(ab_pos2[i][1000:])

			new_y.append(y[i])
			new_y.append(y[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i]) 
		
		if lenth > 2000 and lenth < 3000:
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:2000])
			new_word.append(word[i][2000:])
			
			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:2000])
			new_pos1.append(pos1[i][2000:])
			
			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:2000])
			new_pos2.append(pos2[i][2000:])
			
                        new_ab_pos1.append(ab_pos1[i][:1000])
                        new_ab_pos1.append(ab_pos1[i][1000:2000])
                        new_ab_pos1.append(ab_pos1[i][2000:])

                        new_ab_pos2.append(ab_pos2[i][:1000])
                        new_ab_pos2.append(ab_pos2[i][1000:2000])
                        new_ab_pos2.append(ab_pos2[i][2000:])

			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i]) 
			new_y_tail.append(y_tail[i])

		if lenth > 3000 and lenth < 4000:
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:2000])
			new_word.append(word[i][2000:3000])
			new_word.append(word[i][3000:])
		
			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:2000])
			new_pos1.append(pos1[i][2000:3000])
			new_pos1.append(pos1[i][3000:])

			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:2000])
			new_pos2.append(pos2[i][2000:3000])
			new_pos2.append(pos2[i][3000:])

                        new_ab_pos1.append(ab_pos1[i][:1000])
                        new_ab_pos1.append(ab_pos1[i][1000:2000])
                        new_ab_pos1.append(ab_pos1[i][2000:3000])
                        new_ab_pos1.append(ab_pos1[i][3000:])


                        new_ab_pos2.append(ab_pos2[i][:1000])
                        new_ab_pos2.append(ab_pos2[i][1000:2000])
                        new_ab_pos2.append(ab_pos2[i][2000:3000])
                        new_ab_pos2.append(ab_pos2[i][3000:])

			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i]) 
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i])

		if lenth > 4000:
			
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:2000])
			new_word.append(word[i][2000:3000])
			new_word.append(word[i][3000:4000])
			new_word.append(word[i][4000:])

			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:2000])
			new_pos1.append(pos1[i][2000:3000])
			new_pos1.append(pos1[i][3000:4000])
			new_pos1.append(pos1[i][4000:])

			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:2000])
			new_pos2.append(pos2[i][2000:3000])
			new_pos2.append(pos2[i][3000:4000])
			new_pos2.append(pos2[i][4000:])

                        new_ab_pos1.append(ab_pos1[i][:1000])
                        new_ab_pos1.append(ab_pos1[i][1000:2000])
                        new_ab_pos1.append(ab_pos1[i][2000:3000])                       
                        new_ab_pos1.append(ab_pos1[i][3000:4000])
                        new_ab_pos1.append(ab_pos1[i][4000:])
                        
                        new_ab_pos2.append(ab_pos2[i][:1000])
                        new_ab_pos2.append(ab_pos2[i][1000:2000])
                        new_ab_pos2.append(ab_pos2[i][2000:3000])
                        new_ab_pos2.append(ab_pos2[i][3000:4000])
                        new_ab_pos2.append(ab_pos2[i][4000:])

			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_head.append(y_head[i])
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i]) 
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i])
			new_y_tail.append(y_tail[i])


	new_word = np.array(new_word)
	new_pos1 = np.array(new_pos1)
	new_pos2 = np.array(new_pos2)
	new_ab_pos1 = np.array(new_ab_pos1)
	new_ab_pos2 = np.array(new_ab_pos2)
	new_y = np.array(new_y)
	new_y_head = np.array(new_y_head)
	new_y_tail = np.array(new_y_tail)

	np.save('./data/small_word.npy',new_word)
	np.save('./data/small_pos1.npy',new_pos1)
	np.save('./data/small_pos2.npy',new_pos2)
	np.save('./data/small_ab_pos1.npy', new_ab_pos1)
        np.save('./data/small_ab_pos2.npy', new_ab_pos2)	
	np.save('./data/small_y.npy',new_y)
	np.save('./data/small_y_head.npy',new_y_head)
	np.save('./data/small_y_tail.npy',new_y_tail)

#get answer metric for PR curve evaluation
def getans():
	test_y = np.load('./data/testall_y.npy')
	eval_y = []
	for i in test_y:
		eval_y.append(i[1:])
	allans = np.reshape(eval_y,(-1))
	np.save('./data/allans.npy',allans)

def get_metadata():
	fwrite = open('./data/metadata.tsv','w')
	f = open('./origin_data/vec.txt')
	f.readline()
	while True:
		content = f.readline().strip()
		if content == '':
			break
		name = content.split()[0]
		fwrite.write(name+'\n')
	f.close()
	fwrite.close()

init()
seperate()
getsmall()
getans()
get_metadata()
