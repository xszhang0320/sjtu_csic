import numpy as np

train_y_head = np.load('./data/small_y_head.npy')
train_y_tail = np.load('./data/small_y_tail.npy')
train_y = np.load('./data/small_y.npy')
test_new_y_head = np.load('./data/test_new_y_head.npy')
test_new_y_tail = np.load('./data/test_new_y_tail.npy')

f = open('./tmp/y_head.txt','w')
index = 0
for i in train_y_head:
	if index  == 1000 :
		break
	index += 1
	for j in train_y_head[i]:
        	f.write(str(train_y_head[i][j])+'\t')
	f.write('\n')
f.close()

index = 0                                     #print 'y_tail shape %s' % temp_y_tail.shape
f = open('./tmp/y_tail.txt','w')
for i in train_y_tail:
	if index == 1000:
		break
	index += 1
	for j in train_y_tail[i]:
        	f.write(str(train_y_tail[i][j])+'\t')
	f.write ('\n')
f.close()

index = 0
f = open('./tmp/y.txt','w')
for i in train_y:
        if index == 1000:
                break
	index += 1
        for j in train_y[i]:
                f.write(str(train_y[i][j])+'\t')
        f.write('\n')
f.close()


index = 0
f = open('./tmp/y__test_head.txt','w')
for i in test_new_y_head:
        if index == 1000:
                break
	index += 1
        for j in test_new_y_head[i]:
                f.write(str(test_new_y_head[i][j])+'\t')
        f.write('\n')
f.close()

index = 0                      #print 'y_tail shape %s' % temp_y_tail.shape
f = open('./tmp/y_test_tail.txt','w')
for i in test_new_y_tail:
	if index == 1000:
                break
	index += 1
        for j in test_new_y_tail[i]:
                f.write(str(test_new_y_tail[i][j])+'\t')
        f.write ('\n')
f.close()

