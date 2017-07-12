import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


#plt.clf()
#filename = ['Hoffmann','MIMLRE','Mintz','PCNN+ATT']
#color = ['turquoise', 'darkorange', 'cornflowerblue','darkgreen' ]
#for i in range(len(filename)):
#	precision = np.load('./data/'+filename[i]+'_precision.npy')
#	recall  = np.load('./data/'+filename[i]+'_recall.npy')
#	plt.plot(recall,precision,color = color[i],lw=2,label = filename[i])


#ATTENTION: put the model iters you want to plot into the list
#model_iter = [9500,10000,10500,11000,11500,12000,12500,13000,13500,14000]
#model_iter = [10900]
model_iter = range(9100,15000,100)
for one_iter in model_iter:
	plt.clf()
	filename = ['Hoffmann','MIMLRE','Mintz','PCNN+ATT']
	color = ['turquoise', 'darkorange', 'cornflowerblue','darkgreen' ]
	for i in range(len(filename)):
        	precision = np.load('./data/'+filename[i]+'_precision.npy')
        	recall  = np.load('./data/'+filename[i]+'_recall.npy')
        	plt.plot(recall,precision,color = color[i],lw=2,label = filename[i])

	y_true = np.load('./data/allans.npy')
	y_scores = np.load('./out/allprob_type_iter_'+str(one_iter)+'.npy')
	average_precision = average_precision_score(y_true, y_scores)
	if average_precision < 0.36 :
		continue
	print (average_precision)
	precision,recall,threshold = precision_recall_curve(y_true,y_scores)

	plt.plot(recall[:], precision[:], lw=2, color='navy',label='Tree LSTM')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.0])
	plt.xlim([0.0, 0.4])
	plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
	plt.legend(loc="upper right")
	plt.grid(True)
	plt.savefig('./tmp/type_mt_iter_'+str(one_iter))

	plt.close(0)
