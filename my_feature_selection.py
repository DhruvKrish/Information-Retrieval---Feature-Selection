import math
import numpy as np

def ntp_doc_freq(num_classes, matrix, labels):
	""" Computes ntp for each class and doc_freq. 
	Used in calculating score1 and score2 """
	num_docs, num_terms = matrix.shape
	ntp = np.zeros( (num_classes, num_terms), np.float )  # Sum per class
	# Compute ntp for each class
	for i in range(num_docs):
		for j in range(num_terms):
			if matrix[i][j] == 1 :
				ntp[ labels[i] ][j] += 1
		
	# Compute doc_freq
	doc_freq = ntp.sum( axis = 0 )
	
	return ntp, doc_freq

def calculate_ppd_cpd(num_classes, matrix, labels):
	""" Calculate score1 and score2 """
	num_docs, num_terms = matrix.shape
	ntp, doc_freq = ntp_doc_freq( num_classes, matrix, labels)
	
	return (probability_proportion_difference(ntp, doc_freq), cpd(ntp,doc_freq))

def probability_proportion_difference(ntp, doc_freq):
	""" Computes score2 ( ppd )"""
	num_terms = ntp.shape[1]
	ntn = doc_freq - ntp
	binary_tp = ntp.clip( max=1 )
	binary_tp.sum( axis=1 )
	
	wp = np.sum( binary_tp, axis=1 )
	wn = num_terms - wp
	
	ppd= ntp / ( np.array([wp]).T + float(num_terms) ) + ntn / ( np.array([wn]).T + float(num_terms) )
	
	return ppd


def cpd(ntp,doc_freq):
	""" Computes score1 ( cpd )"""
	num_terms = ntp.shape[1]
	ntn = doc_freq - ntp
	
	ab=abs(ntp-ntn)
	
	su=ntp+ntn
	
	for i in range(ab.shape[0]):
		for j in range(ab.shape[1]):
			ab[i][j]=float(ab[i][j]/float(su[i][j]))
										
	
	return ab
	


def harm_mean(ppd,cpd):
	""" Computes the harmonic mean of ppd and cpd """
	mu=ppd*cpd
	su=ppd+cpd
	for i in range(mu.shape[0]):
		for j in range(mu.shape[1]):
			mu[i][j]=float(mu[i][j]/float(su[i][j]))
	return mu

#k_docs is K*1659
def featureselection(ranked_matrix,k_docs):
	"""  """
	bag=[]
	sel=[0,0,0,0]
	while (len(bag)!=k_docs):		
		j=min(sel)
		i=sel.index(j)
		if ranked_matrix[i][j] not in bag:
			bag.append(ranked_matrix[i][j])
			sel[i]=sel[i]+1
		else:
			sel[i]=sel[i]+1
	return bag	




def ranked_term_classes(ppd_op,cpd_op):
	""" Rank docs according to harmonic mean of score1 and score2 """
	hm= harm_mean(ppd_op,cpd_op)
	
	hom=[]
	for i in range(hm.shape[0]):
		hom.append([])	
		temp=[]
		for j in range (hm.shape[1]):
			temp.append((hm[i][j],j))
		temp.sort(reverse=True)
		
		for j in range (hm.shape[1]):	
			hom[i].append(temp[j][1])									
	
	hom=np.array(hom)
	return hom



def get_selected_term_vectors( vectors, indices, num_docs=73 ):
	""" Extracts selected vectors from doc-term tf-idf matrix """
	matrix=np.zeros( (num_docs, len(indices)), np.float ) 
	for i in range(num_docs):
		for j in range(len(indices)):
			matrix[i][j]=vectors[i, indices[j] ]

	return matrix

def select_top_percent(ranked, voc, vectors, percent):
	
	percent = float(percent)
	terms_per_class= int((percent/100)*(len(voc)))
	indices = featureselection(ranked,terms_per_class)
	selected_vectors = get_selected_term_vectors( vectors, indices, 73 )
	return selected_vectors 


def do_ranking(doc_term_matrix, labels, num_classes=4):
	""" Does feature selection """
	binary_dt_matrix = np.array([ [ 0 if x==0 else 1 for x in d ] for d in doc_term_matrix ])
	
	ppd_op, cpd_op = calculate_ppd_cpd(num_classes, binary_dt_matrix, labels)

	return ranked_term_classes(ppd_op,cpd_op)
	

