# TODO
# NameError: name 'fdist' is not defined
#


# Andrew Melnyk ALs-12
import nltk
from nltk.corpus import brown
sing=[]
pl=[]
for (w,t) in brown.tagged_words(categories='news'): # ³��� ��� �� ����� ��� �������� � �����.
	if t=='NN':
		sing.append(w)

fdistsing = nltk.FreqDist(sing) # ��������� ���������� ������ ������������ ���

for (w,t) in brown.tagged_words(categories='news'): # ³��� ��� �� ����� ��� �������� � ������ � ��������� � ������ ��� �������� �����
	if t=='NNS':
		pl.append(w[:-1])

fdistpl = nltk.FreqDist(pl) # ��������� ���������� ������ ������������ ���

<<<<<<< HEAD
for i in fdist.keys():	# ��������� ������� ��������� � ������ ��������. ���� ������� ��������� ����� � ������ �� � ����� �� ���������� ���� �� ����� ����� �����.
	if fdistpl[i]>fdist[i]:
=======
for i in fdistpl.keys():	# ��������� ������� ��������� � ������ ��������. ���� ������� ��������� ����� � ������ �� � ����� �� ���������� ���� �� ����� ����� �����.
	if fdistpl[i]>fdistsing[i]:
>>>>>>> Випралені задачі
		print i
