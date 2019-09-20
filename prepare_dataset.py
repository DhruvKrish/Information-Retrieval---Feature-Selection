'''
Place the Classic dataset (data files) on creating a folder by name "classic".
Create another folder "Classic_Dataset" and 4 sub folders by name "med", "cran", "cicsi", "cacm".
 
The hardcoded number(a, b, c, d) of files will be copied from the "classic" to corresponding folders ("cacm", "med", "cisi", "cran").

'''
#"""
import shutil
import os
def prepare_dataset():
	cwd = os.getcwd()
	dirPath = cwd + "/Classic_Dataset/med"
	fileList = os.listdir(dirPath)
	for fileName in fileList:
		os.remove(dirPath+"/"+fileName)

	dirPath = cwd + "/Classic_Dataset/cran"
	fileList = os.listdir(dirPath)
	for fileName in fileList:
		os.remove(dirPath+"/"+fileName)

	dirPath = cwd + "/Classic_Dataset/cisi"
	fileList = os.listdir(dirPath)
	for fileName in fileList:
		os.remove(dirPath+"/"+fileName)

	dirPath = cwd + "/Classic_Dataset/cacm"
	fileList = os.listdir(dirPath)
	for fileName in fileList:
		os.remove(dirPath+"/"+fileName)

	#CACM, CISI, CRAN, MED (total number of docs in corresponding classed : 3204, 1460, 1400, 1033)
	a = 15
	b = 20
	c = 14
	d = 24


	for j in range(0, a):
		dest = cwd + "/Classic_Dataset/cacm"
		num = "%.6d" % (j+1)
		name = "/classic/cacm." + (str)(num)
		source = cwd + name
		shutil.copy2(source, dest)

	for j in range(0, b):
		dest = cwd + "/Classic_Dataset/cisi"
		num = "%.6d" % (j+1)
		name = "/classic/cisi." + (str)(num)
		source = cwd + name
		shutil.copy2(source, dest)

	for j in range(0, c):
		dest = cwd + "/Classic_Dataset/cran"
		num = "%.6d" % (j+1)
		name = "/classic/cran." + (str)(num)
		source = cwd + name
		shutil.copy2(source, dest)

	for j in range(0, d):
		dest = cwd + "/Classic_Dataset/med"
		num = "%.6d" % (j+1)
		name = "/classic/med." + (str)(num)
		source = cwd + name
		shutil.copy2(source, dest)
