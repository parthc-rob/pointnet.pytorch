import pandas
import numpy as np
import os
import time

def main(args):

	# df = pandas.read_csv('points.csv',header=None)
	# #dropping DLV points with likelihood < 1e-4
	# df = df.drop(df[df[3] < 1e-4].index)

	# print(df[:][:])
	# df.to_csv('points_shapenet.pts', sep=' ')
	# #x = np.ones(df.shape[0],dtype=int)

	# d = pandas.DataFrame(1, index=np.arange(df.shape[0]), columns=[None])
	# d.to_csv ('test.seg', index = None, header=None)

	for file_name in os.listdir(args.data_path):
		if ".csv" in file_name:
			key_id = file_name.split('.')[0]

			df = pandas.read_csv(args.data_path + file_name,header=None)
			#dropping DLV points with likelihood < 1e-4
			df = df.drop(df[df[3] < 1e-4].index)

			outname =  key_id + '.pts'

			outdir = './points/'
			if not os.path.exists(outdir):
			    os.mkdir(outdir)

			fullname = os.path.join(outdir, outname)    
			# print(df[:][:])
			df.to_csv(fullname, sep=' ', index = None, header=None)
			#x = np.ones(df.shape[0],dtype=int)

			d = pandas.DataFrame(1, index=np.arange(df.shape[0]), columns=[None])

			outname =  key_id + '.seg'
			outdir = './points_label/'
			if not os.path.exists(outdir):
			    os.mkdir(outdir)

			fullname = os.path.join(outdir, outname)
			d.to_csv(fullname, index = None, header=None)

        
if __name__ == '__main__':

	start = time.time()
	# print("hello")
	"""This block parses command line arguments and runs the training/testing main block"""
	print("Parsing arguments...")
	import argparse

	p = argparse.ArgumentParser()

	p.add_argument("--data_path", default='/home/parthc/pgpd/train_data/0905/pos/', type=str, help="where point csvs are")
	p.add_argument("--point_label", default=1, type=int, help="pos/neg - 1 0 ")

	args = p.parse_args()
	main(args)

	end = time.time()
	print(end - start)

#np.savetxt('test.seg', x, delimiter='\n')

#from numpy import genfromtxt
#my_data = genfromtxt('points.csv', delimiter=',')
#print(my_data)
#print(my_data[1:3][:])
