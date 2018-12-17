from mpl_toolkits import mplot3d
# import pandas
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt

def main(args):
	# df = pandas.read_csv('points.csv',header=None)
	#dropping DLV points with likelihood < 1e-4
	# df = df.drop(df[df[3] < 1e-4].index)
	df = pandas.read_csv("points.csv",header=None)
	#dropping DLV points with likelihood < 1e-4
	df = df.truncate(before = args.num_plane*6038,after=args.num_points)
	df = df.drop(df[df[3] < 3e-1].index)

	data = df.values

	# data = np.loadtxt(open("points.csv", "rb"), delimiter=",")

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# Data for three-dimensional scattered points
	start = 1
	end = data.shape[0]
	zdata = data[start:end,2]
	ydata = data[start:end,1]
	xdata = data[start:end,0]
	cdata = 0.005*data[start:end,3]
	# cdata = zdata

	# print(xdata.shape)
	# ### show 3d plot
	# ax.scatter3D(xdata, ydata, zdata,c=cdata, cmap='viridis', linewidth=0.1,s=5)#, c=cdata, cmap='Reds')
	# plt.show()

	heatmap, xedges, yedges = np.histogram2d(xdata*(ydata.min()/ydata.max()), ydata, bins=50)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	plt.clf()
	plt.imshow(heatmap.T, extent=extent, origin='lower')
	plt.show()
if __name__ == '__main__':

	# print("hello")
	"""This block parses command line arguments and runs the training/testing main block"""
	print("Parsing arguments...")
	import argparse

	p = argparse.ArgumentParser()

	# p.add_argument("--num_points", default='/home/parthc/pgpd/train_data/0905/pos/', type=str, help="where point csvs are")
	p.add_argument("--num_points", default=6038, type=int, help="points to plot from sub DLV ")
	p.add_argument("--num_plane", default=0, type=int, help="points to plot from sub DLV ")
	
	#6038 corresponds to 1st plane
	args = p.parse_args()
	main(args)