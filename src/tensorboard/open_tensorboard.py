import os
import argparse
import tensorflow as tf

###############################################################################################################

def load_graph(path_to_pb, log_dir):
	"""
	Load the TF graph from path 'path_to_pb' and write to log directory.
	"""
	# Get TF graph
	with tf.Session() as sess:
		with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
			graph_def = tf.compat.v1.GraphDef()
			graph_def.ParseFromString(f.read())
			g_in = tf.import_graph_def(graph_def)
			print('Nodes in graph :')
			for node in graph_def.node:
				print(node.name)

	# Write to log directory
	train_writer = tf.compat.v1.summary.FileWriter(log_dir)
	train_writer.add_graph(sess.graph)


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g',  '--graph',   type=str,   default='./model/tensorflow/yolov4.pb',  help="Path to the TensorFlow graph. Default is './model/tensorflow/yolov4.pb'.")
	parser.add_argument('-l',  '--log_dir', type=str,   default='./model/tensorflow/tb_logs',    help="Output folder for the TensorBoard log data. Default is '/model/tensorflow/tb_logs'.")
	parser.add_argument('-p',  '--port',    type=str,   default='6006',                          help="TensorBoard port number. Default is '6006'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --graph:', args.graph)
	print(' --log_dir:', args.log_dir)
	print(' --port:', args.port)
	print('------------------------------------\n')

	# Load the graph to use Tensorboard
	load_graph(args.graph, args.log_dir)   

	# Run tensorboard
	print("To visualize the network's graph on Tensorboard open the following link :")
	print("http://localhost:" + args.port + "/#graphs")
	os.system("tensorboard --logdir=" + args.log_dir + " --port " + args.port + " --host localhost")
	

###############################################################################################################

if __name__ == '__main__':
    main()

