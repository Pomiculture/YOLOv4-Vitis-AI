###############################################################################################################

# Source : https://newbedev.com/given-a-tensor-flow-model-graph-how-to-find-the-input-node-and-output-node-names

###############################################################################################################

import argparse
import tensorflow as tf

###############################################################################################################

def load_tf_graph(graph_name):
	"""
	Load a TensorFlow graph 'graph_name'.
	"""
	# Initialize a dataflow graph structure
	graph_def = tf.Graph().as_graph_def()
	# Parse the Tensorflow graph
	graph_def.ParseFromString(tf.io.gfile.GFile(graph_name, "rb").read())
	# Imports the graph from graph_def into the current default Graph
	tf.import_graph_def(graph_def, name = '')
	# Get the default graph for the current thread
	graph = tf.compat.v1.get_default_graph()
	return graph


def get_in_out_tensors(graph):
	"""
	Get the input and output tensors from the TensorFlow graph 'graph'.
	"""
	# Get the graph nodes that perform computation on tensors
	ops = graph.get_operations()
	# Initialize input and output tensors
	inputs = []
	outputs_set = set(ops)
	# Process operations
	for op in ops:
		# The input nodes are nodes without input
		if len(op.inputs) == 0 and op.type != 'Const':
			inputs.append(op)
		# The output nodes are nodes without input
		else:
			for input_tensor in op.inputs:
				if input_tensor.op in outputs_set:
					outputs_set.remove(input_tensor.op)
	outputs = list(outputs_set)
	return inputs, outputs


def display_tensor(tensor_list, disp_shape=False):
	"""
	Display tensor name and shape for each tensor in 'tensor_list'.
	"""
	i = 1
	for tensor in tensor_list:
		if(disp_shape):
			# Extract tensor shape
			shape = str(tensor.get_attr('shape').dim)
			shape = shape.replace('size: ', '')
			shape = shape.replace('\n', '')
			# Display name and shape of tensor
			print("- Tensor n°{0} : name={1}, shape={2}".format(i, tensor.name, shape))
		else:
			# Display tensor name
			print("- Tensor n°{0} : name={1}".format(i, tensor.name))
		i += 1
	print("\n")


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g',  '--graph', type=str,   default='./graph.pb',  help="Path to the TensorFlow graph to analyze. Default is './graph.pb'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print('--graph:', args.graph)
	print('------------------------------------\n')
	
	# Load TensorFlow graph to analyze
	graph = load_tf_graph(args.graph)

	# Read the input and output tensor names and dimensions from the graph
	input_tensors, output_tensors = get_in_out_tensors(graph)       

	# Display input tensor
	print("Input tensor(s) :")
	display_tensor(input_tensors, disp_shape=True)

	# Display output tensor
	print("Output tensor(s) :")
	display_tensor(output_tensors)	


###############################################################################################################

if __name__ == '__main__':
	main()
