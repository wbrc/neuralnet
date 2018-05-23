// Package neuralnet provides basic feed-forward mlp
// neural-networks
package neuralnet

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"time"
)

// R is the source to be used for the initial random weights
var R *rand.Rand

func init() {
	R = rand.New(rand.NewSource(time.Now().Unix()))
}

type neuron struct {
	weights    []float64 // the weights of the inputs
	bias       float64   // the bias for the neuron
	output     float64   // the sum of the weighted inputs
	activation float64   // the output with applied activation function
	delta      float64   // delta of the neuron
}

type layer struct {
	neurons []neuron
}

// NeuralNetwork that contains layers, neurons and biases. It provides
// a fully connected network for supervised learning
type NeuralNetwork struct {
	layers []layer
	a      func(float64) float64 // activation function
	da     func(float64) float64 // derivation of activation function
}

type networkDump struct {
	Layers []layerDump
}

type layerDump struct {
	Neurons []neuronDump
}
type neuronDump struct {
	Weights []float64
	Bias    float64
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidD(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func newLayer(synapses, neurons uint) layer {
	l := layer{}

	l.neurons = make([]neuron, 0, neurons)

	for i := 0; i < int(neurons); i++ {
		l.neurons = append(l.neurons, newNeuron(synapses))
	}

	return l
}

func newNeuron(synapses uint) neuron {
	n := neuron{}

	n.weights = make([]float64, synapses, synapses)
	for i := range n.weights {
		n.weights[i] = R.NormFloat64()
	}
	n.bias = R.NormFloat64()

	return n
}

func (n *NeuralNetwork) feedforward(input []float64) {

	// feed input data into input layer
	for i := range input {
		n.layers[0].neurons[i].activation = input[i]
	}

	// feed forward through each layer
	for layerIndex := 1; layerIndex < len(n.layers); layerIndex++ {
		for neuronIndex := range n.layers[layerIndex].neurons {
			var output float64
			for weightIndex := range n.layers[layerIndex].neurons[neuronIndex].weights {
				output += n.layers[layerIndex-1].neurons[weightIndex].activation * n.layers[layerIndex].neurons[neuronIndex].weights[weightIndex]
			}
			output += n.layers[layerIndex].neurons[neuronIndex].bias
			n.layers[layerIndex].neurons[neuronIndex].output = output
			n.layers[layerIndex].neurons[neuronIndex].activation = n.a(output)
		}
	}
}

func (n *NeuralNetwork) backpropagate(targetOutput []float64, eta float64) {

	// calc deltas of output layer
	for i := range n.layers[len(n.layers)-1].neurons {
		n.layers[len(n.layers)-1].neurons[i].delta = (targetOutput[i] - n.layers[len(n.layers)-1].neurons[i].activation) * n.da(n.layers[len(n.layers)-1].neurons[i].output)
	}

	// calc deltas for hidden layers
	for i := len(n.layers) - 2; i > 0; i-- {
		for j := range n.layers[i].neurons {
			var d float64
			for k := 0; k < len(n.layers[i+1].neurons); k++ {
				d += n.layers[i+1].neurons[k].delta * n.layers[i+1].neurons[k].weights[j]
			}

			n.layers[i].neurons[j].delta = d * n.da(n.layers[i].neurons[j].output)
		}
	}

	// update weights and biases
	for layerIndex := 1; layerIndex < len(n.layers); layerIndex++ {
		for neuronIndex := range n.layers[layerIndex].neurons {
			for weightIndex := range n.layers[layerIndex].neurons[neuronIndex].weights {
				n.layers[layerIndex].neurons[neuronIndex].weights[weightIndex] += eta * n.layers[layerIndex].neurons[neuronIndex].delta * n.layers[layerIndex-1].neurons[weightIndex].activation
			}
			n.layers[layerIndex].neurons[neuronIndex].bias += eta * n.layers[layerIndex].neurons[neuronIndex].delta
		}
	}

}

// NewNeuralNetwork returns a randomized network with the specified
// number of layers and neurons. Example: []uint{6, 3, 2, 2} would
// create a network with 6 input neurons, 2 hidden layers with
// 3 and 2 neurons and an output layer with 2 neurons
func NewNeuralNetwork(layers []uint) *NeuralNetwork {
	if len(layers) < 2 {
		panic("invalid number of layers")
	}

	n := NeuralNetwork{}

	n.layers = make([]layer, 0, len(layers))

	for i, layer := range layers {
		var synapses uint
		if i == 0 {
			// first layer doesn't neet synapses as is is only used as input
			synapses = 0
		} else {
			synapses = layers[i-1]
		}

		n.layers = append(n.layers, newLayer(synapses, layer))
	}
	n.a = sigmoid
	n.da = sigmoidD
	return &n
}

// LoadNeuralNetwork returns a previously dumped network
// based on the encoded data from the io.Reader r
func LoadNeuralNetwork(r io.Reader) (*NeuralNetwork, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	var dn networkDump
	err = json.Unmarshal(b, &dn)
	if err != nil {
		return nil, err
	}

	var n NeuralNetwork
	n.layers = make([]layer, len(dn.Layers), len(dn.Layers))
	for i, l := range dn.Layers {
		n.layers[i].neurons = make([]neuron, len(l.Neurons), len(l.Neurons))
		for j, cn := range l.Neurons {
			n.layers[i].neurons[j] = neuron{cn.Weights, cn.Bias, 0.0, 0.0, 0.0}
		}
	}

	n.a = sigmoid
	n.da = sigmoidD
	return &n, err
}

// Train will feedforward the input and then backpropagate to reduce
// error on output using learning rate eta
func (n *NeuralNetwork) Train(input, output []float64, eta float64) {
	if len(input) != len(n.layers[0].neurons) {
		panic("invalid input length")
	}
	if len(output) != len(n.layers[len(n.layers)-1].neurons) {
		panic("invalid output length")
	}
	if eta == 0.0 {
		panic("invalid learning rate")
	}

	n.feedforward(input)
	n.backpropagate(output, eta)
}

// Predict will feedforward the input and then return the prediction
func (n *NeuralNetwork) Predict(input []float64) []float64 {

	if len(input) != len(n.layers[0].neurons) {
		panic("invalid input length")
	}

	n.feedforward(input)

	output := make([]float64, 0, len(n.layers[len(n.layers)-1].neurons))

	for _, e := range n.layers[len(n.layers)-1].neurons {
		output = append(output, e.activation)
	}
	return output
}

// Dump marshals the neural network to json and writes it
// to the specified io.Writer w
func (n *NeuralNetwork) Dump(w io.Writer) error {

	nd := &networkDump{}
	nd.Layers = make([]layerDump, len(n.layers), len(n.layers))
	for i := 0; i < len(nd.Layers); i++ {
		nd.Layers[i].Neurons = make([]neuronDump, len(n.layers[i].neurons), len(n.layers[i].neurons))
		for j, neuron := range n.layers[i].neurons {
			nd.Layers[i].Neurons[j] = neuronDump{neuron.weights, neuron.bias}
		}
	}

	b, err := json.Marshal(nd)
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}
