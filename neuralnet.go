package neuralnet

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
)

type neuron struct {
	Weights    []float64 // the weights of the inputs
	output     float64   // the sum of the weighted inputs
	activation float64   // the output with applied activation function
	delta      float64   // delta of the neuron
}

type layer struct {
	Neurons []neuron
}

type NeuralNetwork struct {
	Layers []layer
	a      func(float64) float64 // activation function
	da     func(float64) float64 // derivation of activatin function
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func sigmoidD(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

func newLayer(synapses, neurons uint) layer {
	l := layer{}

	l.Neurons = make([]neuron, 0, neurons)

	for i := 0; i < int(neurons); i++ {
		l.Neurons = append(l.Neurons, newNeuron(synapses))
	}

	return l
}

func newNeuron(synapses uint) neuron {
	n := neuron{}

	n.Weights = make([]float64, synapses, synapses)
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*2.0 - 1.0
	}

	return n
}

func (n *NeuralNetwork) feedforward(input []float64) {

	// feed input data into input layer
	for i := range input {
		n.Layers[0].Neurons[i].activation = input[i]
	}

	// feed forward through each layer
	for layerIndex := 1; layerIndex < len(n.Layers); layerIndex++ {
		for neuronIndex := range n.Layers[layerIndex].Neurons {
			var output float64
			for weightIndex := range n.Layers[layerIndex].Neurons[neuronIndex].Weights {
				output += n.Layers[layerIndex-1].Neurons[weightIndex].activation * n.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex]
			}
			n.Layers[layerIndex].Neurons[neuronIndex].output = output
			n.Layers[layerIndex].Neurons[neuronIndex].activation = n.a(output)
		}
	}
}

func (n *NeuralNetwork) backpropagate(targetOutput []float64, eta float64) {

	// calc deltas of output layer
	for i := range n.Layers[len(n.Layers)-1].Neurons {
		n.Layers[len(n.Layers)-1].Neurons[i].delta = (targetOutput[i] - n.Layers[len(n.Layers)-1].Neurons[i].activation) * n.da(n.Layers[len(n.Layers)-1].Neurons[i].output)
	}

	// calc deltas for hidden layers
	for i := len(n.Layers) - 2; i > 0; i-- {
		for j := range n.Layers[i].Neurons {
			var d float64
			for k := 0; k < len(n.Layers[i+1].Neurons); k++ {
				d += n.Layers[i+1].Neurons[k].delta * n.Layers[i+1].Neurons[k].Weights[j]
			}

			n.Layers[i].Neurons[j].delta = d * n.da(n.Layers[i].Neurons[j].output)
		}
	}

	// update weights
	for layerIndex := 1; layerIndex < len(n.Layers); layerIndex++ {
		for neuronIndex := range n.Layers[layerIndex].Neurons {
			for weightIndex := range n.Layers[layerIndex].Neurons[neuronIndex].Weights {
				n.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] += eta * n.Layers[layerIndex].Neurons[neuronIndex].delta * n.Layers[layerIndex-1].Neurons[weightIndex].activation
			}
		}
	}

}

// returns a neural network; specify the number of neurons
// per layer using a uint slice. []uint{3, 2, 2, 1} results
// in a network with 3 inputs, 2 hidden layers with 2 neurons
// each and an output layer with one neuron
func NewNeuralNetwork(layers []uint) *NeuralNetwork {
	if len(layers) < 2 {
		panic("invalid number of layers")
	}

	n := NeuralNetwork{}

	n.Layers = make([]layer, 0, len(layers))

	for i, layer := range layers {
		var synapses uint
		if i == 0 {
			// first layer doesn't neet synapses as is is only used as input
			synapses = 0
		} else {
			synapses = layers[i-1]
		}

		n.Layers = append(n.Layers, newLayer(synapses, layer))
	}
	n.a = sigmoid
	n.da = sigmoidD

	return &n
}

func LoadNeuralNetwork(r io.Reader) (*NeuralNetwork, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	var n NeuralNetwork
	n.da = sigmoidD
	n.a = sigmoid
	err = json.Unmarshal(b, &n)
	return &n, err
}

// will feedforward the input and then backpropagate to reduce
// error on output using learning rate eta
func (n *NeuralNetwork) Train(input, output []float64, eta float64) {
	if len(input) != len(n.Layers[0].Neurons) {
		panic("invalid input length")
	}
	if len(output) != len(n.Layers[len(n.Layers)-1].Neurons) {
		panic("invalid output length")
	}
	if eta == 0.0 {
		panic("invalid learning rate")
	}

	n.feedforward(input)
	n.backpropagate(output, eta)
}

// feedforward the input and returns the prediction
func (n *NeuralNetwork) Predict(input []float64) []float64 {

	n.feedforward(input)

	output := make([]float64, 0, len(n.Layers[len(n.Layers)-1].Neurons))

	for _, e := range n.Layers[len(n.Layers)-1].Neurons {
		output = append(output, e.activation)
	}
	return output
}

// Marshals the neural network to json and writes it
// to the specified writer
func (n *NeuralNetwork) Dump(w io.Writer) error {
	b, err := json.Marshal(n)
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}
