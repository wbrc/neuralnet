package main

import (
	"fmt"

	"github.com/wbrc/neuralnet"
)

func main() {

	// 2 input neurons, 1 hidden layer with 3 neurons and
	// output layer with 1 neuron
	n := neuralnet.NewNeuralNetwork([]uint{2, 3, 1})

	// the dataset
	data := [][][]float64{
		[][]float64{
			[]float64{0.0, 0.0},
			[]float64{0.0},
		},
		[][]float64{
			[]float64{0.0, 1.0},
			[]float64{1.0},
		},
		[][]float64{
			[]float64{1.0, 0.0},
			[]float64{1.0},
		},
		[][]float64{
			[]float64{1.0, 1.0},
			[]float64{0.0},
		},
	}

	// iterate dataset 5000 times
	for i := 0; i < 5000; i++ {
		for _, dataset := range data {
			n.Train(dataset[0], dataset[1], 0.25)
		}
	}

	// show predictions
	for _, dataset := range data {
		prediction := n.Predict(dataset[0])
		fmt.Printf("expected: %v --- got %v\n", dataset[1], prediction)
	}

}
