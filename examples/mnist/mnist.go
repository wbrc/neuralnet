package main

import (
	"log"
	"math/rand"

	"github.com/petar/GoMNIST"
	"github.com/wbrc/neuralnet"
)

func main() {

	// load mnist data
	log.Print("load mnist dataset")
	train, test, err := GoMNIST.Load("./src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err.Error())
	}

	// you can fiddle around with # of hidden layers and # of neurons here
	n := neuralnet.NewNeuralNetwork([]uint{784, 30, 10})
	lrate := 0.35

	// train the network
	log.Print("train the network. 10 epochs")
	for epoch := 0; epoch < 10; epoch++ {
		log.Printf("Epoch %d", epoch+1)
		// randomly go through the dataset
		perm := rand.Perm(train.Count())
		for _, i := range perm {
			image, label := train.Get(i)

			input := make([]float64, 784, 784)
			output := make([]float64, 10, 10)
			for i, b := range image {
				input[i] = float64(b) / 255.0
			}
			output[label] = 1.0

			n.Train(input, output, lrate)
		}

		// reduce learning rate by 10%
		lrate *= 0.9
	}

	right := 0
	wrong := 0

	// test the network
	log.Print("test network")
	for i := 0; i < test.Count(); i++ {
		image, label := test.Get(i)

		input := make([]float64, 784, 784)
		for i, b := range image {
			input[i] = float64(b) / 255.0
		}
		prediction := n.Predict(input)
		if uint(label) == getNum(prediction) {
			right++
		} else {
			wrong++
		}
	}

	p := float64(right) * 100.0 / (float64(right) + float64(wrong))

	log.Printf("predicted %3.2f%% of numbers right", p)

}

func getNum(output []float64) uint {
	max := 0.0
	var index int
	for i, e := range output {
		if e > max {
			index = i
			max = e
		}
	}
	return uint(index)
}
