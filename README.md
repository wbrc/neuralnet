# neuralnet
Create simple feed forward networks in Go

[![GoDoc](https://godoc.org/github.com/goml/gobrain?status.svg)](https://godoc.org/github.com/wbrc/neuralnet)
[![Build Status](https://travis-ci.com/wbrc/neuralnet.svg?branch=master)](https://travis-ci.org/wbrc/neuralnet)
[![codecov](https://codecov.io/gh/wbrc/neuralnet/branch/master/graph/badge.svg)](https://codecov.io/gh/wbrc/neuralnet)
[![rcard](https://goreportcard.com/badge/github.com/wbrc/neuralnet)](https://goreportcard.com/report/github.com/wbrc/neuralnet)

## Getting Started
Create a network with two input neurons, one hidden layer with
three neurons and an output layer with one neuron:
```go
n := neuralnet.NewNeuralNetwork([]uint{2, 3, 1})
```
### Basic usage
Train the network:
```go
input := []float64{1.0, 1.0}
output := []float64{0.0}
n.Train(input, output, 0.25)
```

Predict the output:
```go
prediction := n.Predict(data)
```

### Persisting the network

Persist the network:
```go
file, _ := os.Create("network.json")
n.Dump(file)
```

Load persisted network:
```go
file, _ := os.Open("network.json")
n, _ = neuralnet.LoadNeuralNetwork(ifile)
```

For more detailed info see `examples` folder
