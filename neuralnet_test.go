package neuralnet

import (
	"bytes"
	"math/rand"
	"testing"
	"time"
)

const EPSILON = 0.00000001

func init() {
	rand.Seed(time.Now().Unix())
}

func TestPanicCreate(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Code did not panic")
		}
	}()

	NewNeuralNetwork([]uint{3})
}

func TestCreate(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code did panic")
		}
	}()

	NewNeuralNetwork([]uint{5, 3, 2})
}

func TestPanicTrainInput(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Code did not panic")
		}
	}()

	n := NewNeuralNetwork([]uint{2, 3, 1})
	n.Train([]float64{0.34}, []float64{1.0}, 1.2)
}

func TestPanicTrainOutput(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Code did not panic")
		}
	}()

	n := NewNeuralNetwork([]uint{2, 3, 1})
	n.Train([]float64{0.34, 0.32}, []float64{1.0, 0.44}, 1.2)
}

func TestPanicTrainLrate(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Code did not panic")
		}
	}()

	n := NewNeuralNetwork([]uint{2, 3, 1})
	n.Train([]float64{0.34, 0.75}, []float64{1.0}, 0.0)
}

func TestPanicPredict(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Code did not panic")
		}
	}()

	n := NewNeuralNetwork([]uint{2, 3, 1})
	n.Predict([]float64{0.34, 0.75, 0.11})
}

func TestTrainPredict(t *testing.T) {
	n := NewNeuralNetwork([]uint{3, 2, 1})

	data := genRandomTestData()
	for _, d := range data {
		n.Train(d[0], d[1], 0.25)
	}

	for _, d := range data {
		if len(n.Predict(d[0])) != len(d[1]) {
			t.Errorf("output length mismatch")
		}
	}
}

func TestDumpLoad(t *testing.T) {
	n := NewNeuralNetwork([]uint{3, 2, 1})

	preDumpOut := n.Predict([]float64{1.0, 1.0, 1.0})
	b := make([]byte, 0, 1000)
	buf := bytes.NewBuffer(b)
	n.Dump(buf)
	n, err := LoadNeuralNetwork(buf)
	if err != nil {
		t.Errorf("No error should occur: %s", err.Error())
	}

	postDumpOut := n.Predict([]float64{1.0, 1.0, 1.0})
	if len(preDumpOut) != len(postDumpOut) {
		t.Errorf("len mismatch")
	}
	for i := range preDumpOut {
		if !floatEquals(preDumpOut[i], postDumpOut[i]) {
			t.Errorf("output mismatch")
		}
	}
}

func genRandomTestData() [][][]float64 {
	nrOfDataSets := rand.Intn(1000) + 500
	d := make([][][]float64, 0, nrOfDataSets)
	for i := 0; i < nrOfDataSets; i++ {
		d = append(d, [][]float64{
			{rand.Float64(), rand.Float64(), rand.Float64()},
			{rand.Float64()},
		})
	}

	return d
}

func floatEquals(a, b float64) bool {
	if (a-b) < EPSILON && (b-a) < EPSILON {
		return true
	}
	return false
}
