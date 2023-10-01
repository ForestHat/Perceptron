package main

import (
	"fmt"
	"math"
	"math/rand"
)

const lambda float64 = 0.1
const epochs int = 100000
const bias float64 = 1.0

func main() {
	var examples = [4][2]float64{{0, 1}, {1, 0}, {1, 1}, {0, 0}}
	var results = [4]float64{1, 1, 0, 0}

	weightsInputLayer, weightsOutputLayer := train(examples, results)

	var test = []float64{1, 0}

	var hiddenNeuronOne float64 = f(test[0]*weightsInputLayer[0][0] + test[1]*weightsInputLayer[1][0] + bias*weightsInputLayer[2][0])
	var hiddenNeuronTwo float64 = f(test[0]*weightsInputLayer[0][1] + test[1]*weightsInputLayer[1][1] + bias*weightsInputLayer[2][1])

	var resultNeuron float64 = f(hiddenNeuronOne*weightsOutputLayer[0] + hiddenNeuronTwo*weightsOutputLayer[1] + bias*weightsOutputLayer[2])

	fmt.Println(resultNeuron)
}

func train(examples [4][2]float64, results [4]float64) ([3][2]float64, [3]float64) {
	var weightsInputLayer = [3][2]float64{{rand.Float64(), rand.Float64()}, {rand.Float64(), rand.Float64()}, {rand.Float64(), rand.Float64()}}
	var weightsOutputLayer = [3]float64{rand.Float64(), rand.Float64(), rand.Float64()}

	for epoch := 0; epoch < epochs; epoch++ {
		for example := 0; example < 4; example++ {
			var firstExample = examples[example][0]
			var secondExample = examples[example][1]

			var NeuronFirstHiddenLayer float64 = f(firstExample*weightsInputLayer[0][0] + secondExample*weightsInputLayer[1][0] + bias*weightsInputLayer[2][0])
			var NeuronSecondHiddenLayer float64 = f(firstExample*weightsInputLayer[0][1] + secondExample*weightsInputLayer[1][1] + bias*weightsInputLayer[2][1])

			var resultNeuron float64 = f(NeuronFirstHiddenLayer*weightsOutputLayer[0] + NeuronSecondHiddenLayer*weightsOutputLayer[1] + bias*weightsOutputLayer[2])

			// Back propogation
			var err float64 = resultNeuron - results[example]
			var delta float64 = err * resultNeuron * (1 - resultNeuron)

			weightsOutputLayer[0] -= (lambda * delta * NeuronFirstHiddenLayer)
			weightsOutputLayer[1] -= (lambda * delta * NeuronSecondHiddenLayer)
			weightsOutputLayer[2] -= (lambda * delta * bias)

			var deltaFirstLayer float64 = (delta * weightsOutputLayer[0]) * NeuronFirstHiddenLayer * (1 - NeuronFirstHiddenLayer)
			var deltaSecondLayer float64 = (delta * weightsOutputLayer[1]) * NeuronSecondHiddenLayer * (1 - NeuronSecondHiddenLayer)

			weightsInputLayer[0][0] -= lambda * deltaFirstLayer * firstExample
			weightsInputLayer[0][1] -= lambda * deltaSecondLayer * firstExample

			weightsInputLayer[1][0] -= lambda * deltaFirstLayer * secondExample
			weightsInputLayer[1][1] -= lambda * deltaSecondLayer * secondExample

			weightsInputLayer[2][0] -= lambda * deltaFirstLayer * bias
			weightsInputLayer[2][1] -= lambda * deltaSecondLayer * bias
		}
	}

	return weightsInputLayer, weightsOutputLayer
}

func f(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}
