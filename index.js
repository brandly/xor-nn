var { last, range, sum, reverse, zip } = require('lodash')

function buildLayer (inputs, neurons) {
  return range(0, neurons).map(n => ({
    weights: range(0, inputs).map(_ => Math.random())
  }))
}

function buildNetwork (inputs, hidden, outputs) {
  return [
    buildLayer(inputs, hidden),
    buildLayer(hidden, outputs)
  ]
}

function activate (weights, inputs) {
  return sum(weights.map((_, i) => weights[i] * inputs[i]))
}

function sigmoid (n) {
  return 1 / (1 + Math.exp(-n))
}

function sigmoidDerivative (n) {
  return n * (1 - n)
}

function forwardPropagate (network, inputs) {
  network.forEach(layer => {
    const nextInputs = []
    layer.forEach(neuron => {
      const activation = activate(neuron.weights, inputs)
      neuron.output = sigmoid(activation)
      nextInputs.push(neuron.output)
    })
    inputs = nextInputs
  })
  return inputs
}

function backPropagateError (network, ideal) {
  const backwards = reverse(network.slice(0))
  backwards.forEach((layer, i) => {
    const errors = []

    if (i === 0) {
      layer.forEach((neuron, j) => {
        errors.push(ideal[j] - neuron.output)
      })
    } else {
      layer.forEach((_, j) => {
        let error = 0
        backwards[i - 1].forEach((neuron, k) => {
          error += (neuron.weights[j] * neuron.delta)
        })
        errors.push(error)
      })
    }

    layer.forEach((neuron, j) => {
      neuron.delta = errors[j] * sigmoidDerivative(neuron.output)
    })
  })
}

function updateWeights (network, inputs, rate) {
  network.forEach((layer, i) => {
    if (i !== 0) {
      inputs = network[i - 1].map(n => n.output)
    }
    layer.forEach(neuron => {
      inputs.forEach((input, j) => {
        neuron.weights[j] += rate * neuron.delta * input
      })
    })
  })
}

// mean squared error
function cost (ideal, actual) {
  const mse = sum(ideal.map((_, i) =>
    Math.pow(ideal[i] - actual[i], 2)
  ))
  return mse / actual.length
}

function train (network, data, rate, epochs) {
  range(0, epochs).forEach(epoch => {
    let error = 0
    data.forEach(([input, ideal]) => {
      const actual = forwardPropagate(network, input)
      error += cost(ideal, actual)
      backPropagateError(network, ideal)
      updateWeights(network, input, rate)
    })
    console.log(`epoch ${epoch}, error ${error}`)
  })
}

// XOR
const data = [
  [[0, 0], [0]],
  [[1, 0], [1]],
  [[0, 1], [1]],
  [[1, 1], [0]]
]

class Network {
  constructor (sizes) {
    this.sizes = sizes
    // TODO:
    // this.biases
    this.weights =
      zip(sizes.slice(0, sizes.length - 1), sizes.slice(1)).map(
        // making a input x nodes matrix of random weights
        ([input, nodes]) => range(nodes).map(() =>
          range(input).map(() => Math.random())
        )
      )
  }

  getActivations (input) {
    const activations = []
    this.weights.forEach(layer => {
      const nextInput = []

      layer.forEach(neuronWeights => {
        const activation = activate(neuronWeights, input)
        const output = sigmoid(activation)
        nextInput.push(output)
      })

      activations.push(nextInput)
      input = nextInput
    })
    return activations
  }

  feedforward (input) {
    return last(this.getActivations(input))
  }

  // backprop ()
}

const hiddenNodes = 5
const net = new Network([data[0][0].length, hiddenNodes, data[0][1].length])
console.log(JSON.stringify(net, null, 2))
console.log(JSON.stringify(net.feedforward(data[0][0]), null, 2))
// net.train(data, 0.5, 10000)

const network = buildNetwork(data[0][0].length, hiddenNodes, data[0][1].length)
// train(network, data, 0.5, 10000)
// console.log(JSON.stringify(network, null, 2))

// data.forEach(([input, ideal]) => {
//   const real = network.forwardPropagate(input)
//   console.log(`input ${input}, expected ${ideal}, real ${real.map(n => Math.round(n))} (${real})`)
// })
