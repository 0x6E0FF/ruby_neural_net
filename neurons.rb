#! ruby

require 'matrix'
require 'json'
require_relative 'load_db'

def sigmoid(v)
    1.0 / (1.0 + Math.exp(-v))
end

def sigmoid_prime(v)
    sigmoid(v) * (1 - sigmoid(v))
end

def cost(output, expected)
    Vector[*output.zip(expected).map{|o,e| e - o }].magnitude ** 2
end

class Network
    attr_accessor :sizes, :biases, :weights
    def initialize(sizes)
        @sizes = sizes
        @nb_layers = sizes.size
        @biases = sizes[1..-1].map{ |nb_neurons| 
            # each layer
            nb_neurons.times.map { 
                # each neuron 
                rand(-1.0..1.0) 
            }
        }
        @weights = sizes[0..-2].zip(sizes[1..-1]).map { |nb_inputs, nb_neurons|
            # each layer
            nb_neurons.times.map {
                # each neuron
                nb_inputs.times.map {
                    # each connection
                    rand(-1.0..1.0)
                }
            }
        }
    end

    def compute(inputs)
        @activations = [inputs]
        @weighted_sums = []
        @biases.zip(@weights).each do |b, w|
            # each layer 
            #    - b[nb_neuron]
            #    - w[nb_neuron][nb_inputs]
            @weighted_sums << w.zip(b).map { |wn, bn|
                # each neuron
                Vector[*wn].dot(Vector[*@activations[-1]]) + bn
            }
            @activations << @weighted_sums[-1].map{ |v| sigmoid(v) }
        end
        @activations[-1]
    end

    def backpropagation(inputs, expected_output)

        # biases gradients array 
        biases_grad = Array.new(biases.size)
        # weights gradients array
        weights_grad = sizes[0..-2].zip(sizes[1..-1]).map { |nb_inputs, nb_neurons|
            Array.new(nb_neurons) {
                Array.new(nb_inputs, 0.0)
            }
        }

        # compute output layer
        output = compute(inputs)

        # compute gradient of cost function 
        cost_gradient = output.zip(expected_output).map{ |o, e| o - e }

        # compute error on output layer
        error = cost_gradient.zip(@weighted_sums[-1]).map{ |c, z| c * sigmoid_prime(z) }

        biases_grad[-1] = error
        weights_grad[-1] = error.map { |e| @activations[-2].map { |a| a * e } }

        puts JSON.pretty_generate @weighted_sums

        # propagate error backward on each layer
        # (@biases.size - 1).downto(1).each do |layer_i|
        (2...@nb_layers).map{|l| -l }.each do |layer_i|
            puts layer_i
            puts 'act size '+  @activations[layer_i].size.to_s
            error = @activations[layer_i].size.times.map { |n| # each neuron of current layer
                # puts "layer=#{layer_i} : error.size=#{error.size} neuron##{n}"
                error.map.with_index {|e,i| 
                    @weights[layer_i+1][i][n] * e 
                }.reduce(:+) * sigmoid_prime(@weighted_sums[layer_i][n])
            }
            biases_grad[layer_i] = error
            weights_grad[layer_i] = error.map { |e| @activations[layer_i-1].map { |a| a * e } }
        end
        [biases_grad, weights_grad]
    end

    def to_s
        JSON.pretty_generate({"biases" => @biases, "weights" => @weights})
    end
end

# n = Network.new([28*28,32,32,10])
# i = 0
# load_training(1).each do |img, label|
#     expected = Array.new(10, 0.0)
#     expected[label] = 1.0

#     i += 1
# end

n = Network.new([4,3,5,2])
grad_b,grad_w = n.backpropagation([15, 48, 12, 50], [1,2])
puts JSON.pretty_generate(n.biases)
puts JSON.pretty_generate(grad_b)
puts "-------------------------------------------"
# puts JSON.pretty_generate(n.weights)
# puts JSON.pretty_generate(grad_w)