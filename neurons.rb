#! ruby

require 'matrix'
require 'json'
require_relative 'load_db'

def debug(s)
    puts JSON.pretty_generate s
    puts "***********"
end

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

    def backpropagation(input, expected_output)

        # biases gradients array 
        biases_grad = Array.new(@biases.size)
        # weights gradients array
        weights_grad = Array.new(@weights.size)

        # compute output layer 
        output = compute(input)

        # compute gradient of cost function ∇aC
        cost_gradient = output.zip(expected_output).map{ |o, e| o - e }

        # compute error on output layer
        # -> δL=∇aC⊙σ′(zL)
        error = cost_gradient.zip(@weighted_sums[-1]).map{ |c, z| c * sigmoid_prime(z) }

        biases_grad[-1] = error
        weights_grad[-1] = error.map { |e| @activations[-2].map { |a| a * e } }

        # propagate error backward on each layer
        (2...@nb_layers).map{|l| -l }.each do |layer_i|
            error = @activations[layer_i].size.times.map { |n| # each neuron of current layer

                # compute row n of new error vector (of size the number of neurons of current layer)
                #   -> δl=((wl+1)Tδl+1) ⊙ σ′(zl)
                error.map.with_index {|e,i| 
                    @weights[layer_i+1][i][n] * e 
                }.reduce(:+) * sigmoid_prime(@weighted_sums[layer_i][n])
            }
            biases_grad[layer_i] = error
            weights_grad[layer_i] = error.map { |e| @activations[layer_i-1].map { |a| a * e } }
        end
        [biases_grad, weights_grad]
    end

    def each_bias_index()
        @biases.size.times do |l| 
            @biases[l].size.times do |n|
                yield [l,n]
            end
        end
    end

    def each_weight_index()
        @weights.size.times do |l| 
            @weights[l].size.times do |n|
                @weights[l][n].size.times do |w|
                    yield [l,n,w]
                end
            end
        end
    end

    def train(batch, learning_rate)
        sum_grad_b = biases.map{ |l| Array.new(l.size, 0.0) }
        sum_grad_w = weights.map{ |l| l.map {|n| Array.new(n.size, 0.0 )} }
        batch.each do |input, expected_output|
            grad_b, grad_w = backpropagation(input, expected_output)
            each_bias_index() { |l,n| sum_grad_b[l][n] += grad_b[l][n] }
            each_weight_index() { |l,n,w| sum_grad_w[l][n][w] += grad_w[l][n][w] }
        end
        each_bias_index() do |l,n| 
            @biases[l][n] = @biases[l][n] - (sum_grad_b[l][n] * learning_rate) / batch.size 
        end
        each_weight_index() do |l,n,w| 
            @weights[l][n][w] = @weights[l][n][w] - (sum_grad_w[l][n][w] * learning_rate) / batch.size 
        end
    end

    def to_s
        JSON.pretty_generate({"biases" => @biases, "weights" => @weights})
    end
    
end

def run(network, inputs, verbose=false)
    i = 0
    nb_success = 0
    inputs.each do |img, label|
        found = network.compute(img).each_with_index.max[1]
        puts "##{i} expected=#{label}: found=#{found}" if verbose
        nb_success+=1 if label == found 
        i += 1
    end
    nb_success
end

n = Network.new([28*28,32,32,10])

train_set = load_training(1000)
test_set = load_tests(1000)

test_sample = test_set.sample(100)
# puts "init: #{run(n, test_sample)} / #{test_sample.size}"
#  train network and test on same sample to observe improvment
500.times do |i|
    puts "##{i.to_s.rjust(5,'0')} TRAIN"
    batch = train_set.sample(10).map { |img, label| [img, Array.new(10) { |i| i == label ? 1.0 : 0.0 } ] }
    
    n.train(batch, 2.0);
    if i % 50 == 0
        puts "##{i.to_s.rjust(5,'0')} #{run(n, test_sample)} / #{test_sample.size}"
    end
end

# n = Network.new([4,3,5,2])
# grad_b,grad_w = n.backpropagation([15, 48, 12, 50], [1,2])
# puts JSON.pretty_generate(n.biases)
# puts JSON.pretty_generate(grad_b)
# puts "-------------------------------------------"
# puts JSON.pretty_generate(n.weights)
# puts JSON.pretty_generate(grad_w)
# p n.compute([15, 48, 12, 50])

# train_data = [
#     [[15, 48, 12, 50], [1,2]],
#     [[15, 48, 12, 50], [1,2]],
#     [[15, 48, 12, 50], [1,2]],
#     [[15, 48, 12, 50], [1,2]],
#     [[15, 48, 12, 50], [1,2]]
# ]
# n.train(train_data,0.1)
# p n.compute([15, 48, 12, 50])
