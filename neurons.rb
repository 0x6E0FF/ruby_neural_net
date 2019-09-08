#! ruby

require 'matrix'
require 'json'

class Network
    attr_accessor :sizes, :biases, :weights
    def initialize(sizes)
        @sizes = sizes
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
        @biases.zip(@weights).each do |b, w|
            inputs = w.zip(b).map { |wn, bn|
                Vector[*wn].dot(Vector[*inputs]) + bn
            }
        end
        inputs
    end

    def to_s
        JSON.pretty_generate({"biases" => @biases, "weights" => @weights})
    end
end

n = Network.new([2,3,1])
puts JSON.pretty_generate({"biases" => n.biases, "weights" => n.weights, "outputs" => n.compute([4, 5])})

