#! ruby

require 'erb'

require_relative 'neurons.rb'

n = Marshal.load(File.open(ARGV[0], "rb") { |f| f.read() })

template = <<EOF
<%-n.biases.each_with_index do |b, l|-%>
static float biases_<%=l-%>[<%=b.size-%>] = {
    <%=b.join("F,\n\t")%>
};
<%-end-%>

<%-n.weights.each_with_index do |wl, l|-%>
<%-wl.each_with_index do |w, n|-%>
static float weights_<%=l-%>_<%=n-%>[<%=w.size-%>]=
{
    /* each connection */
    <%=w.join("F,\n\t\t")%>
};
<%-end-%>

static float *weights_<%=l-%>[<%=wl.size-%>] = {
    /* each neuron of layer <%= l -%>*/
    <%-wl.each_with_index do |w, n|-%>
    weights_<%=l-%>_<%=n-%>,
    <%-end-%>
};
<%-end-%>

struct {
    float *biases[<%=n.biases.size-%>];
    float **weights[<%=n.biases.size-%>];
    int nb_neurons[<%=n.biases.size-%>];
} net = {
    { 
    <%-n.biases.size.times do |l|-%>
        biases_<%=l-%>,
    <%-end-%>
    },
    {
    <%-n.weights.size.times do |l|-%>
        weights_<%=l-%>,
    <%-end-%>
    },
    {
    <%-n.biases.each do |b|-%>
        <%=b.size-%>,
    <%-end-%>
    }
};
EOF

puts ERB.new(template, nil, "-").result(binding)
