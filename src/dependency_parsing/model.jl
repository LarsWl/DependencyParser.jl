export Model

using Flux


#=
  Input layer dimension: d x 45
  45 coming from 18 words + 18 pos tags + 9 labels
  hidden_layer dimension: 45 x 200
  output_layer: 200 x 30
=#
struct Model
  hidden_layer::Dense
  output_layer::Dense
end;

function Model()
  hidden_layer = Dense(45, 200)
  output_layer
end

function activation(layer::Dense)

end
