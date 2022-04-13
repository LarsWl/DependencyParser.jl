using DependencyParser.DependencyParsing

function build_model(settings::Settings)
  word_ids = Dict{String, Integer}(
    "here"=>1,
    "some"=>2,
    "words"=>3
  )
  tag_ids = Dict{String, Integer}(
    "TAG1"=>4,
    "TAG2"=>5,
    "TAG3"=>6
  )
  label_ids = Dict{String, Integer}(
    "LABEL1"=>7,
    "LABEL2"=>8
  )

  embeddings = rand(Float32, 8, settings.embeddings_size)
  hidden_bias = rand(Float32, settings.hidden_size)
  hidden_layer = Dense(settings.embeddings_size * settings.batch_size, settings.hidden_size, bias=hidden_bias)
  softmax_layer = Dense(settings.hidden_size, 10)

  Model(embeddings, hidden_layer, softmax_layer, word_ids, tag_ids, label_ids)
end