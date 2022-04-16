using DependencyParser.DependencyParsing

function build_model(settings::Settings)
  word_ids = Dict{String, Integer}(
    "here"=>1,
    "some"=>2,
    "words"=>3,
    DependencyParser.DependencyParsing.UNKNOWN_TOKEN => 4,
    DependencyParser.DependencyParsing.NULL_TOKEN => 5,
    DependencyParser.DependencyParsing.ROOT_TOKEN => 6
  )
  tag_ids = Dict{String, Integer}(
    "TAG1"=>7,
    "TAG2"=>8,
    "TAG3"=>9,
    DependencyParser.DependencyParsing.UNKNOWN_TOKEN => 10,
    DependencyParser.DependencyParsing.NULL_TOKEN => 11,
    DependencyParser.DependencyParsing.ROOT_TOKEN => 12,
  )
  label_ids = Dict{String, Integer}(
    "LABEL1"=>13,
    "LABEL2"=>14,
    DependencyParser.DependencyParsing.NULL_TOKEN => 15
  )

  embeddings = rand(Float32, 15, settings.embeddings_size)
  hidden_bias = rand(Float32, settings.hidden_size)
  hidden_layer = Dense(settings.embeddings_size * settings.batch_size, settings.hidden_size, bias=hidden_bias)
  softmax_layer = Dense(settings.hidden_size, 10)

  Model(embeddings, hidden_layer, softmax_layer, word_ids, tag_ids, label_ids)
end