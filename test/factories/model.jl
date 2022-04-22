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

  embeddings = rand(Float64, 15, settings.embeddings_size)
  hidden_bias = rand(Float64, settings.hidden_size)
  hidden_layer = Dense(settings.embeddings_size * settings.batch_size, settings.hidden_size, bias=hidden_bias)
  output_layer = Dense(settings.hidden_size, 10)

  Model(embeddings, hidden_layer, output_layer, word_ids, tag_ids, label_ids)
end

function build_correct_batch(model::Model, config::Configuration)
  stack = config.stack |> collect

  Integer[
    stack[begin] |> word_id -> model.word_ids[config.sentence.tokens[word_id].name],
    stack[begin + 1] |> word_id -> model.word_ids[config.sentence.tokens[word_id].name],
    model.word_ids[ROOT_TOKEN],
    config.buffer[begin] |> word_id -> model.word_ids[config.sentence.tokens[word_id].name],
    config.buffer[begin + 1] |> word_id -> model.word_ids[config.sentence.tokens[word_id].name],
    config.buffer[begin + 2] |> word_id -> model.word_ids[config.sentence.tokens[word_id].name],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[config.sentence.tokens[1].name],
    model.word_ids[config.sentence.tokens[3].name],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    model.word_ids[NULL_TOKEN],
    stack[begin] |> word_id -> model.tag_ids[config.sentence.pos_tags[word_id].name],
    stack[begin + 1] |> word_id -> model.tag_ids[config.sentence.pos_tags[word_id].name],
    model.tag_ids[ROOT_TOKEN],
    config.buffer[begin] |> word_id -> model.tag_ids[config.sentence.pos_tags[word_id].name],
    config.buffer[begin + 1] |> word_id -> model.tag_ids[config.sentence.pos_tags[word_id].name],
    config.buffer[begin + 2] |> word_id -> model.tag_ids[config.sentence.pos_tags[word_id].name],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[config.sentence.pos_tags[1].name],
    model.tag_ids[config.sentence.pos_tags[3].name],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.tag_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids["SBJ"],
    model.label_ids["IOBJ"],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
    model.label_ids[NULL_TOKEN],
  ]
end