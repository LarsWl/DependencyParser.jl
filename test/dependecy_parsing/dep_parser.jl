using DependencyParser.DependencyParsing

@testset "DepParser" begin
  parser = build_dep_parser()
  config = build_configuration()

  @testset "Test form batch" begin
    stack = config.stack |> collect

    correct_batch = [
      stack[begin] |> word_id -> parser.model.word_ids[config.sentence.tokens[word_id].name],
      stack[begin + 1] |> word_id -> parser.model.word_ids[config.sentence.tokens[word_id].name],
      parser.model.word_ids[ROOT_TOKEN],
      config.buffer[begin] |> word_id -> parser.model.word_ids[config.sentence.tokens[word_id].name],
      config.buffer[begin + 1] |> word_id -> parser.model.word_ids[config.sentence.tokens[word_id].name],
      config.buffer[begin + 2] |> word_id -> parser.model.word_ids[config.sentence.tokens[word_id].name],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[config.sentence.tokens[1].name],
      parser.model.word_ids[config.sentence.tokens[3].name],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      parser.model.word_ids[NULL_TOKEN],
      stack[begin] |> word_id -> parser.model.tag_ids[config.sentence.pos_tags[word_id].name],
      stack[begin + 1] |> word_id -> parser.model.tag_ids[config.sentence.pos_tags[word_id].name],
      parser.model.tag_ids[ROOT_TOKEN],
      config.buffer[begin] |> word_id -> parser.model.tag_ids[config.sentence.pos_tags[word_id].name],
      config.buffer[begin + 1] |> word_id -> parser.model.tag_ids[config.sentence.pos_tags[word_id].name],
      config.buffer[begin + 2] |> word_id -> parser.model.tag_ids[config.sentence.pos_tags[word_id].name],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[config.sentence.pos_tags[1].name],
      parser.model.tag_ids[config.sentence.pos_tags[3].name],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.tag_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids["SBJ"],
      parser.model.label_ids["IOBJ"],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
      parser.model.label_ids[NULL_TOKEN],
    ]

    batch = form_batch(parser, config)
    @test form_batch(parser, config) == correct_batch
  end
end