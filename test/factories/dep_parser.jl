using DependencyParser.DependencyParsing

function build_dep_parser()
  settings = Settings(embeddings_size = 4, hidden_size = 2)

  connlu_file = "test/fixtures/connlu_test.txt"
  emb_file = "test/fixtures/embeddings_test.vec"

  conllu_sentences = load_connlu_file(connlu_file)
  system = ArcEager.ArcEagerSystem()
  model = Model(settings, system , emb_file, conllu_sentences)

  DepParser(settings, model, system)
end

function build_correct_batch(parser::DepParser, config::Configuration)
  stack = config.stack |> collect

  Integer[
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
end