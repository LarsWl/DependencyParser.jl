export DepParser
export execute_transition, form_batch, predict_transition

using ..DependencyParser.Units
import .ArcEager: execute_transition

using TextAnalysis

struct DepParser <: AbstractDepParser
  settings::Settings
  parsing_system::ParsingSystem
  model::Model

  function DepParser(settings::Settings, model_file::String)
    model = Model(model_file)
    system =  ArcEager.ArcEagerSystem()
    sort(collect(model.label_ids), by=pair->pair[end]) |>
      pairs -> map(pair -> pair[begin], pairs) |>
      labels -> set_labels!(system, labels)

    new(settings, system, model)
  end

  function DepParser(settings::Settings, model::Model, system::ParsingSystem)
    system =  ArcEager.ArcEagerSystem()
    sort(collect(model.label_ids), by=pair->pair[end]) |>
      pairs -> map(pair -> pair[begin], pairs) |>
      labels -> set_labels!(system, labels)

    new(settings, system, model)
  end
end

function (parser::DepParser)(tokens_with_tags)
  sentence = Sentence(tokens_with_tags)
  config = Configuration(sentence)
  transitions_number = 0

  while !is_terminal(config)
    transition = predict_transition(parser, config)
    transition === nothing && break
    execute_transition(parser, config, transition)

    transitions_number += 1
    transitions_number > LIMIT_TRANSITIONS_NUMBER && break
  end

  config.tree
end

function execute_transition(parser::DepParser, config::Configuration, transition::Transition)
  execute_transition(config, transition, parser.parsing_system)
end

function train!(system::ParsingSystem, train_file::String, embeddings_file::String, model_file::String)
  println("Parse train file...")
  connlu_sentences = load_connlu_file(train_file)
  settings = Settings()

  println("Build initial model...")
  model = Model(settings, system, embeddings_file, connlu_sentences)

  train!(model, settings, system, connlu_sentences)
end

function train!(train_file::String, test_file::String, results_file::String, model::Model, system::ParsingSystem, model_file::String)
  println("Parse train file...")
  connlu_sentences = load_connlu_file(train_file)
  test_sentences = load_connlu_file(test_file)
  settings = Settings(embeddings_size=100)


  training_context = TrainingContext(
    system,
    settings,
    train_file,
    connlu_sentences,
    test_sentences,
    test_file,
    results_file,
    model_file,
    beam_coef = 0.05
  )

  train!(model, training_context)
end

# Temp code for test

function default_train()
  system = ArcEager.ArcEagerSystem()
  train_file = "F:\\ed_soft\\parser_materials\\UD_English-EWT-master\\en_ewt-ud-train.conllu"
  test_file = "F:\\ed_soft\\parser_materials\\UD_English-EWT-master\\en_ewt-ud-dev.conllu"
  embeddings_file = "F:\\ed_soft\\parser_materials\\model.txt"
  model_file = "tmp/model_b3000_adam_fl2_e100_beam_new_oracle"
  results_file = "tmp/results_b3000_adam_fl2_e100_beam_new_oracle"

  connlu_sentences = load_connlu_file(train_file)
  settings = Settings(embeddings_size=100)
  model = cache_data((args...) -> Model(args...), "tmp/cache", "model_cache_e100-ewt", settings, system, embeddings_file, connlu_sentences)
  # model.gpu_available = false
  enable_cuda(model)

  # model = Model(model_file * "_last.txt")

  sort(collect(model.label_ids), by=pair->pair[end]) |>
        pairs -> map(pair -> pair[begin], pairs) |>
        labels -> set_labels!(system, labels)

  train!(train_file, test_file, results_file, model, system, model_file)
end

function predict_transition(parser::DepParser, config::Configuration)
  predict_transition(parser.model, parser.settings, parser.parsing_system, config)
end