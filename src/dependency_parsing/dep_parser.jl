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

  while !is_terminal(config)``
    transition = predict_transition(parser, config)
    println(transition)
    execute_transition(parser, config, transition) || return config.tree
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

  write_to_file!(model, model_file)
end

function train!(train_file::String, model::Model, system::ParsingSystem, model_file::String)
  println("Parse train file...")
  connlu_sentences = load_connlu_file(train_file)
  settings = Settings()

  train!(model, system, settings, connlu_sentences)

  write_to_file!(model, model_file)
end

# Temp code for test

# system = ArcEager.ArcEagerSystem()
# train_file = "/Users/admin/education/materials/UD_English-ParTUT/en_partut-ud-train.conllu"
# embeddings_file = "/Users/admin/education/materials/fastvec.vec"
# model_file = "tmp/test6.txt"

# connlu_sentences = load_connlu_file(train_file)
# settings = Settings()
# model = cache_data((args...) -> Model(args[1], args[2], args[3], args[4]), "tmp/cache", "model_cache_2", settings, system, embeddings_file, connlu_sentences)

# sort(collect(model.label_ids), by=pair->pair[end]) |>
#       pairs -> map(pair -> pair[begin], pairs) |>
#       labels -> set_labels!(system, labels)

# train!(train_file, model, system,model_file)
# write_to_file!(model, model_file)

function predict_transition(parser::DepParser, config::Configuration)
  predict_transition(parser.model, parser.settings, parser.parsing_system, config)
end