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
    beam_coef = 0.2
  )

  train!(model, training_context)
end

# Temp code for test

function default_train()
  system = ArcEager.ArcEagerSystem()
  train_file = "/home/egor/UD_English-EWT/en_ewt-ud-train.conllu"
  test_file = "/home/egor/UD_English-EWT/en_ewt-ud-dev.conllu"
  model_file = "tmp/model_v1.bson"
  results_file = "tmp/results_v1"

  connlu_sentences = load_connlu_file(train_file)
  settings = Settings(embeddings_size=100)
  model = Model(settings, system, connlu_sentences)
  # model.gpu_available = false
  # enable_cuda(model)

  # model = Model(model_file * "_last.txt")

  sort(collect(model.label_ids), by=pair->pair[end]) |>
        pairs -> map(pair -> pair[begin], pairs) |>
        labels -> set_labels!(system, labels)

  train!(train_file, test_file, results_file, model, system, model_file)
end

function predict_transition(parser::DepParser, config::Configuration)
  predict_transition(parser.model, parser.settings, parser.parsing_system, config)
end

function init_dataset()
  fetch_cache("ewt_processed_dataset") do 
    system = ArcEager.ArcEagerSystem()
    train_file = "/home/egor/UD_English-EWT/en_ewt-ud-train.conllu"
    test_file = "/home/egor/UD_English-EWT/en_ewt-ud-dev.conllu"
    model_file = "tmp/model_v1.bson"
    results_file = "tmp/results_v1"

    connlu_sentences = load_connlu_file(train_file)
    settings = Settings(embeddings_size=100)
    model = Model(settings, system, connlu_sentences)
    # model.gpu_available = false
    # enable_cuda(model)

    # model = Model(model_file * "_last.txt")

    sort(collect(model.label_ids), by=pair->pair[end]) |>
      pairs -> map(pair -> pair[begin], pairs) |>
      labels -> set_labels!(system, labels)

    test_sentences = load_connlu_file(test_file)

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

    cache_file_path = "tmp/cache/dataset.jld2"
    dataset_parts = Flux.DataLoader(connlu_sentences, batchsize=500)

    dataset_index = 1
    for dataset_part in dataset_parts
      GC.gc()

      processed_dataset_part = build_dataset(model, dataset_part, training_context)

      cache_key = "dataset_part_$dataset_index"

      write_cache(cache_key, processed_dataset_part;file_path=cache_file_path)

      dataset_index += 1
    end

    @info "Merging dataset parts..."
    dataset = []
    dataset_index = 1
    for i in 1:20
      cache_key = "dataset_part_$i"
      part = read_cache(cache_key,;file_path=cache_file_path)
      dataset = vcat(dataset, part)
    end

    dataset
  end
end