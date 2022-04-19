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
    
    new(settings, ArcEager.ArcEagerSystem(), model)
  end

  DepParser(settings::Settings, model::Model, system::ParsingSystem) = new(settings, system, model)
end

function (parser::DepParser)(tokens_with_tags)
  println(tokens_with_tags)
end

function execute_transition(parser::DepParser, transition::Transition)
  execute_transition(parser.config, transition, parser.parsing_system)
end

function train!(system::ParsingSystem, train_file::String, embeddings_file::String, model_file::String)
  println("Parse train file...")
  connlu_sentences = load_connlu_file(train_file)
  settings = Settings()

  println("Build initial model...")
  model = Model(settings, system, embeddings_file, connlu_sentences)

  train!(model, settings, connlu_sentences)

  write_to_file!(model, model_file)
end

function train!(train_file::String, model::Model, model_file::String)
  println("Parse train file...")
  connlu_sentences = load_connlu_file(train_file)
  settings = Settings()

  train!(model, settings, connlu_sentences)

  write_to_file!(model, model_file)
end

# system = ArcEager.ArcEagerSystem()
# train_file = "/Users/admin/education/materials/UD_English-ParTUT/en_partut-ud-train.conllu"
# embeddings_file = "/Users/admin/education/materials/fastvec.vec"
# model_file = "tmp/test.txt"

# connlu_sentences = load_connlu_file(train_file)
# settings = Settings()
# model = Model(settings, system, embeddings_file, connlu_sentences)

# train!(train_file, model, model_file)

function predict_transition(parser::DepParser, config::Configuration)
  form_batch(parser.model, parser.settings, config) |> 
    batch -> predict(parser.model, batch) |>
    findmax |>
    max_score_wiht_index -> parser.parsing_system.transitions[max_score_wiht_index[end]]
end