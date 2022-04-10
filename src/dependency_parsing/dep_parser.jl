export DepParser
export execute_transition

using ..DependencyParser.Units
import .ArcEager: GoldState
import .ArcEager: execute_transition, zero_cost

using TextAnalysis
using TextModels

struct DepParser <: AbstractDepParser
  settings::Settings
  config::Configuration
  parsing_system::ParsingSystem
  model::Model

  function DepParser(model_file::String)
    model = Model(model_file)
    
    new(undef, ArcEager.ArcEagerSystem(), model)
  end
end

function (parser::DepParser)(tokens_with_tags)
  println(tokens_with_tags)
end

function execute_transition(parser::DepParser, transition::Transition)
  execute_transition(parser.config, transition, parser.parsing_system)
end

function train!(system::ParsingSystem, train_file::String, embeddings_file::String)
  iterations = 10
  conllu_sentences = load_connlu_file(train_file)
  settings = Settings()

  model = Model(settings, system, embeddings_file, conllu_sentences)

  for i = 1:iterations
    for (string_doc, gold_tree) in corpus
      sentence = tokens(string_doc.text) |> pos |> Sentence

      config = Configuration(sentence)
      gold_state = GoldState(gold_tree, config)

      preticted_transition = predict(model, config)
      zero_cost_transitions = zero_cost(gold_state)

      if predicted_transition in zero_cost_transitions
        update_model(model)
      end

      transition = zero_cost_transitions[begin]

      execute_transition(config, transition, system)
    end
  end

  write!(model)
end