export DepParser
export execute_transition

using ..Units
import .ArcEager: execute_transition

struct DepParser <: AbstractDepParser
  config::Configuration
  parsing_system::ParsingSystem
  model

  DepParser() = new()
  DepParser(sentence::Sentence) = new(Configuration(sentence), ArcEager.ArcEagerSystem())
end

function (parser::DepParser)(tokens_with_tags)
  println(tokens_with_tags)
end

function execute_transition(parser::DepParser, transition::Transition)
  execute_transition(parser.config, transition, parser.parsing_system)
end
