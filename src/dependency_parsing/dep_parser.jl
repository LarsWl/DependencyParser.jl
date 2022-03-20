export DepParser

using ..Units

struct DepParser <: AbstractDepParser
  config::Configuration
  parsing_system::ParsingSystem
  model

  DepParser() = new()
  DepParser(sentence::Vector{Tuple{String, String}}) = new(Configuration(sentence), ArcEager.ArcEagerSystem())
end

function (parser::DepParser)(tokens_with_tags)
  println(tokens_with_tags)
end
