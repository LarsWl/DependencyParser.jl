export Sentence

mutable struct Sentence
  tokens::Vector{Token}
  pos_tags::Vector{PosTag}
  length::Integer

  Sentence() = new()
  function Sentence(tokens_with_tags::Vector{Tuple{String, String}})
    tokens = map(index -> Token(index, first(tokens_with_tags[index])), 1:length(tokens_with_tags))
    tags = map(tokens_with_tag -> PosTag(second(tokens_with_tag)), tokens_with_tags)

    new(tokens, tags, length(tokens_with_tags))
  end
end