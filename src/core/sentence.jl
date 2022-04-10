export Sentence

mutable struct Sentence
  tokens::Vector{Token}
  pos_tags::Vector{PosTag}
  length::Integer

  Sentence() = new()
  function Sentence(tokens_with_tags::Vector{Tuple{String, String}})
    tokens = map(index -> Token(index, tokens_with_tags[index][begin]), 1:length(tokens_with_tags))
    tags = map(tokens_with_tag -> PosTag(tokens_with_tag[begin + 1]), tokens_with_tags)

    new(tokens, tags, length(tokens_with_tags))
  end
end