export Sentence

mutable struct Sentence
  tokens::Vector{Token}
  pos_tags::Vector{PosTag}
  length::Integer
end