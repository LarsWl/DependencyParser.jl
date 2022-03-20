export SentenceSplitter

using ..Units
using WordTokenizers

struct SentenceSplitter <: AbstractSentenceSplitter end;

function (splitter::SentenceSplitter)(text)
  WordTokenizers.split_sentences(text) .|> StringDocument
end
