export Tokenizer

using ..Units
using TextAnalysis

struct Tokenizer <: AbstractTokenizer end;

function (tokenizer::Tokenizer)(sentence::StringDocument)
  tokens(sentence)
end
