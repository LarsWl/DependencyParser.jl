export POSTagger

using ..Units
using TextModels

struct POSTagger <: AbstractPOSTagger
  pos_model::TextModels.PerceptronTagger

  POSTagger() = new(TextModels.PerceptronTagger(true));
end

function (tagger::POSTagger)(sentence_tokens::Vector{String})
  tagger.pos_model(sentence_tokens)
end