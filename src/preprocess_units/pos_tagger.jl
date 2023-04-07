# export POSTagger

# using ..Units
# using TextModels
# using BSON

# MISSING_TAG = "-MISSING-"

# struct POSTagger <: AbstractPOSTagger
#   pos_model::TextModels.PerceptronTagger

#   function POSTagger(model_file)
#     tagger = TextModels.PerceptronTagger();
#     pretrained = BSON.load(model_file)

#     tagger.model.weights = pretrained[:weights]
#     tagger.tagdict = pretrained[:tagdict]
#     tagger.classes = tagger.model.classes = Set(pretrained[:classes])
#     println("loaded successfully")

#     new(tagger)
#   end
# end

# function (tagger::POSTagger)(sentence_tokens::Vector{String})
#   map(tagger.pos_model(sentence_tokens)) do (token, tag)
#     tag = tag === missing ? MISSING_TAG : tag

#     (token, tag)
#   end |> vec -> convert(Vector{Tuple{String, String}}, vec)
# end