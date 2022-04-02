using .Units
using .PreprocessUnits
using ..DependencyParsing

struct Pipeline
  sentence_splitter::AbstractSentenceSplitter
  tokenizer::AbstractTokenizer
  pos_tagger::AbstractPOSTagger
  dep_parser::AbstractDepParser

  Pipeline(;
    sentence_splitter =  SentenceSplitter(), 
    tokenizer = Tokenizer(), 
    pos_tagger = POSTagger(), 
    dep_parser = DepParser()
  ) = new(sentence_splitter, tokenizer, pos_tagger, dep_parser)
end

function (pipeline::Pipeline)(text)
  pipeline.sentence_splitter(text) .|>
    sentence -> pipeline.tokenizer(sentence) |>
    sentence_tokens -> pipeline.pos_tagger(sentence_tokens) |>
    sentence_tokens_with_tags -> pipeline.dep_parser(sentence_tokens_with_tags);
end


