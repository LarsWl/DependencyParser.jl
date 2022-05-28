using .Units
using .PreprocessUnits
using .DependencyParsing
using DependencyParserTest

struct Pipeline
  sentence_splitter::AbstractSentenceSplitter
  tokenizer::AbstractTokenizer
  pos_tagger::AbstractPOSTagger
  dep_parser::AbstractDepParser

  Pipeline(;
    sentence_splitter =  SentenceSplitter(), 
    tokenizer = Tokenizer(), 
    pos_tagger = POSTagger("tmp\\pos_model_2.bson"), 
    dep_parser = DepParser(DependencyParsing.Settings(), "tmp/model_b3000_adam_fl2_e100_beam_nt_2_best.txt")
  ) = new(sentence_splitter, tokenizer, pos_tagger, dep_parser)
end

function (pipeline::Pipeline)(text)
  completed_trees = Dict();

  mutex = ReentrantLock()

  parse_tree = (sentence_tokens_with_tags, sentence_index) -> begin
    tree = pipeline.dep_parser(sentence_tokens_with_tags)

    lock(mutex) do
      completed_trees[sentence_index] = tree
    end
  end

  sentences = pipeline.sentence_splitter(text)

  Threads.@threads for i in 1:length(sentences)
    tokens_with_tags = pipeline.tokenizer(sentences[i]) |> sentence_tokens -> pipeline.pos_tagger(sentence_tokens)

    parse_tree(tokens_with_tags, i)
  end

  sort(completed_trees |> collect, by = pair -> pair[begin]) |> pairs -> map(pair -> pair[end], pairs)
end

function pipeline_test(connlu_file)
  text = DependencyParserTest.Converting.extract_sentences_from_conllu(connlu_file)
  text = replace(text, r"[éè]" => "e")
  text = replace(text, r"[ô]" => "o")
  pipe = Pipeline()

  trees = pipe(text)

  test_file = "tmp/test_parsed_trees.txt"

  open(test_file, "w") do file
    foreach(trees) do tree
      tree_text = DependencyParser.DependencyParsing.convert_to_string(tree)

      if tree_text != "\n"
        write(file, tree_text)
        write(file, "\n\n")
      end
    end
  end

  conllu_source = DependencyParserTest.Sources.ConlluSource(connlu_file)
  parser_source = DependencyParserTest.Sources.CoreNLPSource(test_file)

  DependencyParserTest.Benchmark.test_accuracy(conllu_source, parser_source)
end
