export ConnluSentence
export load_connlu_file

using TextAnalysis

struct ConnluSentence
  token_doc::TokenDocument
  string_doc::StringDocument
  pos_tags::Vector{String}
  gold_tree::DependencyTree
end

function load_connlu_file(filename::String)
  sentences = Vector{ConnluSentence}()

  open(filename) do file
    sentence_lines = []
  
    while !eof(file)
      line = readline(file)
  
      if !occursin(r"^\s*$", line)
        push!(sentence_lines, line)
      else
        convert(sentence_lines) |> sentence_data -> ConnluSentence(sentence_data...) |> sentence -> push!(sentences, sentence)
        sentence_lines = []
      end
    end
  end

  sentences
end

function convert(lines)
  root = TreeNode(0)
  words = filter(word -> (length(word) > 1) && !(occursin(r"^(# |(\d+\-\d+)|(\d+\.\d+))", word)), lines) .|> 
      (word -> split(word, r"\t")) |>
      (words -> map(word -> [word[1], word[2], word[4], word[7], word[8]], words))

  nodes::Vector{TreeNode} = map(words) do word
    dependent_word_number = parse(Int32, word[1])
    head_word_number = parse(Int32, word[4])
    dependency = uppercase(word[5])

    TreeNode(dependent_word_number, String(dependency), head_word_number)
  end

  tree = DependencyTree(root, nodes, length(nodes))
  token_doc = TokenDocument(map(word -> String(word[2]), words))
  string_doc = findfirst(line -> occursin(r"# text = ", line), lines) |> 
    index -> StringDocument(replace(lines[index], r"# text = " => ""))
  pos_tags = map(word -> String(word[3]), words)

  [token_doc, string_doc, pos_tags, tree]
end
