export load_connlu_file

using TextAnalysis

function load_connlu_file(filename::String)
  open(f -> read(f, String), filename) |> 
    text -> split(text, r"# sent_id.+\n") |>
    sentences_data -> filter(sentences_data -> length(sentences_data) > 1, sentences_data) |>
    sentences_data -> map(sentence_data -> convert(sentence_data), sentences_data)
end

function convert(sentence_data)
  root = TreeNode(0)
  lines = split(sentence_data, "\n")
  words = filter(word -> (length(word) > 1) && !(occursin(r"(text = )|(\d+-\d+)", word)), lines) .|> 
      (word -> split(word, r"\t")) |>
      (words -> map(word -> [word[1], word[2], word[7], word[8]], words))

  nodes::Vector{TreeNode} = map(words) do word
    dependent_word_number = parse(Int32, word[1])
    head_word_number = parse(Int32, word[3])
    dependency = word[4]

    TreeNode(dependent_word_number, String(dependency), head_word_number)
  end

  tree = DependencyTree(root, nodes, length(nodes))
  string_doc = findfirst(line -> occursin(r"# text = ", line), lines) |> 
    index -> StringDocument(replace(lines[index], r"# text = " => ""))
  

  [string_doc, tree]
end