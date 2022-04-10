using DependencyParser.Core

# Sentence: He wrote her a letter.
function build_sentence()
  words = ["He", "wrote", "her", "a", "letter", "."]
  tags = ["PRP", "VBN", "PRP", "DT", "NN", "JJ"]
  
  Sentence(zip(words, tags) |> collect)
end
